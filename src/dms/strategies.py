"""Execution strategies: how a task actually gets answered.

Routing is one strategy (decide first, call once). Cascading is a different one
(call cheap, check, escalate only if needed) and it does not fit the Router
contract, because its decision happens *after* generation rather than before.

That ex-ante / post-hoc split is the useful mental model, and it is the framing
the ETH cascade-routing work formalises: routing needs a good prediction of
quality *before* you spend anything; cascading needs a good judgement of quality
*after* you have already paid for one answer. The optimal system uses both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from dms.client import ModelClient, SimulationHint
from dms.grading import grade
from dms.routers.base import Decision, Outcome, Router, Spend, answer_with
from dms.workload import Task

VERIFIER_SYSTEM = (
    "You check answers. Given a question and a proposed answer, reply with "
    "exactly one word: yes if the answer is correct, no if it is not."
)


class Strategy(Protocol):
    name: str

    def run(self, task: Task, client: ModelClient) -> Outcome: ...


@dataclass(frozen=True, slots=True)
class RoutedStrategy(Strategy):
    """Route once, answer once. Router cost is included in the outcome."""

    router: Router
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.router.name)

    def run(self, task: Task, client: ModelClient) -> Outcome:
        decision: Decision = self.router.route(task, client)
        answer, answer_spend = answer_with(client, decision.model, task)

        return Outcome(
            task_id=task.id,
            difficulty=task.difficulty,
            strategy=self.name,
            answer=answer,
            correct=grade(answer, task.expected, task.grader),
            chosen_model=decision.model,
            why=decision.why,
            spends=(*decision.spends, answer_spend),
        )


@dataclass(frozen=True, slots=True)
class CascadeStrategy(Strategy):
    """Answer cheap, self-verify, escalate on doubt. The FrugalGPT shape.

    Three costs, and the bench charges all three:

    1. the cheap answer -- paid even when it is thrown away,
    2. the verification call -- paid on every single task,
    3. the strong answer -- paid on escalation.

    So a cascade that escalates on most tasks costs *more* than always calling
    the strong model. The break-even is governed by the verifier, not the models:
    a verifier that cannot tell a wrong answer from a right one turns the cheap
    attempt into pure waste. FrugalGPT's reported range across datasets is
    59-98% savings; the 98% figure is one narrow classification set, and the
    spread between those two numbers is almost entirely verifier quality.
    """

    weak_model: str = "claude-haiku-4-5"
    strong_model: str = "claude-opus-5"
    verify: bool = True
    name: str = "cascade"

    def run(self, task: Task, client: ModelClient) -> Outcome:
        answer, weak_spend = answer_with(client, self.weak_model, task, role="answer")
        spends: list[Spend] = [weak_spend]

        weak_is_right = grade(answer, task.expected, task.grader)
        accept, verify_spend, why = self._judge(task, answer, weak_is_right, client)
        if verify_spend is not None:
            spends.append(verify_spend)

        final_answer, chosen = answer, self.weak_model
        if not accept:
            final_answer, escalation_spend = answer_with(
                client, self.strong_model, task, role="escalation"
            )
            spends.append(escalation_spend)
            chosen = self.strong_model

        return Outcome(
            task_id=task.id,
            difficulty=task.difficulty,
            strategy=self.name,
            answer=final_answer,
            correct=grade(final_answer, task.expected, task.grader),
            chosen_model=chosen,
            why=why,
            spends=tuple(spends),
        )

    def _judge(
        self, task: Task, answer: str, weak_is_right: bool, client: ModelClient
    ) -> tuple[bool, Spend | None, str]:
        """Decide whether to accept the cheap answer."""
        if not self.verify:
            return True, None, "no verification -- accepted blind"

        verdict = client.complete(
            model=self.weak_model,
            prompt=f"Question:\n{task.prompt}\n\nProposed answer:\n{answer}",
            system=VERIFIER_SYSTEM,
            max_tokens=8,
            # ASSUMPTION (simulate only): the verifier is exactly as reliable as
            # the weak model is on a medium task. Its errors are therefore
            # symmetric -- it misses bad answers and false-alarms on good ones at
            # the same rate. Real verifiers are rarely symmetric; measure yours.
            hint=SimulationHint(
                difficulty="medium", expected="yes" if weak_is_right else "no"
            ),
        )
        spend = Spend(
            model=verdict.model,
            usage=verdict.usage,
            role="verify",
            latency_ms=verdict.latency_ms,
            simulated=verdict.simulated,
        )
        accept = "yes" in verdict.text.strip().lower()
        return accept, spend, f"verifier said {'accept' if accept else 'escalate'}"


@dataclass(frozen=True, slots=True)
class OracleStrategy(Strategy):
    """Perfect routing: the cheapest model that would actually get this right.

    Unbuildable -- it reads the answer key. It is here because it is the only
    honest ceiling. Every real router should be reported as a fraction of the
    gap between random routing and this, and the remaining gap is the headroom
    that router research is actually fighting over. RouterArena's finding is that
    nobody gets close: routers are systematically over-cautious, sending work to
    the expensive model that the cheap one would have handled.
    """

    ladder: tuple[str, ...] = ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5")
    name: str = "oracle"
    _cache: dict[str, str] = field(default_factory=dict, compare=False)

    def run(self, task: Task, client: ModelClient) -> Outcome:
        for model in self.ladder:
            answer, spend = answer_with(client, model, task)
            if grade(answer, task.expected, task.grader):
                return Outcome(
                    task_id=task.id,
                    difficulty=task.difficulty,
                    strategy=self.name,
                    answer=answer,
                    correct=True,
                    chosen_model=model,
                    why=f"cheapest model that succeeds: {model}",
                    spends=(spend,),  # only the winning call is charged
                )

        # Nothing in the ladder got it right; report the strongest attempt.
        answer, spend = answer_with(client, self.ladder[-1], task)
        return Outcome(
            task_id=task.id,
            difficulty=task.difficulty,
            strategy=self.name,
            answer=answer,
            correct=False,
            chosen_model=self.ladder[-1],
            why="no model in the ladder answered correctly",
            spends=(spend,),
        )
