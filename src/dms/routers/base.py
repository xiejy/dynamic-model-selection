"""Router and strategy contracts.

The critical modelling decision is here: a strategy reports **every** call it
made, tagged by role. That is what lets the bench charge a router's own tokens
to the strategy that used it, and charge a cascade for the cheap attempt it threw
away. Most published savings figures quietly omit both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from dms.client import ModelClient, SimulationHint
from dms.usage import UsageRecord
from dms.workload import Task

# Tier names are deliberately the same three LiteLLM's Auto Router ships with,
# so the audience can map this onto something deployable.
TIER_MODELS: dict[str, str] = {
    "simple": "claude-haiku-4-5",
    "medium": "claude-sonnet-5",
    "complex": "claude-opus-5",
}

ANSWER_SYSTEM = (
    "You are answering short developer questions in a benchmark. "
    "Reply with the answer only -- no preamble, no explanation, no punctuation "
    "beyond what the answer itself needs."
)

# Two settings applied uniformly to every answering call, for stated reasons:
#
# ANSWER_MAX_TOKENS is deliberately generous. On Claude Opus 5 thinking is ON by
# default and `max_tokens` caps thinking PLUS response text, so a tight cap
# truncates the answer -- which grades as WRONG and would quietly understate the
# expensive model. Never let the token cap masquerade as a capability gap.
#
# ANSWER_EFFORT is `low` because these are short-answer recall and single-step
# reasoning tasks. Anthropic's own guidance is to sweep effort downward from
# xhigh and stop where evals stop holding; `low` is the right end of that sweep
# for this workload. The client drops it for models that reject the parameter.
ANSWER_MAX_TOKENS = 2048
ANSWER_EFFORT = "low"


@dataclass(frozen=True, slots=True)
class Spend:
    """One billable call, tagged by what it was for."""

    model: str
    usage: UsageRecord
    role: str  # router | answer | verify | escalation
    latency_ms: float = 0.0
    simulated: bool = False


@dataclass(frozen=True, slots=True)
class Decision:
    """A router's choice, plus whatever it cost to make it."""

    model: str
    why: str
    spends: tuple[Spend, ...] = ()
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class Outcome:
    """Everything the bench needs to score one task under one strategy."""

    task_id: str
    difficulty: str
    strategy: str
    answer: str
    correct: bool
    chosen_model: str
    why: str
    spends: tuple[Spend, ...] = field(default_factory=tuple)

    @property
    def simulated(self) -> bool:
        """True if any call behind this outcome was synthetic rather than measured."""
        return any(spend.simulated for spend in self.spends)

    @property
    def latency_ms(self) -> float:
        return sum(spend.latency_ms for spend in self.spends)

    @property
    def router_spends(self) -> tuple[Spend, ...]:
        return tuple(spend for spend in self.spends if spend.role == "router")

    @property
    def escalated(self) -> bool:
        return any(spend.role == "escalation" for spend in self.spends)


class Router(Protocol):
    """Chooses a model for a prompt, possibly at a cost."""

    name: str

    def route(self, task: Task, client: ModelClient) -> Decision: ...


def answer_with(
    client: ModelClient, model: str, task: Task, role: str = "answer"
) -> tuple[str, Spend]:
    """Run one answering call and return its text plus its billable record."""
    call = client.complete(
        model=model,
        prompt=task.prompt,
        system=ANSWER_SYSTEM,
        max_tokens=ANSWER_MAX_TOKENS,
        effort=ANSWER_EFFORT,
        hint=SimulationHint(difficulty=task.difficulty, expected=task.expected),
    )
    return call.text, Spend(
        model=call.model,
        usage=call.usage,
        role=role,
        latency_ms=call.latency_ms,
        simulated=call.simulated,
    )
