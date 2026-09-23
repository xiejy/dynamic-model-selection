"""Jev as a pre-request router. No network: the Jev call is injected."""
import json
from decimal import Decimal

import pytest

from dms.client import Mode, ModelClient
from dms.pricing import PriceBook
from dms.replay import FixtureStore
from dms.routers.jev import (
    HIGH_LEVEL,
    JEV_MODEL,
    LEVELS,
    JevRouter,
    nearest_level,
)
from dms.workload import Task

LOW, HIGH = "claude-haiku-4-5", "claude-opus-5"
TIERS = {"simple": LOW, "medium": HIGH, "complex": HIGH}


def _task(prompt: str = "What is 2+2?", expected: str = "4") -> Task:
    return Task(id="t1", difficulty="easy", kind="k", prompt=prompt,
                expected=expected, grader="exact_ci")


def _client() -> ModelClient:
    return ModelClient(mode=Mode.SIMULATE, book=PriceBook.load(), store=FixtureStore("/tmp/x"))


def _answer(score: float, confidence: float = 0.9, tokens: int = 800) -> dict:
    return {
        "model": JEV_MODEL,
        "answers": {
            "route": {"type": "score", "score": score, "confidence": confidence},
            "multi_step": {"type": "noul", "noul": 0.1},
            "trap": {"type": "noul", "noul": 0.1},
        },
        "usage": {"input_tokens": tokens, "output_tokens": 0},
    }


class Recorder:
    """Stands in for the Jev endpoint and records exactly what was sent."""

    def __init__(self, response: dict | Exception) -> None:
        self.response = response
        self.sent: list[dict] = []

    def __call__(self, state, questions, model):
        self.sent.append({"state": state, "questions": questions, "model": model})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


# ------------------------------------------------------------- the decision rule


@pytest.mark.parametrize(
    "score,level",
    [(0.0, 0), (0.49, 0), (0.5, 1), (1.49, 1), (1.5, 2), (2.5, 3), (3.0, 3)],
)
def test_score_rounds_half_up_not_bankers(score, level) -> None:
    """Python's round() sends 2.5 to 2 (banker's rounding). The documented cuts
    are at 0.5, 1.5, 2.5, so the router must round half up."""
    assert nearest_level(score, len(LEVELS)) == level


def test_the_decision_level_is_the_preregistered_one_not_a_fitted_one() -> None:
    """36 items with 6 positives cannot support a fitted threshold."""
    assert HIGH_LEVEL == 2
    assert len(LEVELS) == 4


def test_a_low_score_routes_to_the_cheap_model() -> None:
    decision = JevRouter(tiers=TIERS, asker=Recorder(_answer(0.3))).route(_task(), _client())

    assert decision.model == LOW


def test_a_high_score_routes_to_the_expensive_model() -> None:
    decision = JevRouter(tiers=TIERS, asker=Recorder(_answer(2.2))).route(_task(), _client())

    assert decision.model == HIGH


def test_the_boundary_sits_at_one_and_a_half() -> None:
    router = lambda s: JevRouter(tiers=TIERS, asker=Recorder(_answer(s)))  # noqa: E731

    assert router(1.49).route(_task(), _client()).model == LOW
    assert router(1.50).route(_task(), _client()).model == HIGH


def test_the_reason_records_score_level_and_confidence() -> None:
    why = JevRouter(tiers=TIERS, asker=Recorder(_answer(2.2, 0.83))).route(
        _task(), _client()
    ).why

    assert "2.20" in why and "level 2" in why and "0.83" in why


# ----------------------------------------------------------------- accounting


def test_the_jev_call_is_billed_to_the_router() -> None:
    """Routing is never free unless it spends nothing; charge what it spent."""
    decision = JevRouter(tiers=TIERS, asker=Recorder(_answer(0.3, tokens=800))).route(
        _task(), _client()
    )

    (spend,) = decision.spends
    assert spend.role == "router"
    assert spend.model == JEV_MODEL
    assert spend.usage.input_tokens == 800
    assert spend.usage.output_tokens == 0


def test_jev_is_priced_at_its_published_rate() -> None:
    """$0.042 per million input tokens, output free."""
    from dms.usage import UsageRecord

    cost = PriceBook.load().cost_usd(UsageRecord(input_tokens=1_000_000), JEV_MODEL)

    assert cost == Decimal("0.042")


# ------------------------------------------------------------ what leaves the box


def test_only_the_prompt_leaves_the_machine() -> None:
    """The expected answer and the difficulty label must never reach a third
    party -- the same guard the cascade has against answer-key leakage."""
    recorder = Recorder(_answer(0.3))
    task = _task(prompt="PROMPT_TEXT", expected="SECRET_EXPECTED_ANSWER")

    JevRouter(tiers=TIERS, asker=recorder).route(task, _client())

    blob = json.dumps(recorder.sent)
    assert recorder.sent[0]["state"] == "PROMPT_TEXT"
    assert "SECRET_EXPECTED_ANSWER" not in blob
    assert "easy" not in blob  # the difficulty label is not the state


def test_the_model_is_pinned_not_latest() -> None:
    """A cache keyed on jev-latest keeps serving answers from whatever model the
    alias used to mean."""
    recorder = Recorder(_answer(0.3))

    JevRouter(tiers=TIERS, asker=recorder).route(_task(), _client())

    assert recorder.sent[0]["model"] == "jev-1.13.0"


def test_the_question_bundle_is_one_score_and_two_diagnostic_nouls() -> None:
    questions = JevRouter(tiers=TIERS, asker=Recorder(_answer(0.3))).questions()

    assert questions["route"]["type"] == "score"
    assert questions["route"]["criteria"] == list(LEVELS)
    assert {questions["multi_step"]["type"], questions["trap"]["type"]} == {"noul"}


# --------------------------------------------------------------- failure modes


def test_a_transient_jev_failure_fails_toward_quality() -> None:
    """Jev down must not silently downgrade traffic to the cheap model."""
    decision = JevRouter(
        tiers=TIERS, asker=Recorder(RuntimeError("HTTP 503 after retries"))
    ).route(_task(), _client())

    assert decision.model == HIGH
    assert "503" in decision.why
    assert decision.spends == ()  # nothing billed for a call that failed


def test_a_missing_key_is_a_loud_configuration_error(monkeypatch) -> None:
    """Every request quietly routed high because a key was never set is exactly
    the silent failure this repo keeps finding. Refuse instead."""
    from dms.routers.jev import JevCredentialMissing, default_asker

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    with pytest.raises(JevCredentialMissing, match="TYPESAFE_API_KEY"):
        default_asker("state", {}, JEV_MODEL)


def test_an_unparseable_answer_fails_toward_quality() -> None:
    decision = JevRouter(tiers=TIERS, asker=Recorder({"answers": {}})).route(
        _task(), _client()
    )

    assert decision.model == HIGH
