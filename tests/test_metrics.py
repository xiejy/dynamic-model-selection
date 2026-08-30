"""Metrics invariants -- the accounting rules that keep the numbers honest."""
import json
from decimal import Decimal

import pytest

from dms.bench import run_bench
from dms.client import Mode, ModelClient
from dms.metrics import (
    beats_random,
    cost_saving_vs,
    interpolate_random,
    pareto_frontier,
    performance_gap_recovered,
    summarise,
)
from dms.pricing import PriceBook
from dms.replay import FixtureStore
from dms.routers.base import Outcome, Spend
from dms.strategies import CascadeStrategy, RoutedStrategy
from dms.routers.baseline import AlwaysRouter
from dms.routers.llm_classifier import LLMClassifierRouter
from dms.usage import UsageRecord
from dms.workload import Workload


@pytest.fixture(scope="module")
def book() -> PriceBook:
    return PriceBook.load()


@pytest.fixture(scope="module")
def workload() -> Workload:
    return Workload.load()


def _client(book: PriceBook, tmp_path=None) -> ModelClient:
    return ModelClient(mode=Mode.SIMULATE, book=book, store=FixtureStore("/tmp/unused"))


def _outcome(*spends: Spend, correct: bool = True) -> Outcome:
    return Outcome(
        task_id="t",
        difficulty="easy",
        strategy="s",
        answer="a",
        correct=correct,
        chosen_model=spends[-1].model if spends else "claude-opus-5",
        why="",
        spends=spends,
    )


# ---------------------------------------------------------------- cost attribution


def test_router_tokens_are_billed_to_the_strategy(book: PriceBook) -> None:
    """The single most-omitted line item in published routing savings."""
    outcome = _outcome(
        Spend("claude-haiku-4-5", UsageRecord(input_tokens=500, output_tokens=5), "router"),
        Spend("claude-haiku-4-5", UsageRecord(input_tokens=100, output_tokens=50), "answer"),
    )

    result = summarise("s", [outcome], book)

    assert result.router_cost_usd > 0
    assert result.cost_usd > result.router_cost_usd
    # Total must include the router call, not just the answer.
    answer_only = book.cost_usd(
        UsageRecord(input_tokens=100, output_tokens=50), "claude-haiku-4-5"
    )
    assert result.cost_usd > answer_only


def test_a_discarded_cheap_attempt_is_counted_as_waste(book: PriceBook) -> None:
    """A cascade that escalates paid for the cheap answer AND the strong one."""
    outcome = _outcome(
        Spend("claude-haiku-4-5", UsageRecord(input_tokens=100, output_tokens=50), "answer"),
        Spend("claude-haiku-4-5", UsageRecord(input_tokens=100, output_tokens=5), "verify"),
        Spend("claude-opus-5", UsageRecord(input_tokens=100, output_tokens=50), "escalation"),
    )

    result = summarise("s", [outcome], book)

    assert result.escalations == 1
    assert result.wasted_cost_usd > 0
    assert 0 < result.overhead_share < 1


def test_an_accepted_cheap_answer_wastes_only_the_verification(book: PriceBook) -> None:
    outcome = _outcome(
        Spend("claude-haiku-4-5", UsageRecord(input_tokens=100, output_tokens=50), "answer"),
        Spend("claude-haiku-4-5", UsageRecord(input_tokens=100, output_tokens=5), "verify"),
    )

    result = summarise("s", [outcome], book)

    assert result.escalations == 0
    assert result.wasted_cost_usd > 0  # verification is paid on every task


def test_summarise_rejects_an_empty_strategy(book: PriceBook) -> None:
    with pytest.raises(ValueError, match="no outcomes"):
        summarise("s", [], book)


# --------------------------------------------------------------------------- PGR


def test_pgr_anchors_at_zero_for_weak_and_one_for_strong(book: PriceBook) -> None:
    weak = summarise("w", [_outcome(_s(), correct=False)], book)
    strong = summarise("s", [_outcome(_s(), correct=True)], book)

    assert performance_gap_recovered(weak, weak, strong) == 0.0
    assert performance_gap_recovered(strong, weak, strong) == 1.0


def test_pgr_is_not_clamped_when_a_strategy_beats_the_strong_model(
    book: PriceBook,
) -> None:
    """A cascade can exceed the strong model by catching its mistakes. Clamping
    that to 1.0 would hide a real result."""
    weak = summarise("w", [_outcome(_s(), correct=False), _outcome(_s(), correct=False)], book)
    strong = summarise("s", [_outcome(_s(), correct=True), _outcome(_s(), correct=False)], book)
    better = summarise("b", [_outcome(_s(), correct=True), _outcome(_s(), correct=True)], book)

    assert performance_gap_recovered(better, weak, strong) > 1.0


def test_identical_models_make_routing_free(book: PriceBook) -> None:
    same = summarise("x", [_outcome(_s(), correct=True)], book)

    assert performance_gap_recovered(same, same, same) == 1.0


# ------------------------------------------------------------------ random baseline


def test_interpolation_sits_between_the_sampled_points(book: PriceBook) -> None:
    curve = [
        summarise("r0", [_outcome(_s("claude-haiku-4-5"), correct=False)], book),
        summarise("r1", [_outcome(_s("claude-opus-5"), correct=True)], book),
    ]

    accuracy, cost = interpolate_random(curve, 0.5)

    assert 0.0 < accuracy < 1.0
    assert curve[0].cost_usd < cost < curve[1].cost_usd


def test_interpolation_needs_a_curve(book: PriceBook) -> None:
    with pytest.raises(ValueError, match="empty"):
        interpolate_random([], 0.5)


def test_beats_random_returns_both_deltas(book: PriceBook) -> None:
    curve = [
        summarise("r0", [_outcome(_s("claude-haiku-4-5"), correct=False)], book),
        summarise("r1", [_outcome(_s("claude-opus-5"), correct=True)], book),
    ]
    candidate = summarise("c", [_outcome(_s("claude-opus-5"), correct=True)], book)

    delta_accuracy, delta_cost = beats_random(candidate, curve)

    assert isinstance(delta_accuracy, float)
    assert isinstance(delta_cost, Decimal)


# ------------------------------------------------------------------------- pareto


def test_pareto_drops_strategies_that_are_worse_on_both_axes(book: PriceBook) -> None:
    cheap_good = summarise("cheap-good", [_outcome(_s("claude-haiku-4-5"), correct=True)], book)
    dear_bad = summarise("dear-bad", [_outcome(_s("claude-opus-5"), correct=False)], book)

    frontier = pareto_frontier([cheap_good, dear_bad])

    assert [r.name for r in frontier] == ["cheap-good"]


def test_saving_is_negative_when_a_strategy_costs_more(book: PriceBook) -> None:
    cheap = summarise("cheap", [_outcome(_s("claude-haiku-4-5"))], book)
    dear = summarise("dear", [_outcome(_s("claude-opus-5"))], book)

    assert cost_saving_vs(dear, cheap) < 0


# ------------------------------------------------------------------ end to end


def test_full_bench_runs_offline_and_charges_the_llm_router(
    book: PriceBook, workload: Workload
) -> None:
    client = _client(book)
    strategies = (
        RoutedStrategy(AlwaysRouter(model="claude-haiku-4-5")),
        RoutedStrategy(AlwaysRouter(model="claude-opus-5")),
        RoutedStrategy(LLMClassifierRouter()),
        CascadeStrategy(),
    )

    report = run_bench(workload, client, strategies, include_random_curve=False)

    assert report.simulated is True
    assert len(report.results) == 4
    assert report.by_name("llm-classifier").router_cost_usd > 0
    assert report.by_name("always:haiku").router_cost_usd == 0
    assert report.by_name("cascade").wasted_cost_usd > 0


def test_bench_is_reproducible(book: PriceBook, workload: Workload) -> None:
    strategies = (RoutedStrategy(AlwaysRouter(model="claude-haiku-4-5")),)

    first = run_bench(workload, _client(book), strategies, include_random_curve=False)
    second = run_bench(workload, _client(book), strategies, include_random_curve=False)

    assert first.results[0].cost_usd == second.results[0].cost_usd
    assert first.results[0].accuracy == second.results[0].accuracy


def _s(model: str = "claude-opus-5") -> Spend:
    return Spend(model, UsageRecord(input_tokens=100, output_tokens=50), "answer")


# ------------------------------------------------------- answer-key leak guards


def test_the_answer_key_never_reaches_a_live_call(book: PriceBook, tmp_path) -> None:
    """Integrity guard for the whole benchmark.

    CascadeStrategy computes `weak_is_right` by grading against the answer key so
    the SIMULATE path can fake a verifier verdict. That value must never reach a
    real request -- if it did, the cascade would be an oracle wearing a costume
    and every cost figure it produced would be meaningless.

    This asserts the structural property directly: capture every kwarg sent to
    the API in RECORD mode and confirm no expected answer appears in any of them.
    """
    from dms.client import Call, Mode
    from dms.workload import Task

    sent: list[dict] = []

    class RecordingAPI:
        class messages:
            @staticmethod
            def create(**kwargs):
                sent.append(kwargs)

                class R:
                    content = [type("B", (), {"type": "text", "text": "no"})()]
                    stop_reason = "end_turn"
                    stop_details = None
                    usage = type(
                        "U", (), {"input_tokens": 5, "output_tokens": 2}
                    )()

                return R()

    task = Task(
        id="leak", difficulty="hard", kind="k",
        prompt="What is the capital of France?",
        expected="ZZQQ_SECRET_ANSWER_ZZQQ", grader="exact_ci",
    )
    client = ModelClient(
        mode=Mode.RECORD, book=book, store=FixtureStore(tmp_path), api=RecordingAPI()
    )

    CascadeStrategy().run(task, client)

    assert sent, "no API calls were captured -- the guard would pass vacuously"
    blob = json.dumps(sent)
    assert task.expected not in blob, "the answer key reached a live request"


def test_simulation_hints_are_dropped_outside_simulate_mode(book: PriceBook, tmp_path) -> None:
    """The hint carries the expected answer. Only the simulator may see it."""
    from dms.client import Call, Mode, SimulationHint

    sent: list[dict] = []

    class RecordingAPI:
        class messages:
            @staticmethod
            def create(**kwargs):
                sent.append(kwargs)

                class R:
                    content = [type("B", (), {"type": "text", "text": "ok"})()]
                    stop_reason = "end_turn"
                    stop_details = None
                    usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})()

                return R()

    client = ModelClient(
        mode=Mode.RECORD, book=book, store=FixtureStore(tmp_path), api=RecordingAPI()
    )
    client.complete(
        model="claude-haiku-4-5",
        prompt="hello",
        hint=SimulationHint(difficulty="hard", expected="LEAKED_TOKEN_42"),
    )

    assert "LEAKED_TOKEN_42" not in json.dumps(sent)
