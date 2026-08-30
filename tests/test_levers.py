"""Cost-lever arithmetic. These assertions are the talk's load-bearing claims."""
from decimal import Decimal

import pytest

from dms.levers.caching import (
    agent_loop_cost,
    agent_loop_usage,
    alternating_model_loop_cost,
    crossover_step,
    quadratic_growth,
    tension_scenarios,
)
from dms.levers.toolkit import context_edited_loop_usage, rank_levers
from dms.pricing import PriceBook

OPUS = "claude-opus-5"
HAIKU = "claude-haiku-4-5"


@pytest.fixture(scope="module")
def book() -> PriceBook:
    return PriceBook.load()


# ------------------------------------------------------------ quadratic cost growth


def test_billed_input_grows_quadratically_not_linearly() -> None:
    """20 steps at 1k tokens/step bills ~210k input tokens, not 20k."""
    records = agent_loop_usage(
        steps=20, system_tokens=0, per_step_tokens=1000, output_tokens=0, caching=False
    )

    billed = sum(record.prompt_tokens for record in records)

    assert billed == 210_000  # sum(1..20) * 1000
    assert billed == 10.5 * 20_000  # 10.5x the naive per-step estimate


def test_doubling_the_steps_roughly_quadruples_the_cost(book: PriceBook) -> None:
    ten = agent_loop_cost(book, label="", model=OPUS, steps=10, system_tokens=0).cost_usd
    twenty = agent_loop_cost(book, label="", model=OPUS, steps=20, system_tokens=0).cost_usd

    assert twenty / ten > Decimal("3.0")


def test_quadratic_growth_table_is_monotonic(book: PriceBook) -> None:
    rows = quadratic_growth(book)

    assert [row[1] for row in rows] == sorted(row[1] for row in rows)


# ---------------------------------------------------------------------- caching


def test_caching_cuts_a_long_loop_by_more_than_half(book: PriceBook) -> None:
    plain = agent_loop_cost(
        book, label="", model=OPUS, steps=20, system_tokens=3000, caching=False
    )
    cached = agent_loop_cost(
        book, label="", model=OPUS, steps=20, system_tokens=3000, caching=True
    )

    assert cached.cost_usd < plain.cost_usd / 2
    assert cached.cache_hit_rate > 0.9


def test_caching_below_the_model_floor_silently_does_nothing(book: PriceBook) -> None:
    """A 3k prefix is above Opus 5's 512 floor and below Haiku 4.5's 4096.
    The API does not error -- it just never caches."""
    requested = agent_loop_cost(
        book, label="", model=HAIKU, steps=20, system_tokens=3000, caching=True
    )
    not_requested = agent_loop_cost(
        book, label="", model=HAIKU, steps=20, system_tokens=3000, caching=False
    )

    assert requested.cached_effectively is False
    assert requested.cache_status == "SILENT NO-OP"
    assert requested.cost_usd == not_requested.cost_usd  # paid for nothing


def test_the_same_prefix_does_cache_on_the_expensive_model(book: PriceBook) -> None:
    cached = agent_loop_cost(
        book, label="", model=OPUS, steps=20, system_tokens=3000, caching=True
    )

    assert cached.cached_effectively is True


def test_haiku_caches_once_the_prefix_clears_its_floor(book: PriceBook) -> None:
    cached = agent_loop_cost(
        book, label="", model=HAIKU, steps=20, system_tokens=5000, caching=True
    )

    assert cached.cached_effectively is True


# ------------------------------------------------------ the routing/caching tension


def test_cached_flagship_eventually_beats_the_uncached_cheap_model(
    book: PriceBook,
) -> None:
    """The talk's headline. Cache reads are 0.10x input, so a cached Opus 5
    prefix effectively costs $0.50/MTok against Haiku 4.5's uncached $1.00/MTok.
    Past the crossover the 5x-more-expensive model is the cheaper way to run."""
    crossover = crossover_step(book)

    assert crossover is not None
    assert 20 < crossover < 60, f"crossover moved to step {crossover}"

    beyond = crossover + 20
    expensive = agent_loop_cost(
        book, label="", model=OPUS, steps=beyond, system_tokens=3000, caching=True
    )
    cheap = agent_loop_cost(
        book, label="", model=HAIKU, steps=beyond, system_tokens=3000, caching=False
    )

    assert expensive.cost_usd < cheap.cost_usd


def test_rerouting_every_step_forfeits_cache_reads(book: PriceBook) -> None:
    """Caches are model-scoped with no escape hatch, so a switch re-pays the
    write premium. A rerouting loop caches strictly worse than a pinned one."""
    pinned = agent_loop_cost(
        book, label="", model=OPUS, steps=20, system_tokens=3000, caching=True
    )
    rerouted = alternating_model_loop_cost(
        book, label="", models=(OPUS, HAIKU), steps=20, system_tokens=3000
    )

    assert rerouted.cache_hit_rate < pinned.cache_hit_rate


def test_tension_scenarios_cover_both_cache_outcomes(book: PriceBook) -> None:
    statuses = {scenario.cache_status for scenario in tension_scenarios(book)}

    assert "cached" in statuses
    assert "SILENT NO-OP" in statuses


# ------------------------------------------------------------- context editing


def test_context_editing_makes_history_linear_instead_of_quadratic() -> None:
    kwargs = dict(steps=40, system_tokens=0, per_step_tokens=1000, output_tokens=0)

    unbounded = sum(
        r.prompt_tokens for r in agent_loop_usage(**kwargs, caching=False)
    )
    bounded = sum(
        r.prompt_tokens for r in context_edited_loop_usage(**kwargs, keep_last_turns=3)
    )

    assert bounded < unbounded / 4


# ------------------------------------------------------------------- ranking


def test_lever_ranking_is_sorted_and_baseline_is_worst(book: PriceBook) -> None:
    levers = rank_levers(book)
    costs = [lever.cost_usd for lever in levers]

    assert costs == sorted(costs)
    assert levers[-1].name == "do nothing (baseline)"


def test_caching_alone_beats_the_batch_api_on_a_long_loop(book: PriceBook) -> None:
    """Batch is a flat 50%; caching on a repeated prefix beats that comfortably
    -- and without the up-to-24-hour latency trade."""
    by_name = {lever.name: lever for lever in rank_levers(book)}

    assert by_name["prompt caching"].cost_usd < by_name["Batch API"].cost_usd


def test_every_ranked_lever_reports_whether_it_is_exact(book: PriceBook) -> None:
    assert all(isinstance(lever.exact, bool) for lever in rank_levers(book))
