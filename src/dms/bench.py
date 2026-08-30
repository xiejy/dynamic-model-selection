"""The benchmark harness: run every strategy over the same workload."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from dms.client import ModelClient
from dms.metrics import StrategyResult, summarise
from dms.routers.base import Outcome
from dms.routers.baseline import AlwaysRouter, RandomRouter
from dms.routers.heuristic import HeuristicRouter
from dms.routers.lexical import LexicalRouter
from dms.routers.llm_classifier import LLMClassifierRouter
from dms.strategies import CascadeStrategy, OracleStrategy, RoutedStrategy, Strategy
from dms.workload import Workload

WEAK_MODEL = "claude-haiku-4-5"
MID_MODEL = "claude-sonnet-5"
STRONG_MODEL = "claude-opus-5"

# Sampled call fractions for the random-routing baseline curve.
RANDOM_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True, slots=True)
class BenchReport:
    """Everything one bench run produced."""

    results: tuple[StrategyResult, ...]
    random_curve: tuple[StrategyResult, ...]
    outcomes: dict[str, tuple[Outcome, ...]]
    mode: str
    total_spend_usd: Decimal
    calls_made: int
    workload_mix: dict[str, int]
    refusals: tuple[tuple[str, str], ...] = ()
    truncations: tuple[tuple[str, str], ...] = ()

    @property
    def simulated(self) -> bool:
        return any(result.simulated for result in self.results)

    def by_name(self, name: str) -> StrategyResult:
        for result in self.results:
            if result.name == name:
                return result
        raise KeyError(f"no strategy named {name!r}")


def default_strategies() -> tuple[Strategy, ...]:
    """The line-up, cheapest-to-decide first."""
    return (
        RoutedStrategy(AlwaysRouter(model=WEAK_MODEL)),
        RoutedStrategy(AlwaysRouter(model=MID_MODEL)),
        RoutedStrategy(AlwaysRouter(model=STRONG_MODEL)),
        RoutedStrategy(HeuristicRouter()),
        RoutedStrategy(LexicalRouter()),
        RoutedStrategy(LLMClassifierRouter()),
        CascadeStrategy(weak_model=WEAK_MODEL, strong_model=STRONG_MODEL),
        CascadeStrategy(
            weak_model=WEAK_MODEL, strong_model=STRONG_MODEL, verify=False,
            name="cascade-no-verify",
        ),
        OracleStrategy(),
    )


def two_tier_strategies() -> tuple[Strategy, ...]:
    """The binary high/low line-up: does a dispatcher beat the trivial answers?"""
    from dms.twotier import HIGH_MODEL, LOW_MODEL, two_tier_map

    tiers = two_tier_map()
    return (
        RoutedStrategy(AlwaysRouter(model=LOW_MODEL)),
        RoutedStrategy(AlwaysRouter(model=HIGH_MODEL)),
        RoutedStrategy(HeuristicRouter(tiers=tiers)),
        RoutedStrategy(LexicalRouter(tiers=tiers)),
        RoutedStrategy(LLMClassifierRouter(tiers=tiers)),
        CascadeStrategy(weak_model=LOW_MODEL, strong_model=HIGH_MODEL),
        OracleStrategy(ladder=(LOW_MODEL, HIGH_MODEL)),
    )


def random_baseline_strategies(
    fractions: Sequence[float] = RANDOM_FRACTIONS,
) -> tuple[Strategy, ...]:
    return tuple(
        RoutedStrategy(
            RandomRouter(
                strong_model=STRONG_MODEL,
                weak_model=WEAK_MODEL,
                strong_fraction=fraction,
            )
        )
        for fraction in fractions
    )


def run_strategy(
    strategy: Strategy, workload: Workload, client: ModelClient
) -> tuple[Outcome, ...]:
    return tuple(strategy.run(task, client) for task in workload)


def run_bench(
    workload: Workload,
    client: ModelClient,
    strategies: Sequence[Strategy] | None = None,
    *,
    include_random_curve: bool = True,
) -> BenchReport:
    """Run every strategy over every task and fold the results."""
    chosen = tuple(strategies) if strategies is not None else default_strategies()

    outcomes: dict[str, tuple[Outcome, ...]] = {}
    results: list[StrategyResult] = []
    for strategy in chosen:
        produced = run_strategy(strategy, workload, client)
        outcomes[strategy.name] = produced
        results.append(summarise(strategy.name, produced, client.book))

    curve: list[StrategyResult] = []
    if include_random_curve:
        for strategy in random_baseline_strategies():
            produced = run_strategy(strategy, workload, client)
            outcomes[strategy.name] = produced
            curve.append(summarise(strategy.name, produced, client.book))

    return BenchReport(
        results=tuple(results),
        random_curve=tuple(curve),
        outcomes=outcomes,
        mode=str(client.mode),
        total_spend_usd=client.total_spend_usd,
        calls_made=client.calls_made,
        workload_mix=workload.mix(),
        refusals=tuple(client.refusals),
        truncations=tuple(client.truncated_calls),
    )
