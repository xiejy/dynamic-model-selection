"""Scoring strategies honestly.

Three rules this module enforces, each of which a published cost-savings figure
somewhere quietly breaks:

1. **Every call counts.** Router classification calls, cascade verification
   calls, and thrown-away cheap attempts are all billed to the strategy that
   made them. A saving computed from answer calls only is not a saving.

2. **The baseline is random routing at the same strong-model call fraction**,
   not always-strong. Any router that sends 40% of traffic to the expensive
   model automatically looks better than always-cheap on quality and better than
   always-expensive on cost. Beating a weighted coin at the same rate is the
   real bar, and it is the one RouteLLM's own evaluation uses.

3. **Quality is reported next to cost, always.** A 3x cheaper model that fails
   20% of the time costs more than the expensive one that does not, once retries
   and escalations are counted -- and a degraded answer is well-formed, so
   nothing alerts.

PGR (performance gap recovered) places a strategy between the weak model (0.0)
and the strong model (1.0). It is the standard axis in the routing literature and
makes routers with different call fractions comparable.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean

from dms.pricing import PriceBook
from dms.routers.base import Outcome
from dms.workload import DIFFICULTIES


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """Aggregated cost and quality for one strategy over the whole workload."""

    name: str
    tasks: int
    correct: int
    cost_usd: Decimal
    router_cost_usd: Decimal
    wasted_cost_usd: Decimal
    strong_calls: int
    escalations: int
    avg_latency_ms: float
    accuracy_by_difficulty: dict[str, float]
    model_mix: dict[str, int]
    simulated: bool

    @property
    def accuracy(self) -> float:
        return self.correct / self.tasks if self.tasks else 0.0

    @property
    def cost_per_task_usd(self) -> Decimal:
        return self.cost_usd / self.tasks if self.tasks else Decimal(0)

    @property
    def strong_call_fraction(self) -> float:
        return self.strong_calls / self.tasks if self.tasks else 0.0

    @property
    def overhead_share(self) -> float:
        """Fraction of spend that bought no answer: routing + discarded attempts."""
        if self.cost_usd == 0:
            return 0.0
        return float((self.router_cost_usd + self.wasted_cost_usd) / self.cost_usd)


def summarise(
    name: str,
    outcomes: Sequence[Outcome],
    book: PriceBook,
    *,
    strong_model: str = "claude-opus-5",
) -> StrategyResult:
    """Fold a strategy's per-task outcomes into one comparable row."""
    if not outcomes:
        raise ValueError(f"strategy {name!r} produced no outcomes")

    total = Decimal(0)
    router_cost = Decimal(0)
    wasted = Decimal(0)
    model_mix: dict[str, int] = {}

    for outcome in outcomes:
        escalated = outcome.escalated
        for spend in outcome.spends:
            cost = book.cost_usd(spend.usage, spend.model)
            total += cost
            if spend.role == "router":
                router_cost += cost
            # A cheap answer that was superseded by an escalation bought nothing.
            elif spend.role in {"verify"} or (spend.role == "answer" and escalated):
                wasted += cost
        model_mix[outcome.chosen_model] = model_mix.get(outcome.chosen_model, 0) + 1

    return StrategyResult(
        name=name,
        tasks=len(outcomes),
        correct=sum(1 for outcome in outcomes if outcome.correct),
        cost_usd=total,
        router_cost_usd=router_cost,
        wasted_cost_usd=wasted,
        strong_calls=sum(1 for o in outcomes if o.chosen_model == strong_model),
        escalations=sum(1 for o in outcomes if o.escalated),
        avg_latency_ms=fmean(outcome.latency_ms for outcome in outcomes),
        accuracy_by_difficulty=_accuracy_by_difficulty(outcomes),
        model_mix=dict(sorted(model_mix.items())),
        simulated=any(o.simulated for o in outcomes),
    )


def performance_gap_recovered(
    result: StrategyResult, weak: StrategyResult, strong: StrategyResult
) -> float:
    """Where this strategy sits between always-weak (0.0) and always-strong (1.0).

    Can exceed 1.0 or go negative -- a cascade that catches the strong model's
    mistakes can beat it, and a bad router can be worse than always-weak. Both
    are real outcomes and neither is clamped away.
    """
    span = strong.accuracy - weak.accuracy
    if span == 0:
        return 1.0  # the models are indistinguishable here; routing is free
    return (result.accuracy - weak.accuracy) / span


def cost_saving_vs(result: StrategyResult, baseline: StrategyResult) -> float:
    """Fraction of the baseline's spend avoided. Negative means it cost more."""
    if baseline.cost_usd == 0:
        return 0.0
    return float((baseline.cost_usd - result.cost_usd) / baseline.cost_usd)


def interpolate_random(
    curve: Sequence[StrategyResult], strong_fraction: float
) -> tuple[float, Decimal]:
    """(accuracy, cost) a random router would reach at this call fraction.

    Linear interpolation between the sampled random-routing points. This is the
    number a real router has to beat; matching it means the routing logic added
    nothing a coin could not do.
    """
    if not curve:
        raise ValueError("random curve is empty")

    points = sorted(curve, key=lambda r: r.strong_call_fraction)
    if strong_fraction <= points[0].strong_call_fraction:
        return points[0].accuracy, points[0].cost_usd
    if strong_fraction >= points[-1].strong_call_fraction:
        return points[-1].accuracy, points[-1].cost_usd

    for lower, upper in zip(points, points[1:], strict=False):
        if lower.strong_call_fraction <= strong_fraction <= upper.strong_call_fraction:
            span = upper.strong_call_fraction - lower.strong_call_fraction
            weight = 0.0 if span == 0 else (strong_fraction - lower.strong_call_fraction) / span
            accuracy = lower.accuracy + weight * (upper.accuracy - lower.accuracy)
            cost = lower.cost_usd + Decimal(str(weight)) * (upper.cost_usd - lower.cost_usd)
            return accuracy, cost

    return points[-1].accuracy, points[-1].cost_usd


def beats_random(
    result: StrategyResult, curve: Sequence[StrategyResult]
) -> tuple[float, Decimal]:
    """(accuracy delta, cost delta) against random routing at the same rate.

    Positive accuracy delta with non-positive cost delta is the only combination
    that justifies a router existing.
    """
    accuracy, cost = interpolate_random(curve, result.strong_call_fraction)
    return result.accuracy - accuracy, result.cost_usd - cost


def pareto_frontier(results: Sequence[StrategyResult]) -> tuple[StrategyResult, ...]:
    """Strategies not dominated on both cost and accuracy by another."""
    frontier = [
        candidate
        for candidate in results
        if not any(
            other.cost_usd <= candidate.cost_usd
            and other.accuracy >= candidate.accuracy
            and (other.cost_usd < candidate.cost_usd or other.accuracy > candidate.accuracy)
            for other in results
        )
    ]
    return tuple(sorted(frontier, key=lambda r: r.cost_usd))


def _accuracy_by_difficulty(outcomes: Sequence[Outcome]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for level in DIFFICULTIES:
        subset = [outcome for outcome in outcomes if outcome.difficulty == level]
        if subset:
            scores[level] = sum(1 for o in subset if o.correct) / len(subset)
    return scores
