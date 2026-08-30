"""Binary high/low dispatch: is a dispatcher worth building at all?

This is the question stripped to its core. Two models, one decision per request:
send it to the cheap one or the expensive one. Everything else -- three tiers,
model families, gateways -- is elaboration on this.

A dispatcher is only worth building if it beats BOTH trivial answers (always-low,
always-high) AND the non-trivial one everybody forgets: **a weighted coin that
sends the same fraction of traffic to the high model**. If a dispatcher does not
beat random routing at its own call fraction, its logic contributed nothing --
the savings came from the mix, not the decision.

The value of a dispatcher is therefore measured on two axes:

    quality lift  = accuracy(dispatcher) - accuracy(random at same rate)
    cost delta    = cost(dispatcher)     - cost(random at same rate)

and bounded above by the oracle: the unbuildable dispatcher that always picks the
cheapest model that would actually have been right.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from dms.metrics import StrategyResult, beats_random, interpolate_random

LOW_MODEL = "claude-haiku-4-5"
HIGH_MODEL = "claude-opus-5"


def two_tier_map(low: str = LOW_MODEL, high: str = HIGH_MODEL) -> dict[str, str]:
    """Collapse the three-tier router vocabulary onto two models.

    Anything a router judges harder than 'simple' goes high. This is the
    conservative reading, and it is what a team shipping a binary dispatcher
    would do -- ambiguity resolves toward quality.
    """
    return {"simple": low, "medium": high, "complex": high}


@dataclass(frozen=True, slots=True)
class DispatchValue:
    """What one dispatcher was worth, against every relevant baseline."""

    name: str
    accuracy: float
    cost_usd: Decimal
    high_share: float
    overhead_share: float

    # versus the two trivial answers
    accuracy_vs_low: float
    accuracy_vs_high: float
    saving_vs_high: float
    extra_cost_vs_low: float

    # versus the answer that actually matters
    accuracy_vs_random: float
    cost_vs_random: Decimal

    # versus the ceiling
    oracle_gap_captured: float

    @property
    def verdict(self) -> str:
        """Blunt call on whether the dispatch logic earned its existence."""
        if self.accuracy_vs_random < -0.005:
            return "NOT WORTH IT -- worse than a coin at the same rate"
        if abs(self.accuracy_vs_random) <= 0.005 and self.cost_vs_random >= 0:
            return "NO VALUE -- indistinguishable from a coin"
        if self.accuracy_vs_random > 0 and self.cost_vs_random <= 0:
            return "WORTH IT -- better and cheaper than a coin"
        return "MARGINAL -- better than a coin, but pays for it"


def evaluate(
    result: StrategyResult,
    *,
    low: StrategyResult,
    high: StrategyResult,
    oracle: StrategyResult,
    random_curve: tuple[StrategyResult, ...],
) -> DispatchValue:
    """Score one dispatcher against low, high, random-at-same-rate, and oracle."""
    accuracy_delta, cost_delta = beats_random(result, random_curve)
    random_accuracy, _ = interpolate_random(random_curve, result.strong_call_fraction)

    # How much of the headroom between a coin and perfect dispatch was captured.
    headroom = oracle.accuracy - random_accuracy
    captured = (result.accuracy - random_accuracy) / headroom if headroom > 0 else 0.0

    return DispatchValue(
        name=result.name,
        accuracy=result.accuracy,
        cost_usd=result.cost_usd,
        high_share=result.strong_call_fraction,
        overhead_share=result.overhead_share,
        accuracy_vs_low=result.accuracy - low.accuracy,
        accuracy_vs_high=result.accuracy - high.accuracy,
        saving_vs_high=(
            float((high.cost_usd - result.cost_usd) / high.cost_usd)
            if high.cost_usd
            else 0.0
        ),
        extra_cost_vs_low=(
            float((result.cost_usd - low.cost_usd) / low.cost_usd) if low.cost_usd else 0.0
        ),
        accuracy_vs_random=accuracy_delta,
        cost_vs_random=cost_delta,
        oracle_gap_captured=captured,
    )


# --------------------------------------------------------------------- routability


@dataclass(frozen=True, slots=True)
class Routability:
    """How much of a workload a dispatcher could possibly affect.

    This is the number to measure BEFORE building anything. On every request
    where the cheap and expensive models return the same verdict, the dispatcher
    is irrelevant -- it can only change cost, never quality. The routable share
    is therefore a hard ceiling on dispatch value, and it is a property of the
    workload, not of the router.

    It also splits into two very different halves:
      * high_only  -- the tasks a dispatcher must catch to avoid a regression
      * low_only   -- tasks the EXPENSIVE model gets wrong and the cheap one right,
                      which no amount of router cleverness can be blamed for
    """

    total: int
    both_right: int
    both_wrong: int
    high_only: tuple[str, ...]
    low_only: tuple[str, ...]

    @property
    def routable(self) -> int:
        return len(self.high_only) + len(self.low_only)

    @property
    def routable_share(self) -> float:
        return self.routable / self.total if self.total else 0.0

    @property
    def agreement(self) -> float:
        return (self.both_right + self.both_wrong) / self.total if self.total else 0.0

    @property
    def headroom_note(self) -> str:
        return (
            f"{self.routable_share:.0%} of this workload is routable; on the other "
            f"{self.agreement:.0%} a dispatcher changes cost only, never quality."
        )


def routability(
    low_outcomes: dict[str, bool], high_outcomes: dict[str, bool]
) -> Routability:
    """Compare per-task correctness of the two anchors."""
    ids = sorted(set(low_outcomes) & set(high_outcomes))
    return Routability(
        total=len(ids),
        both_right=sum(1 for i in ids if low_outcomes[i] and high_outcomes[i]),
        both_wrong=sum(1 for i in ids if not low_outcomes[i] and not high_outcomes[i]),
        high_only=tuple(i for i in ids if high_outcomes[i] and not low_outcomes[i]),
        low_only=tuple(i for i in ids if low_outcomes[i] and not high_outcomes[i]),
    )


def mcnemar_p(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """Exact binomial McNemar test on paired per-task correctness.

    Only discordant pairs carry information, which is why an accuracy gap that
    looks large can still be unresolvable: at n=36 with ~7 discordant tasks,
    nothing short of a near-total sweep reaches significance.
    """
    import math

    ids = sorted(set(a) & set(b))
    a_only = sum(1 for i in ids if a[i] and not b[i])
    b_only = sum(1 for i in ids if b[i] and not a[i])
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return a_only, b_only, min(1.0, 2 * tail / 2**n)


def tasks_needed(effect_pp: float = 5.0, discordance: float = 0.19) -> int:
    """Roughly how many tasks are needed to resolve `effect_pp` at 80% power.

    Defaults describe this suite: a 5-percentage-point effect at the 19%
    disagreement rate measured here.
    """
    import math

    psi = 0.5 + (effect_pp / 100) / (2 * discordance)
    psi = min(psi, 0.999)
    numerator = (1.96 * 0.5 + 0.84 * math.sqrt(psi * (1 - psi))) ** 2
    return math.ceil(numerator / (discordance * (psi - 0.5) ** 2))
