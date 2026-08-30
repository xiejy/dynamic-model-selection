"""Rendering: terminal tables for the demo, JSON for the slides.

Stdlib formatting only -- no table library. Keeps the dependency list at one
package, which matters for a repo whose whole subject is unnecessary spend.
"""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from dms.bench import STRONG_MODEL, WEAK_MODEL, BenchReport
from dms.metrics import (
    StrategyResult,
    beats_random,
    cost_saving_vs,
    pareto_frontier,
    performance_gap_recovered,
)

SIMULATED_BANNER = (
    "!! SIMULATED RUN -- these are ESTIMATES from config/simulation.json, not\n"
    "!! measurements. Re-run with --mode record and a real ANTHROPIC_API_KEY\n"
    "!! before quoting any number here."
)
MEASURED_BANNER = "== MEASURED RUN -- figures come from response.usage."


def render(report: BenchReport) -> str:
    """Full terminal report."""
    sections = [
        SIMULATED_BANNER if report.simulated else MEASURED_BANNER,
        "",
        _header(report),
        "",
        _main_table(report),
        "",
        _router_honesty_table(report),
        "",
        _pareto(report),
        "",
        _difficulty_table(report),
    ]
    return "\n".join(sections)


# ------------------------------------------------------------------------ sections


def _header(report: BenchReport) -> str:
    mix = ", ".join(f"{count} {level}" for level, count in report.workload_mix.items())
    lines = [
        f"mode={report.mode}  tasks={sum(report.workload_mix.values())} ({mix})",
        f"calls={report.calls_made}  spend=${report.total_spend_usd:.4f}",
    ]
    # Refusals and truncations grade as wrong answers unless called out. Both
    # are properties of the request, not of the model's capability.
    if report.refusals:
        by_model: dict[str, int] = {}
        for model, category in report.refusals:
            by_model[f"{model}/{category or 'uncategorised'}"] = (
                by_model.get(f"{model}/{category or 'uncategorised'}", 0) + 1
            )
        detail = ", ".join(f"{k} x{v}" for k, v in sorted(by_model.items()))
        lines.append(f"!! {len(report.refusals)} safety refusal(s): {detail}")
        lines.append("   (stop_reason=refusal returns empty content and grades as WRONG)")
    if report.truncations:
        lines.append(f"!! {len(report.truncations)} answer(s) hit max_tokens and were truncated")
    return "\n".join(lines)


def _main_table(report: BenchReport) -> str:
    baseline = report.by_name("always:opus")
    weak = report.by_name("always:haiku")

    rows = [
        (
            result.name,
            f"{result.accuracy:6.1%}",
            f"${result.cost_usd:8.4f}",
            f"{cost_saving_vs(result, baseline):+7.1%}",
            f"{performance_gap_recovered(result, weak, baseline):+6.2f}",
            f"{result.strong_call_fraction:6.0%}",
            f"{result.avg_latency_ms:7.0f}",
        )
        for result in report.results
    ]
    return _table(
        "COST vs QUALITY  (savings and PGR are both vs always-opus)",
        ("strategy", "accuracy", "cost", "saving", "PGR", "%strong", "ms/task"),
        rows,
    )


def _router_honesty_table(report: BenchReport) -> str:
    """Where the money went that did not buy an answer, and the random check."""
    rows = []
    for result in report.results:
        delta_accuracy, delta_cost = (
            beats_random(result, report.random_curve) if report.random_curve else (0.0, Decimal(0))
        )
        verdict = _verdict(delta_accuracy, delta_cost)
        rows.append(
            (
                result.name,
                f"${result.router_cost_usd:7.4f}",
                f"${result.wasted_cost_usd:7.4f}",
                f"{result.overhead_share:6.1%}",
                f"{delta_accuracy:+6.1%}",
                f"${delta_cost:+7.4f}",
                verdict,
            )
        )
    return _table(
        "DOES THE ROUTER EARN ITS KEEP?  (vs random routing at the same %strong)",
        ("strategy", "router $", "wasted $", "overhead", "d-acc", "d-cost", "verdict"),
        rows,
    )


def _pareto(report: BenchReport) -> str:
    frontier = pareto_frontier(report.results)
    names = "\n".join(
        f"  {result.name:<22} ${result.cost_usd:.4f}  {result.accuracy:.1%}"
        for result in frontier
    )
    dominated = sorted(
        {result.name for result in report.results} - {r.name for r in frontier}
    )
    return (
        "PARETO FRONTIER (nothing is both cheaper and better)\n"
        f"{names}\n"
        f"  dominated: {', '.join(dominated) if dominated else '(none)'}"
    )


def _difficulty_table(report: BenchReport) -> str:
    rows = [
        (
            result.name,
            *(
                f"{result.accuracy_by_difficulty.get(level, 0.0):6.1%}"
                for level in ("easy", "medium", "hard")
            ),
        )
        for result in report.results
    ]
    return _table(
        "ACCURACY BY DIFFICULTY  (routing lives or dies on the hard column)",
        ("strategy", "easy", "medium", "hard"),
        rows,
    )


# -------------------------------------------------------------------------- export


def to_dict(report: BenchReport) -> dict[str, Any]:
    """JSON-serialisable form. This is what the slide deck reads."""
    baseline = report.by_name("always:opus")
    weak = report.by_name("always:haiku")

    return {
        "mode": report.mode,
        "simulated": report.simulated,
        "workload_mix": report.workload_mix,
        "calls_made": report.calls_made,
        "refusals": [{"model": m, "category": c} for m, c in report.refusals],
        "truncations": len(report.truncations),
        "total_spend_usd": str(report.total_spend_usd),
        "models": {"weak": WEAK_MODEL, "strong": STRONG_MODEL},
        "strategies": [
            _result_dict(result, baseline, weak, report) for result in report.results
        ],
        "random_curve": [
            {
                "strong_fraction": result.strong_call_fraction,
                "accuracy": result.accuracy,
                "cost_usd": str(result.cost_usd),
            }
            for result in report.random_curve
        ],
        "pareto": [result.name for result in pareto_frontier(report.results)],
        # Per-task correctness, so strategies can be compared PAIRED (McNemar)
        # rather than by eyeballing two accuracy percentages whose confidence
        # intervals overlap heavily at n=36.
        "per_task": {
            name: {
                o.task_id: {
                    "correct": o.correct,
                    "difficulty": o.difficulty,
                    "model": o.chosen_model,
                }
                for o in outcomes
            }
            for name, outcomes in report.outcomes.items()
        },
    }


def _result_dict(
    result: StrategyResult,
    baseline: StrategyResult,
    weak: StrategyResult,
    report: BenchReport,
) -> dict[str, Any]:
    delta_accuracy, delta_cost = (
        beats_random(result, report.random_curve)
        if report.random_curve
        else (0.0, Decimal(0))
    )
    return {
        "name": result.name,
        "tasks": result.tasks,
        "accuracy": result.accuracy,
        "accuracy_by_difficulty": result.accuracy_by_difficulty,
        "cost_usd": str(result.cost_usd),
        "cost_per_task_usd": str(result.cost_per_task_usd),
        "router_cost_usd": str(result.router_cost_usd),
        "wasted_cost_usd": str(result.wasted_cost_usd),
        "overhead_share": result.overhead_share,
        "saving_vs_strong": cost_saving_vs(result, baseline),
        "pgr": performance_gap_recovered(result, weak, baseline),
        "strong_call_fraction": result.strong_call_fraction,
        "escalations": result.escalations,
        "avg_latency_ms": result.avg_latency_ms,
        "model_mix": result.model_mix,
        "vs_random": {
            "accuracy_delta": delta_accuracy,
            "cost_delta_usd": str(delta_cost),
            "verdict": _verdict(delta_accuracy, delta_cost),
        },
    }


def dumps(report: BenchReport) -> str:
    return json.dumps(to_dict(report), indent=2)


# ------------------------------------------------------------------------- helpers


def render_levers(book) -> str:
    """The toolkit ranking, the quadratic curve, and the caching/routing tension."""
    from dms.levers.caching import crossover_step, quadratic_growth, tension_scenarios
    from dms.levers.toolkit import effort_note, rank_levers

    ranking = _table(
        "COST LEVERS RANKED  (20-step agent loop, 3k system prompt, Opus 5)",
        ("lever", "cost", "saving", "exact?", "trade-off"),
        [
            (
                lever.name,
                f"${lever.cost_usd:8.4f}",
                f"{lever.saving_vs_baseline:+7.1%}",
                "yes" if lever.exact else "MEASURE",
                lever.tradeoff,
            )
            for lever in rank_levers(book)
        ],
    )

    growth = _table(
        "WHY AGENT LOOPS ARE EXPENSIVE  (billed input grows with the SQUARE of steps)",
        ("steps", "billed input tokens", "cost"),
        [
            (str(steps), f"{tokens:,}", f"${cost:.4f}")
            for steps, tokens, cost in quadratic_growth(book)
        ],
    )

    tension = _table(
        "ROUTING vs CACHING  (3k prefix: above Opus 5's 512 floor, below Haiku's 4096)",
        ("scenario", "cost", "cache hit", "cache status"),
        [
            (
                scenario.label,
                f"${scenario.cost_usd:8.4f}",
                f"{scenario.cache_hit_rate:6.1%}",
                scenario.cache_status,
            )
            for scenario in tension_scenarios(book)
        ],
    )

    crossover = crossover_step(book)
    punchline = (
        f"CROSSOVER: from step {crossover}, a CACHED Opus 5 loop costs less than an\n"
        f"UNCACHED Haiku 4.5 loop -- despite Haiku being 5x cheaper per token.\n"
        "Cache reads are 0.10x input, so cached Opus input effectively costs\n"
        "$0.50/MTok against Haiku's uncached $1.00/MTok. Fix caching before you\n"
        "argue about model choice."
        if crossover
        else "No crossover within the search range."
    )

    return "\n\n".join(
        [
            "== EXACT ARITHMETIC over documented pricing. No model behaviour assumed.",
            ranking,
            effort_note(),
            growth,
            tension,
            punchline,
        ]
    )


def _verdict(delta_accuracy: float, delta_cost: Decimal) -> str:
    """Did this strategy beat a coin weighted to the same call fraction?"""
    if abs(delta_accuracy) < 0.005 and abs(delta_cost) < Decimal("0.0001"):
        return "= random"
    if delta_accuracy >= 0 and delta_cost <= 0:
        return "beats random"
    if delta_accuracy < 0 and delta_cost > 0:
        return "LOSES to random"
    return "mixed"


def _table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    line = "  ".join("-" * width for width in widths)
    head = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    body = "\n".join(
        "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    )
    return f"{title}\n{head}\n{line}\n{body}"


def render_dispatch_value(report) -> str:
    """The binary-dispatch verdict table: was the routing logic worth building?"""
    from dms.twotier import HIGH_MODEL, LOW_MODEL, evaluate

    low = report.by_name(f"always:{LOW_MODEL.split('-')[1]}")
    high = report.by_name(f"always:{HIGH_MODEL.split('-')[1]}")
    oracle = report.by_name("oracle")

    values = [
        evaluate(r, low=low, high=high, oracle=oracle, random_curve=report.random_curve)
        for r in report.results
        if r.name not in {low.name, high.name}
    ]

    headline = _table(
        "IS THE DISPATCHER WORTH IT?  (vs a coin sending the same share to the high model)",
        ("dispatcher", "%high", "accuracy", "cost", "d-acc vs coin", "verdict"),
        [
            (
                v.name,
                f"{v.high_share:5.0%}",
                f"{v.accuracy:7.1%}",
                f"${v.cost_usd:7.4f}",
                f"{v.accuracy_vs_random:+8.1%}",
                v.verdict,
            )
            for v in values
        ],
    )

    anchors = _table(
        "AGAINST THE TWO TRIVIAL ANSWERS",
        ("dispatcher", "vs always-low", "vs always-high", "saving vs high", "oracle gap captured"),
        [
            (
                v.name,
                f"{v.accuracy_vs_low:+7.1%} acc",
                f"{v.accuracy_vs_high:+7.1%} acc",
                f"{v.saving_vs_high:+7.1%}",
                f"{v.oracle_gap_captured:6.0%}",
            )
            for v in values
        ],
    )

    return "\n\n".join([headline, anchors])
