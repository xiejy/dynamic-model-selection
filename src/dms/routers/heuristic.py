"""Rule-based complexity scorer. Zero API calls, sub-millisecond, no training.

Seven signals, mirroring the dimension set LiteLLM's Auto Router v2 ships as its
*default* classifier. That default is the interesting part: the most-deployed
production router is a bag of regexes, not a model.

The independent benchmarks make this less embarrassing than it sounds.
LLMRouterBench finds leading routing methods "broadly comparable" and a 22.7M
-parameter embedder performing about as well as large ones -- method innovation
has plateaued. If a regex captures most of the win at zero marginal cost, the
learned router has to beat it *net of its own inference cost*, which is a much
harder bar than beating it on accuracy alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from dms.client import ModelClient
from dms.routers.base import TIER_MODELS, Decision, Router
from dms.workload import Task

# Signal vocabularies. Kept explicit and greppable rather than learned -- the
# point of this router is that you can read it, audit it, and explain a decision.
REASONING_MARKERS = (
    "why", "explain", "prove", "derive", "trade-off", "tradeoff", "root cause",
    "complexity", "deadlock", "vulnerable", "worst-case", "lower bound",
    "violated", "guarantee", "is deadlock", "name the cause", "name the bug",
    "optimi", "theoretical",
)
MULTI_STEP_MARKERS = (
    "repeat", "step by step", "sequence", "in total", "each time",
    "starting from", "start with", "occur", "trace", "final value",
)
# Task-shape markers ONLY. Deliberately not the answer-format instructions
# ("answer with the number only", "yes or no") -- those appear at every
# difficulty, so keying on them is a spurious feature. The first revision of this
# router did exactly that, scored every hard task negative, and routed 100% of
# traffic to Haiku while reading perfectly sensibly in code review. Surface form
# is not difficulty, and that failure is invisible unless you measure quality.
SIMPLE_MARKERS = (
    "extract the", "which programming language", "file extension",
    "classify the", "give the git command", "give the shell command",
    "decimal value of", "how many bytes", "how many kibibytes",
)
TECHNICAL_TERMS = (
    "mutex", "goroutine", "ieee-754", "regex", "lru", "btree", "b-tree",
    "replica", "idempotent", "backtracking", "closure", "rebase", "semver",
    "cron", "dockerfile", "index", "recursive",
)
CODE_PATTERN = re.compile(r"```|\bdef \b|\bSELECT\b|[{}()\[\]]{2,}|=>|->|\$\d", re.I)

# Score bands -> tier. Tuned once against the workload's difficulty labels; the
# thresholds are the router's only "training", and they are two numbers you can
# argue with in a code review.
MEDIUM_THRESHOLD = 1.0
COMPLEX_THRESHOLD = 2.5


@dataclass(frozen=True, slots=True)
class Signal:
    name: str
    weight: float
    hit: bool

    @property
    def contribution(self) -> float:
        return self.weight if self.hit else 0.0


@dataclass(frozen=True, slots=True)
class HeuristicRouter(Router):
    """Score a prompt on seven dimensions, then band the score into a tier."""

    tiers: dict[str, str] = field(default_factory=lambda: dict(TIER_MODELS))
    medium_threshold: float = MEDIUM_THRESHOLD
    complex_threshold: float = COMPLEX_THRESHOLD
    name: str = "heuristic"

    def score(self, prompt: str) -> tuple[float, tuple[Signal, ...]]:
        text = prompt.lower()
        words = len(text.split())

        signals = (
            Signal("long_prompt", 1.0, words > 35),
            Signal("very_long_prompt", 1.0, words > 70),
            Signal("contains_code", 1.0, bool(CODE_PATTERN.search(prompt))),
            Signal("reasoning_markers", 1.5, _any_in(text, REASONING_MARKERS)),
            Signal("multi_step", 1.5, _count_in(text, MULTI_STEP_MARKERS) >= 1),
            Signal("technical_terms", 1.0, _count_in(text, TECHNICAL_TERMS) >= 1),
            # The only negative signal: explicit markers of a lookup/extraction
            # task pull the score back down.
            Signal("simple_markers", -1.5, _any_in(text, SIMPLE_MARKERS)),
        )
        return sum(signal.contribution for signal in signals), signals

    def route(self, task: Task, client: ModelClient) -> Decision:
        total, signals = self.score(task.prompt)
        fired = [signal.name for signal in signals if signal.hit] or ["none"]

        if total >= self.complex_threshold:
            tier = "complex"
        elif total >= self.medium_threshold:
            tier = "medium"
        else:
            tier = "simple"

        return Decision(
            model=self.tiers[tier],
            why=f"score={total:.1f} -> {tier} [{', '.join(fired)}]",
            spends=(),  # the entire point: routing cost is exactly zero
        )


def _any_in(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _count_in(text: str, needles: tuple[str, ...]) -> int:
    return sum(needle in text for needle in needles)
