"""LLM-as-router: ask Haiku which tier a prompt belongs to.

This is the router people reach for first, and the one whose economics are most
often left unexamined. It is the only router here that spends tokens to decide,
so it is the only one where the bench's accounting can embarrass it -- which is
the point of including it.

The arithmetic to do on stage, before any benchmark: the classifier runs on
**100%** of traffic. Downgrading 40% of traffic from Opus ($5/$25) to Haiku
($1/$5) only pays if the classifier's own per-request cost is small against the
per-request saving. Cheap on a long prompt, ruinous on a short one -- and short
prompts are exactly the ones you were hoping to downgrade.

Latency has the same shape. ~300 ms of classifier is free against a single 2 s
answer, and is 9 s of pure tax across a 30-step agent loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from dms.client import ModelClient, SimulationHint
from dms.routers.base import TIER_MODELS, Decision, Router, Spend
from dms.workload import Task

ROUTER_MODEL = "claude-haiku-4-5"

CLASSIFIER_SYSTEM = """You are a routing classifier. Read the developer question and reply with exactly one word:

simple  - lookup, extraction, or single-step recall
medium  - one step of reasoning, code comprehension, or short generation
complex - multi-step reasoning, tracing state, or subtle systems knowledge

Reply with the tier word only."""

VALID_TIERS = ("simple", "medium", "complex")


@dataclass(frozen=True, slots=True)
class LLMClassifierRouter(Router):
    """Route by asking a cheap model to label the tier."""

    tiers: dict[str, str] = field(default_factory=lambda: dict(TIER_MODELS))
    router_model: str = ROUTER_MODEL
    fallback_tier: str = "complex"
    name: str = "llm-classifier"

    def route(self, task: Task, client: ModelClient) -> Decision:
        call = client.complete(
            model=self.router_model,
            prompt=task.prompt,
            system=CLASSIFIER_SYSTEM,
            max_tokens=8,  # one word; never let a router write an essay
            hint=SimulationHint(
                difficulty=task.difficulty,
                expected=_tier_for(task.difficulty),
            ),
        )
        spend = Spend(
            model=call.model,
            usage=call.usage,
            role="router",  # billed to this strategy, not hidden
            latency_ms=call.latency_ms,
            simulated=call.simulated,
        )

        tier = _parse_tier(call.text)
        if tier is None:
            # An unparseable classification must fail toward quality, not cost.
            return Decision(
                model=self.tiers[self.fallback_tier],
                why=f"unparseable tier {call.text.strip()[:20]!r} -> fallback",
                spends=(spend,),
                latency_ms=call.latency_ms,
            )

        return Decision(
            model=self.tiers[tier],
            why=f"classified {tier} ({call.usage.total_tokens} router tokens)",
            spends=(spend,),
            latency_ms=call.latency_ms,
        )


def _parse_tier(text: str) -> str | None:
    lowered = text.strip().lower()
    for tier in VALID_TIERS:
        if tier in lowered:
            return tier
    return None


def _tier_for(difficulty: str) -> str:
    """Ground-truth tier label, used only to simulate the classifier's answer."""
    return {"easy": "simple", "medium": "medium", "hard": "complex"}[difficulty]
