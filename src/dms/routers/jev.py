"""Jev as a pre-request router: a typed judgment model deciding low vs high.

TypeSafe's Jev (System One) is not an LLM. It generates no text: it takes a
`state` and typed questions and returns calibrated answers -- a Score with a
confidence here. That makes it a candidate for exactly the job the LLM
classifier does badly in this repo: one cheap, fast judgment per request.

Measured elsewhere (catalog dedup, 2026-09-22), Jev is a **strong ranker and a
weak judge**: high AUC, but its verdict at a fixed cut is only moderately
precise, and confidence is the lever. So the decision rule here is deliberately
the simplest honest one -- round the Score, no fitted threshold -- and the
evaluation reports ranking and confidence separately.

The decision rule and the question wording are fixed in
`docs/jev-router-prereg.md`, written before any Jev response existed.

Phase 1 of adopting Jev keeps the LLM option: this router sits *beside*
`LLMClassifierRouter`, selectable, never replacing it by default.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dms.client import ModelClient
from dms.routers.base import TIER_MODELS, Decision, Router, Spend
from dms.usage import UsageRecord
from dms.workload import Task

JEV_URL = "https://api.typesafe.ai/v1/systemone"
# Pinned. `jev-latest` moves, and a cache keyed on it would keep serving
# answers from whatever model the alias used to mean.
JEV_MODEL = "jev-1.13.0"
JEV_TIMEOUT_SECONDS = 60
JEV_MAX_ATTEMPTS = 4
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})

ROUTE_INSTRUCTIONS = (
    "How likely is a small, fast AI model to answer this question incorrectly? "
    "Judge the question itself; do not try to answer it."
)

# Situations on ONE axis -- the risk that a fast answer is wrong -- low first.
# Each stands alone: Jev never sees a level's index or its neighbours.
LEVELS: tuple[str, ...] = (
    "Answering it only requires stating one well-known fact, or copying a value "
    "that already appears in the question.",
    "Answering it requires applying one familiar rule or definition once, leaving "
    "little room for a mistake.",
    "Answering it requires carrying out several dependent steps exactly, such as "
    "tracing a loop, counting through a sequence, or evaluating code, where a "
    "single slip produces a wrong answer.",
    "Answering it hinges on a subtle distinction or precise terminology where a "
    "plausible-sounding answer is usually wrong, or on a long exact derivation.",
)

# Pre-registered: route HIGH at level 2 or above. Not fitted -- 36 items with 6
# positives cannot support a fitted threshold.
HIGH_LEVEL = 2

DIAGNOSTIC_NOULS = {
    "multi_step": (
        "Does answering this require working through several dependent steps "
        "exactly, such as simulating a loop, counting through a sequence, or "
        "evaluating code, rather than recalling a single fact?"
    ),
    "trap": (
        "Is this a question where a confident, plausible-sounding answer is "
        "commonly wrong, because of subtle terminology, easily confused "
        "standards, or a trick detail?"
    ),
}

Asker = Callable[[Any, dict[str, Any], str], dict[str, Any]]


class JevCredentialMissing(RuntimeError):
    """No TYPESAFE_API_KEY. A configuration error, raised loudly: quietly routing
    every request high because a key was never set is a silent failure."""


def nearest_level(score: float, n_levels: int) -> int:
    """Round a Score to its level, half UP.

    Python's round() is banker's rounding (2.5 -> 2), but the cuts are at 0.5,
    1.5, 2.5. A Score is a probability-weighted mean of level indices, never a
    fraction between levels, so rounding is the whole decision.
    """
    return max(0, min(n_levels - 1, int(score + 0.5)))


def default_asker(state: Any, questions: dict[str, Any], model: str) -> dict[str, Any]:
    """POST to Jev with capped exponential backoff on retryable statuses.

    The key is read from the environment at call time and never logged,
    echoed, cached or placed in an error message.
    """
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise JevCredentialMissing(
            "TYPESAFE_API_KEY is not set; load it with "
            "`source ~/.config/typesafe/env` in the same shell as the call"
        )
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()

    for attempt in range(JEV_MAX_ATTEMPTS):
        request = urllib.request.Request(
            JEV_URL,
            method="POST",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=JEV_TIMEOUT_SECONDS) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # 401 and 422 are our bugs: read the body, fix the request, never retry.
            if exc.code not in RETRYABLE_STATUS or attempt == JEV_MAX_ATTEMPTS - 1:
                detail = exc.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"jev HTTP {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            if attempt == JEV_MAX_ATTEMPTS - 1:
                raise RuntimeError(f"jev connection failed: {exc.reason}") from None
        time.sleep(min(8.0, 0.5 * 2**attempt))

    raise RuntimeError("jev: retries exhausted")  # pragma: no cover - loop always returns/raises


@dataclass(frozen=True, slots=True)
class JevRouter(Router):
    """Route by asking Jev how error-prone a prompt is for a small model."""

    tiers: dict[str, str] = field(default_factory=lambda: dict(TIER_MODELS))
    asker: Asker = default_asker
    model: str = JEV_MODEL
    name: str = "jev"

    def questions(self) -> dict[str, Any]:
        """One Score decides; two Nouls ride along as diagnostics.

        Bundling costs almost nothing: questions in one request are scored
        independently, and TypeSafe measured 13 bundled questions at 12.2x
        cheaper than 13 separate calls with identical answers.
        """
        return {
            "route": {
                "type": "score",
                "instructions": ROUTE_INSTRUCTIONS,
                "criteria": list(LEVELS),
            },
            **{
                key: {"type": "noul", "instructions": text}
                for key, text in DIAGNOSTIC_NOULS.items()
            },
        }

    def route(self, task: Task, client: ModelClient) -> Decision:
        high = self.tiers["complex"]
        low = self.tiers["simple"]

        # State is the prompt ONLY. Never the expected answer, never the
        # difficulty label: this leaves the machine for a third party.
        try:
            response = self.asker(task.prompt, self.questions(), self.model)
        except JevCredentialMissing:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded in `why`, fails toward quality
            return Decision(model=high, why=f"jev error ({exc}) -> fallback high")

        answer = (response.get("answers") or {}).get("route") or {}
        score = answer.get("score")
        if score is None:
            return Decision(model=high, why="jev returned no route score -> fallback high")

        level = nearest_level(float(score), len(LEVELS))
        confidence = float(answer.get("confidence") or 0.0)
        spend = Spend(
            model=response.get("model", self.model),
            usage=UsageRecord(
                input_tokens=int((response.get("usage") or {}).get("input_tokens") or 0)
            ),
            role="router",  # billed to this strategy, like the LLM classifier's call
        )
        chosen = high if level >= HIGH_LEVEL else low
        return Decision(
            model=chosen,
            why=f"jev score={float(score):.2f} -> level {level} "
            f"({confidence:.2f} conf) -> {'high' if chosen == high else 'low'}",
            spends=(spend,),
        )
