"""The dispatcher: pick a model per request, and account for every call.

This is the production path. It has no dependency on the benchmark's `Task`, no
access to an answer key, and no grader -- the verifier is the model's own
judgement of its own output, which is all you have in production.

Two strategies, and the difference between them is measured rather than assumed:

* **cascade** (default) answers on the low model, asks the low model whether that
  answer is right, and escalates only on doubt. Measured at 59% cheaper than
  always-high, escalating 14% of requests against a true need of 14%.
* **heuristic** scores the prompt before generating anything. Measured at 34%
  cheaper. Same decision quality on the requests that mattered, but it overspent
  on 55% of the requests where the two models agreed anyway.

Cascade costs an extra round trip and cannot stream, so streaming requests fall
back to the pre-request strategy. That is a real trade, not an implementation
gap: the verifier needs the complete cheap answer before it can judge.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from dms.dispatch.affinity import SessionAffinity
from dms.dispatch.config import DispatchConfig
from dms.dispatch.providers import (
    Completion,
    ProviderRegistry,
    Request,
)
from dms.pricing import PriceBook
from dms.routers.heuristic import HeuristicRouter
from dms.usage import UsageRecord


@dataclass(frozen=True, slots=True)
class Leg:
    """One upstream call made while serving a request."""

    model: str
    role: str  # answer | verify | escalation | retry
    usage: UsageRecord
    cost_usd: Decimal
    latency_ms: float
    stop_reason: str = "end_turn"


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """What the dispatcher returned, and everything it spent getting there."""

    text: str
    model: str
    why: str
    legs: tuple[Leg, ...]
    request_id: str
    strategy: str
    session_id: str | None = None
    stop_reason: str = "end_turn"
    completion: Completion | None = None

    @property
    def usage(self) -> UsageRecord:
        return sum((leg.usage for leg in self.legs), UsageRecord())

    @property
    def cost_usd(self) -> Decimal:
        return sum((leg.cost_usd for leg in self.legs), Decimal(0))

    @property
    def latency_ms(self) -> float:
        return sum(leg.latency_ms for leg in self.legs)

    @property
    def escalated(self) -> bool:
        return any(leg.role == "escalation" for leg in self.legs)

    @property
    def overhead_usd(self) -> Decimal:
        """Spend that bought no part of the returned answer."""
        wasted = [
            leg
            for leg in self.legs
            if leg.role == "verify" or (leg.role == "answer" and self.escalated)
        ]
        return sum((leg.cost_usd for leg in wasted), Decimal(0))

    def to_log(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "strategy": self.strategy,
            "model": self.model,
            "why": self.why,
            "escalated": self.escalated,
            "stop_reason": self.stop_reason,
            "legs": [
                {
                    "model": leg.model,
                    "role": leg.role,
                    "cost_usd": str(leg.cost_usd),
                    "latency_ms": round(leg.latency_ms, 1),
                    **leg.usage.to_dict(),
                }
                for leg in self.legs
            ],
            "total_cost_usd": str(self.cost_usd),
            "overhead_usd": str(self.overhead_usd),
            "total_latency_ms": round(self.latency_ms, 1),
        }


class Dispatcher:
    """Routes one request to a low or high model and reports what it cost."""

    def __init__(
        self,
        config: DispatchConfig | None = None,
        *,
        book: PriceBook | None = None,
        providers: ProviderRegistry | None = None,
        affinity: SessionAffinity | None = None,
        on_result=None,
    ) -> None:
        self.config = config or DispatchConfig()
        self.book = book or PriceBook.load()
        self.providers = providers or ProviderRegistry(
            cache_system=self.config.cache_system_prompt,
            cache_ttl=self.config.cache_ttl,
            book=self.book,
        )
        self.affinity = affinity or SessionAffinity(self.config.affinity_ttl_seconds)
        self._scorer = HeuristicRouter(medium_threshold=self.config.medium_threshold)
        self._on_result = on_result

        self.requests_served = 0
        self.total_cost_usd = Decimal(0)

    # ------------------------------------------------------------------- public

    def dispatch(
        self, request: Request, *, session_id: str | None = None
    ) -> DispatchResult:
        strategy = (
            self.config.streaming_strategy if request.stream else self.config.strategy
        )
        request_id = uuid.uuid4().hex[:12]

        pinned = self.affinity.get(session_id) if self.config.session_affinity else None
        if pinned is not None:
            result = self._single(
                request,
                model=pinned.model,
                why=f"session affinity: {pinned.reason}",
                strategy=strategy,
                request_id=request_id,
                session_id=session_id,
            )
        elif strategy == "cascade":
            result = self._cascade(request, request_id, session_id)
        else:
            model, why = self._choose(request, strategy)
            result = self._single(
                request,
                model=model,
                why=why,
                strategy=strategy,
                request_id=request_id,
                session_id=session_id,
            )

        self._remember(result, session_id)
        self._record(result)
        return result

    def stream(
        self, request: Request, *, session_id: str | None = None
    ) -> tuple[str, Iterator[str], list[UsageRecord]]:
        """Choose a model, then stream from it.

        Returns (model, token iterator, usage_sink). The sink is empty until the
        iterator is exhausted, at which point it holds the final usage -- call
        `bill_stream()` then, or streaming traffic is never counted.

        Only pre-request strategies can serve this path; see the module docstring.
        """
        pinned = self.affinity.get(session_id) if self.config.session_affinity else None
        if pinned is not None:
            model, why = pinned.model, f"session affinity: {pinned.reason}"
        else:
            model, why = self._choose(request, self.config.streaming_strategy)
        self.affinity.set(session_id, model, why)

        provider = self.providers.for_model(model)
        sink: list[UsageRecord] = []
        return model, provider.stream(model, request, sink), sink

    def bill_stream(self, model: str, usage_sink: list[UsageRecord], why: str) -> Leg | None:
        """Record a drained stream. Returns the billed leg, or None if the
        provider reported no usage."""
        if not usage_sink:
            self.requests_served += 1
            return None
        usage = sum(usage_sink, UsageRecord())
        leg = Leg(
            model=model,
            role="answer",
            usage=usage,
            cost_usd=self.book.cost_usd(usage, model),
            latency_ms=0.0,
        )
        self.requests_served += 1
        self.total_cost_usd += leg.cost_usd
        return leg

    # ---------------------------------------------------------------- strategies

    def _choose(self, request: Request, strategy: str) -> tuple[str, str]:
        """Pre-request model choice. No tokens spent."""
        if strategy == "always_low":
            return self.config.low_model, "fixed: always low"
        if strategy == "always_high":
            return self.config.high_model, "fixed: always high"

        score, signals = self._scorer.score(request.text_prompt)
        fired = [s.name for s in signals if s.hit] or ["none"]
        # Two-tier: anything above the band goes high. Ambiguity resolves toward
        # quality, never toward cost.
        if score >= self.config.medium_threshold:
            return self.config.high_model, f"heuristic {score:.1f} -> high [{','.join(fired)}]"
        return self.config.low_model, f"heuristic {score:.1f} -> low [{','.join(fired)}]"

    def _cascade(
        self, request: Request, request_id: str, session_id: str | None
    ) -> DispatchResult:
        low, high = self.config.low_model, self.config.high_model

        cheap, cheap_leg = self._call(low, request, role="answer")
        legs = [cheap_leg]

        # A refusal or an empty body from the cheap model is not an answer.
        # Escalate without wasting a verification call on nothing.
        if cheap.refused or cheap.empty:
            why = f"low model returned {cheap.stop_reason or 'empty'} -> escalate"
            # No usable cheap answer to fall back to, so pass None.
            return self._escalate(request, legs, why, request_id, session_id, high, None)

        accepted, verify_leg, verdict = self._verify(request, cheap.text, low)
        legs.append(verify_leg)
        if accepted:
            return DispatchResult(
                text=cheap.text,
                model=low,
                why=f"verifier accepted the low-model answer ({verdict})",
                legs=tuple(legs),
                request_id=request_id,
                strategy="cascade",
                session_id=session_id,
                stop_reason=cheap.stop_reason,
                completion=cheap,
            )

        return self._escalate(
            request, legs, f"verifier rejected ({verdict}) -> escalate",
            request_id, session_id, high, cheap,
        )

    def _escalate(
        self,
        request: Request,
        legs: list[Leg],
        why: str,
        request_id: str,
        session_id: str | None,
        high: str,
        cheap: Completion | None,
    ) -> DispatchResult:
        strong, strong_leg = self._call(high, request, role="escalation")
        legs.append(strong_leg)

        # Measured here: the HIGH model refused a benign shell question the low
        # model answered fine. When the escalation target refuses, the cheap
        # answer we already paid for is the better response -- if we have one.
        if (
            strong.refused
            and self.config.retry_other_tier_on_refusal
            and cheap is not None
            and not cheap.empty
        ):
            return DispatchResult(
                text=cheap.text,
                model=self.config.low_model,
                why=f"{why}; high model refused ({strong.refusal_category}) "
                "-> kept the low-model answer we had already paid for",
                legs=tuple(legs),
                request_id=request_id,
                strategy="cascade",
                session_id=session_id,
                stop_reason="end_turn",
                completion=cheap,
            )

        return DispatchResult(
            text=strong.text,
            model=high,
            why=why,
            legs=tuple(legs),
            request_id=request_id,
            strategy="cascade",
            session_id=session_id,
            stop_reason=strong.stop_reason,
            completion=strong,
        )

    def _single(
        self,
        request: Request,
        *,
        model: str,
        why: str,
        strategy: str,
        request_id: str,
        session_id: str | None,
    ) -> DispatchResult:
        completion, leg = self._call(model, request, role="answer")
        legs = [leg]

        if completion.refused and self.config.retry_other_tier_on_refusal:
            other = (
                self.config.high_model
                if model == self.config.low_model
                else self.config.low_model
            )
            retry, retry_leg = self._call(other, request, role="retry")
            legs.append(retry_leg)
            if not retry.refused:
                return DispatchResult(
                    text=retry.text,
                    model=other,
                    why=f"{why}; {model} refused ({completion.refusal_category}) "
                    f"-> retried on {other}",
                    legs=tuple(legs),
                    request_id=request_id,
                    strategy=strategy,
                    session_id=session_id,
                    stop_reason=retry.stop_reason,
                    completion=retry,
                )
            completion = retry
            model = other

        return DispatchResult(
            text=completion.text,
            model=model,
            why=why,
            legs=tuple(legs),
            request_id=request_id,
            strategy=strategy,
            session_id=session_id,
            stop_reason=completion.stop_reason,
            completion=completion,
        )

    # ------------------------------------------------------------------ helpers

    def _verify(
        self, request: Request, answer: str, model: str
    ) -> tuple[bool, Leg, str]:
        """Ask the cheap model whether its own answer is right.

        No grader, no answer key -- this is the model's judgement of its own
        output, which is the only signal available in production.
        """
        probe = Request(
            messages=(
                {
                    "role": "user",
                    "content": f"Question:\n{request.text_prompt}\n\n"
                    f"Proposed answer:\n{answer}",
                },
            ),
            system=self.config.verify_system,
            max_tokens=self.config.verify_max_tokens,
        )
        verdict, leg = self._call(model, probe, role="verify")
        text = verdict.text.strip().lower()
        # Fail toward quality: an unreadable verdict escalates rather than
        # silently accepting an answer nobody vouched for.
        accepted = text.startswith("yes")
        return accepted, leg, text[:20] or "empty"

    def _call(self, model: str, request: Request, *, role: str) -> tuple[Completion, Leg]:
        provider = self.providers.for_model(model)
        started = time.perf_counter()
        completion = provider.complete(model, request)
        latency = completion.latency_ms or (time.perf_counter() - started) * 1000

        return completion, Leg(
            model=model,
            role=role,
            usage=completion.usage,
            cost_usd=self.book.cost_usd(completion.usage, model),
            latency_ms=latency,
            stop_reason=completion.stop_reason,
        )

    def _remember(self, result: DispatchResult, session_id: str | None) -> None:
        if self.config.session_affinity and session_id:
            self.affinity.set(session_id, result.model, result.why)

    def _record(self, result: DispatchResult) -> None:
        self.requests_served += 1
        self.total_cost_usd += result.cost_usd
        if self._on_result is not None:
            self._on_result(result)
