"""One call to a Claude model, in four modes.

    LIVE      -- call the API, keep nothing
    RECORD    -- call the API, write a fixture
    REPLAY    -- read fixtures only; never touches the network
    SIMULATE  -- no API at all; deterministic synthetic usage from an explicit
                 assumptions file

SIMULATE exists because this repo must be runnable and testable with no
credential. Its numbers are ESTIMATES, not measurements, and every Call it
produces carries `simulated=True` so the report and the slides can say so. Do
not quote a simulated figure as a measured one.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from dms.pricing import CacheTTL, PriceBook
from dms.replay import FixtureStore
from dms.usage import UsageRecord

SIMULATION_CONFIG = Path(__file__).resolve().parents[2] / "config" / "simulation.json"
DEFAULT_MAX_TOKENS = 1024

# `output_config.effort` is GA on the 4.6+ generation and REJECTED on Haiku 4.5
# and Sonnet 4.5. Passing it to Haiku is a 400, so it is dropped per model rather
# than per call site -- otherwise every caller has to remember the matrix.
EFFORT_CAPABLE = frozenset(
    {
        "claude-opus-5", "claude-fable-5", "claude-mythos-5",
        "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
        "claude-sonnet-5", "claude-sonnet-4-6",
    }
)

# Models where omitting `thinking` still thinks (adaptive is the default). On
# those, `max_tokens` is a hard cap on thinking PLUS response text -- so a tight
# cap silently truncates the answer and the model looks wrong rather than
# throttled. Every answering call must leave headroom for both.
THINKS_BY_DEFAULT = frozenset({"claude-opus-5", "claude-fable-5", "claude-mythos-5"})


class Mode(StrEnum):
    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"
    SIMULATE = "simulate"


class SpendLimitExceeded(RuntimeError):
    """Raised before a call that would breach --max-spend."""


class MissingCredential(RuntimeError):
    """Raised when a live/record run has no ANTHROPIC_API_KEY."""


@dataclass(frozen=True, slots=True)
class SimulationHint:
    """What the simulator needs to fake a plausible answer.

    Ignored entirely outside SIMULATE mode -- it never reaches the API.
    """

    difficulty: str
    expected: str = ""


@dataclass(frozen=True, slots=True)
class Call:
    """The result of one model call."""

    model: str
    text: str
    usage: UsageRecord = field(default_factory=UsageRecord)
    latency_ms: float = 0.0
    simulated: bool = False
    stop_reason: str = "end_turn"
    refusal_category: str = ""

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "text": self.text,
            "usage": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 3),
            "simulated": self.simulated,
            "stop_reason": self.stop_reason,
            "refusal_category": self.refusal_category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            model=data["model"],
            text=data["text"],
            usage=UsageRecord.from_dict(data.get("usage", {})),
            latency_ms=float(data.get("latency_ms", 0.0)),
            simulated=bool(data.get("simulated", False)),
            stop_reason=data.get("stop_reason", "end_turn"),
            refusal_category=data.get("refusal_category", ""),
        )


class ModelClient:
    """Mode-aware Claude client that accounts for every token it spends."""

    def __init__(
        self,
        *,
        mode: Mode | str = Mode.SIMULATE,
        store: FixtureStore | None = None,
        book: PriceBook | None = None,
        api: Any | None = None,
        max_spend_usd: Decimal | str | None = None,
        simulation_config: Path | None = None,
    ) -> None:
        self.mode = Mode(mode)
        self.book = book or PriceBook.load()
        self.store = store or FixtureStore(
            Path(__file__).resolve().parents[2] / "fixtures"
        )
        self.max_spend_usd = Decimal(max_spend_usd) if max_spend_usd is not None else None
        self._api = api
        self._sim = _Simulator(simulation_config or SIMULATION_CONFIG)

        self.calls_made = 0
        self.total_usage = UsageRecord()
        self.total_spend_usd = Decimal(0)
        self.fixture_misses = 0
        # A truncated answer grades as wrong. Counting these separately keeps a
        # token cap from being mistaken for a capability gap.
        self.truncated_calls: list[tuple[str, str]] = []
        # A safety refusal returns HTTP 200 with stop_reason="refusal", empty
        # content, and zero billed output. Graded naively it looks like the model
        # got the answer wrong, which would understate whichever tier refused.
        self.refusals: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ the call

    def fixture_key(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str | None = None,
        cache_system: bool = False,
        cache_ttl: CacheTTL = CacheTTL.FIVE_MINUTES,
    ) -> str:
        """Fixture key for a request. Public so tooling and tests derive it the
        same way `complete()` does rather than reimplementing the option set."""
        return self.store.key(
            model=self.book.resolve(model),
            system=system,
            prompt=prompt,
            options={
                "max_tokens": max_tokens,
                "effort": effort,
                "cache_system": cache_system,
                "cache_ttl": str(cache_ttl),
            },
        )

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str | None = None,
        cache_system: bool = False,
        cache_ttl: CacheTTL = CacheTTL.FIVE_MINUTES,
        hint: SimulationHint | None = None,
    ) -> Call:
        model_id = self.book.resolve(model)
        key = self.fixture_key(
            model=model_id,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            cache_system=cache_system,
            cache_ttl=cache_ttl,
        )

        # The budget applies in every mode, so you can dry-run a spend plan in
        # SIMULATE before pointing RECORD at a real key.
        self._guard_spend(model_id, prompt, system, max_tokens, cache_ttl)

        match self.mode:
            case Mode.REPLAY:
                call = self.store.get(key)
                if call is None:
                    self.fixture_misses += 1
                    raise LookupError(
                        f"no fixture for {model_id} / {_preview(prompt)!r}. "
                        "Run once with --mode record (needs ANTHROPIC_API_KEY), "
                        "or use --mode simulate."
                    )
            case Mode.SIMULATE:
                call = self._sim.complete(
                    model=model_id,
                    prompt=prompt,
                    system=system,
                    hint=hint,
                    cache_system=cache_system,
                    book=self.book,
                )
            case Mode.LIVE | Mode.RECORD:
                call = self._live(
                    model_id=model_id,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    effort=effort,
                    cache_system=cache_system,
                    cache_ttl=cache_ttl,
                )
                if self.mode is Mode.RECORD:
                    self.store.put(key, call)

        self._account(call, cache_ttl)
        return call

    # ------------------------------------------------------------------ internals

    def _live(
        self,
        *,
        model_id: str,
        prompt: str,
        system: str | None,
        max_tokens: int,
        effort: str | None,
        cache_system: bool,
        cache_ttl: CacheTTL,
    ) -> Call:
        api = self._api or _default_api()
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            block: dict[str, Any] = {"type": "text", "text": system}
            if cache_system:
                ttl = CacheTTL(cache_ttl)
                block["cache_control"] = (
                    {"type": "ephemeral"}
                    if ttl is CacheTTL.FIVE_MINUTES
                    else {"type": "ephemeral", "ttl": "1h"}
                )
            kwargs["system"] = [block]
        if effort is not None and model_id in EFFORT_CAPABLE:
            kwargs["output_config"] = {"effort": effort}

        started = time.perf_counter()
        response = api.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000

        # stop_details is populated ONLY when stop_reason == "refusal"; it is
        # None for every other stop reason, so guard before reading it.
        stop_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", "") or "" if details else ""

        return Call(
            model=model_id,
            text=_text_of(response),
            usage=UsageRecord.from_response(response),
            latency_ms=latency_ms,
            simulated=False,
            stop_reason=stop_reason,
            refusal_category=category,
        )

    @staticmethod
    def thinks_by_default(model: str) -> bool:
        return model in THINKS_BY_DEFAULT

    def _guard_spend(
        self,
        model_id: str,
        prompt: str,
        system: str | None,
        max_tokens: int,
        cache_ttl: CacheTTL,
    ) -> None:
        """Refuse a call whose worst case would breach the budget."""
        if self.max_spend_usd is None:
            return
        worst_case = UsageRecord(
            input_tokens=_estimate_tokens((system or "") + prompt),
            output_tokens=max_tokens,
        )
        projected = self.total_spend_usd + self.book.cost_usd(
            worst_case, model_id, cache_ttl=cache_ttl
        )
        if projected > self.max_spend_usd:
            raise SpendLimitExceeded(
                f"next {model_id} call would put spend at ~${projected:.4f}, "
                f"over the ${self.max_spend_usd} cap after {self.calls_made} calls"
            )

    def _account(self, call: Call, cache_ttl: CacheTTL) -> None:
        self.calls_made += 1
        if call.stop_reason == "max_tokens":
            self.truncated_calls.append((call.model, call.text[:60]))
        elif call.stop_reason == "refusal":
            self.refusals.append((call.model, call.refusal_category))
        self.total_usage = self.total_usage + call.usage
        self.total_spend_usd += self.book.cost_usd(
            call.usage, call.model, cache_ttl=cache_ttl
        )


class _Simulator:
    """Deterministic synthetic responses driven by an explicit assumptions file.

    The assumptions live in config/simulation.json precisely so they can be
    argued with. Change a number there, rerun, and watch which conclusions move
    -- that sensitivity is itself the point.
    """

    def __init__(self, config_path: Path) -> None:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.chars_per_token: float = raw["chars_per_token"]
        self.accuracy: dict[str, dict[str, float]] = raw["accuracy_by_model_and_difficulty"]
        self.output_tokens: dict[str, list[int]] = raw["output_tokens_by_model"]
        self.latency_ms: dict[str, list[float]] = raw["latency_ms_by_model"]

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        hint: SimulationHint | None,
        cache_system: bool,
        book: PriceBook,
    ) -> Call:
        rng = random.Random(
            hashlib.sha256(f"{model}|{system}|{prompt}".encode()).hexdigest()
        )

        system_tokens = _estimate_tokens(system or "", self.chars_per_token)
        prompt_tokens = _estimate_tokens(prompt, self.chars_per_token)

        # The cache only engages above the model's floor -- below it the API
        # silently no-ops, which is exactly the trap this repo exists to show.
        cached = cache_system and book.will_cache(model, system_tokens)
        usage = UsageRecord(
            input_tokens=prompt_tokens if cached else prompt_tokens + system_tokens,
            output_tokens=_pick_int(rng, self.output_tokens.get(model, [80, 200])),
            cache_read_input_tokens=system_tokens if cached else 0,
        )

        difficulty = hint.difficulty if hint else "medium"
        correct = rng.random() < self._accuracy(model, difficulty)
        text = (hint.expected if correct else _wrong_answer(rng, hint.expected)) if hint else "(simulated)"

        return Call(
            model=model,
            text=text,
            usage=usage,
            latency_ms=_pick_float(rng, self.latency_ms.get(model, [500.0, 2000.0])),
            simulated=True,
        )

    def _accuracy(self, model: str, difficulty: str) -> float:
        by_difficulty = self.accuracy.get(model)
        if by_difficulty is None:
            raise KeyError(f"simulation.json has no accuracy row for {model!r}")
        return by_difficulty[difficulty]


# ------------------------------------------------------------------------ helpers


def _default_api() -> Any:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingCredential(
            "ANTHROPIC_API_KEY is not set. Use --mode simulate (no credential "
            "needed) or --mode replay against committed fixtures."
        )
    from anthropic import Anthropic

    return Anthropic()


def _text_of(response: Any) -> str:
    parts = [
        block.text
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
    ]
    return "".join(parts).strip()


def _estimate_tokens(text: str, chars_per_token: float = 3.6) -> int:
    """Rough token count for budgeting and simulation only.

    Never use this for a reported figure -- the real number comes from
    `response.usage`, or from `client.messages.count_tokens` pre-flight.
    tiktoken is wrong for Claude by 15-20% and much worse on code.
    """
    return max(1, round(len(text) / chars_per_token))


def _pick_int(rng: random.Random, bounds: list[int]) -> int:
    return rng.randint(int(bounds[0]), int(bounds[1]))


def _pick_float(rng: random.Random, bounds: list[float]) -> float:
    return round(rng.uniform(float(bounds[0]), float(bounds[1])), 1)


def _wrong_answer(rng: random.Random, expected: str) -> str:
    return f"__wrong_{rng.randrange(1000)}__" if expected else "(simulated miss)"


def _preview(text: str, limit: int = 60) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."
