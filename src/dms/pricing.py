"""Exact, dated cost arithmetic for Claude models.

Two design choices worth defending in the talk:

1. **Decimal, never float.** Money is decimal; binary floats drift. This mirrors
   the approach already used in an internal cost-tracking tool that predates this repo.
2. **Dated snapshots, not a flat table.** The Sonnet 5 introductory rate
   ($2/$10) lapses 2026-08-31 and reverts to $3/$15. A flat table silently
   under-reports every cost computed from September onward.

Cost formula, with the cache multipliers applied to the model's *input* rate:

    cost = input_tokens                * P_in
         + cache_creation_input_tokens * P_in * W     # W = 1.25 (5m) | 2.00 (1h)
         + cache_read_input_tokens     * P_in * 0.10
         + output_tokens               * P_out
    (all of it halved when the request went through the Batches API)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

TOKENS_PER_MILLION = Decimal(1_000_000)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "model-pricing.json"


class CacheTTL(StrEnum):
    """Cache lifetime. The write premium differs; the read discount does not."""

    FIVE_MINUTES = "5m"
    ONE_HOUR = "1h"


@dataclass(frozen=True, slots=True)
class Rate:
    """Per-million-token prices for one model at one point in time."""

    model: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    # OpenAI publishes an explicit cached-input rate; Anthropic derives one from
    # a multiplier on the input rate. When a provider states the rate directly we
    # must use it -- applying Anthropic's 0.10x to a GPT model would be wrong.
    cached_input_usd_per_million: Decimal | None = None

    def input_usd(self, tokens: int) -> Decimal:
        return self.input_usd_per_million * Decimal(tokens) / TOKENS_PER_MILLION

    def output_usd(self, tokens: int) -> Decimal:
        return self.output_usd_per_million * Decimal(tokens) / TOKENS_PER_MILLION


@dataclass(frozen=True, slots=True)
class PriceBook:
    """Loaded pricing config. Immutable; construct once and share."""

    snapshots: tuple[dict[str, Any], ...]
    cache_read_multiplier: Decimal
    cache_write_multipliers: dict[CacheTTL, Decimal]
    batch_multiplier: Decimal
    min_cache_prefix: dict[str, int]
    context_window: dict[str, int]

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        raw = json.loads((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
        _validate(raw)
        multipliers = raw["cache_multipliers"]
        return cls(
            snapshots=tuple(
                sorted(raw["snapshots"], key=lambda s: _parse_time(s["effective_from"]))
            ),
            cache_read_multiplier=Decimal(multipliers["read"]),
            cache_write_multipliers={
                CacheTTL.FIVE_MINUTES: Decimal(multipliers["write_5m"]),
                CacheTTL.ONE_HOUR: Decimal(multipliers["write_1h"]),
            },
            batch_multiplier=Decimal(raw["batch_multiplier"]),
            min_cache_prefix=_without_notes(raw["min_cache_prefix_tokens"]),
            context_window=_without_notes(raw["context_window"]),
        )

    # ------------------------------------------------------------------ lookup

    def _snapshot(self, at: str | datetime) -> dict[str, Any]:
        """Latest snapshot effective at or before `at`."""
        when = _parse_time(at) if isinstance(at, str) else at
        eligible = [
            snap
            for snap in self.snapshots
            if _parse_time(snap["effective_from"]) <= when
        ]
        if not eligible:
            raise ValueError(f"no pricing snapshot is effective at {when.isoformat()}")
        return eligible[-1]

    def resolve(self, model: str, at: str | datetime = "now") -> str:
        """Map an alias ('opus') to a full model id ('claude-opus-5').

        A `codex-cli/<model>` id costs against `<model>`'s published rates. Spend
        there lands on a ChatGPT subscription rather than a metered key, so the
        figure is what the equivalent API call would cost -- comparable, not a bill.
        """
        snapshot = self._snapshot(_now_if_needed(at))
        if model in snapshot["models"]:
            return model

        # Namespaced ids -- "codex-cli/gpt-5.6-sol" (this repo) or
        # "openai/gpt-5.6-sol" (OpenRouter) -- price against the bare model.
        # The transport differs; the per-token rates do not.
        if "/" in model:
            bare = model.split("/", 1)[1]
            if bare in snapshot["models"]:
                return bare
            model = snapshot.get("aliases", {}).get(bare, bare)
            if model in snapshot["models"]:
                return model

        return snapshot.get("aliases", {}).get(model, model)

    def rate(self, model: str, at: str | datetime = "now") -> Rate:
        when = _now_if_needed(at)
        snapshot = self._snapshot(when)
        model_id = self.resolve(model, when)
        entry = snapshot["models"].get(model_id)
        if entry is None:
            raise KeyError(
                f"{model!r} is not priced in snapshot {snapshot['id']!r}; "
                f"known models: {', '.join(sorted(snapshot['models']))}"
            )
        explicit = entry.get("cached_input_usd_per_million")
        return Rate(
            model=model_id,
            input_usd_per_million=Decimal(entry["input_usd_per_million"]),
            output_usd_per_million=Decimal(entry["output_usd_per_million"]),
            cached_input_usd_per_million=Decimal(explicit) if explicit else None,
        )

    def models(self, at: str | datetime = "now") -> tuple[str, ...]:
        return tuple(sorted(self._snapshot(_now_if_needed(at))["models"]))

    # ------------------------------------------------------------------ costing

    def cost_usd(
        self,
        usage: Any,
        model: str,
        *,
        at: str | datetime = "now",
        cache_ttl: CacheTTL = CacheTTL.FIVE_MINUTES,
        batch: bool = False,
    ) -> Decimal:
        """Exact dollar cost of one call's token usage.

        `usage` is a UsageRecord (duck-typed so tests can pass any shape with the
        four token fields).
        """
        rate = self.rate(model, at)
        write_multiplier = self.cache_write_multipliers[CacheTTL(cache_ttl)]

        if rate.cached_input_usd_per_million is not None:
            # Provider states the cached rate outright (OpenAI). Caching there is
            # automatic with no write premium, so a cache write is billed as
            # ordinary input rather than at 1.25x/2x.
            cache_read = (
                rate.cached_input_usd_per_million
                * Decimal(usage.cache_read_input_tokens)
                / TOKENS_PER_MILLION
            )
            cache_write = rate.input_usd(usage.cache_creation_input_tokens)
        else:
            cache_read = (
                rate.input_usd(usage.cache_read_input_tokens) * self.cache_read_multiplier
            )
            cache_write = (
                rate.input_usd(usage.cache_creation_input_tokens) * write_multiplier
            )

        total = (
            rate.input_usd(usage.input_tokens)
            + cache_write
            + cache_read
            + rate.output_usd(usage.output_tokens)
        )
        return total * self.batch_multiplier if batch else total

    # ------------------------------------------- the routing/caching interaction

    def min_cache_prefix_tokens(self, model: str, at: str | datetime = "now") -> int:
        """Smallest prefix this model will cache.

        Not monotonic with price: Opus 5 caches from 512 tokens, Haiku 4.5 needs
        4096. The cheap model is the harder one to cache.
        """
        model_id = self.resolve(model, _now_if_needed(at))
        try:
            return self.min_cache_prefix[model_id]
        except KeyError:
            # Providers with automatic caching publish no floor; nothing to clear.
            if self.rate(model_id, at).cached_input_usd_per_million is not None:
                return 0
            raise KeyError(f"no cache-floor recorded for {model!r}") from None

    def will_cache(self, model: str, prefix_tokens: int, at: str | datetime = "now") -> bool:
        """Whether a prefix of this size actually caches.

        Below the floor the API does not error -- it returns
        `cache_creation_input_tokens: 0` and you keep paying full price forever.
        """
        return prefix_tokens >= self.min_cache_prefix_tokens(model, at)


# ------------------------------------------------------------------------ helpers


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _now_if_needed(at: str | datetime) -> datetime:
    if at == "now":
        return datetime.now(timezone.utc)
    return _parse_time(at) if isinstance(at, str) else at


def _without_notes(mapping: dict[str, Any]) -> dict[str, Any]:
    """Drop the `_note` documentation keys embedded in the JSON."""
    return {k: v for k, v in mapping.items() if not k.startswith("_")}


def _validate(raw: Any) -> None:
    """Fail fast and loudly -- a wrong price silently corrupts every number."""
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("pricing config must be an object with schema_version == 1")
    for key in ("cache_multipliers", "batch_multiplier", "snapshots"):
        if key not in raw:
            raise ValueError(f"pricing config is missing {key!r}")
    snapshots = raw["snapshots"]
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("pricing config needs at least one snapshot")
    for snapshot in snapshots:
        for key in ("id", "effective_from", "models"):
            if key not in snapshot:
                raise ValueError(f"snapshot is missing {key!r}")
        for model, entry in snapshot["models"].items():
            if model.startswith("_"):
                continue
            for key in ("input_usd_per_million", "output_usd_per_million"):
                value = entry.get(key)
                if not isinstance(value, str):
                    raise ValueError(
                        f"{snapshot['id']}.{model}.{key} must be a decimal STRING "
                        "(floats lose precision on money)"
                    )
                if Decimal(value) < 0:
                    raise ValueError(f"{snapshot['id']}.{model}.{key} must be >= 0")
