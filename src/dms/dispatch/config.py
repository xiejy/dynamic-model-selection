"""Dispatcher configuration.

Every default here traces to a measurement in this repo rather than a guess;
the docstrings say which. Nothing is tuned by vibes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self

LOW_MODEL = "claude-haiku-4-5"
HIGH_MODEL = "claude-opus-5"

# Measured: rerouting mid-session forfeits the prompt cache, and a cache read is
# 0.10x input while a write is 1.25x. A reroute can therefore cost more than the
# tier gap it saves, so a session's first choice is pinned. One hour matches the
# long cache TTL; LiteLLM's session_affinity ships the same default.
AFFINITY_TTL_SECONDS = 3600

# The verifier is asked for one word. Anything longer is the model ignoring the
# instruction, and we only need enough tokens to read a yes or no.
VERIFY_MAX_TOKENS = 16

VERIFY_SYSTEM = (
    "You check answers. Given a question and a proposed answer, reply with "
    "exactly one word: yes if the answer is correct and complete, no if it is "
    "not. Reply with the single word only."
)


@dataclass(frozen=True, slots=True)
class DispatchConfig:
    """How the dispatcher behaves. Immutable; use `replace()` to derive variants."""

    low_model: str = LOW_MODEL
    high_model: str = HIGH_MODEL

    # cascade | heuristic | always_low | always_high
    strategy: str = "cascade"

    # Streaming cannot use the cascade: the verifier needs the complete cheap
    # answer before it can judge, so a streamed cascade would buffer the whole
    # response and destroy time-to-first-token. Streaming requests fall back to
    # a pre-request router, which decides before any tokens are generated.
    streaming_strategy: str = "heuristic"

    session_affinity: bool = True
    affinity_ttl_seconds: int = AFFINITY_TTL_SECONDS

    # A refusal returns HTTP 200 with empty content. Measured here: Opus 5
    # refused a benign POSIX shell question that Haiku answered fine, so the
    # escalation target can refuse work the cheap model would have done.
    # Retrying the other tier turns a dead response into a live one.
    retry_other_tier_on_refusal: bool = True

    # Agent clients send a large, stable system prompt on every single turn --
    # Codex's is ~13k tokens. Uncached that is the dominant cost of the whole
    # workload, far larger than anything model choice can recover. Caching it
    # costs 1.25x once and 0.10x thereafter.
    cache_system_prompt: bool = True
    cache_ttl: str = "5m"

    verify_max_tokens: int = VERIFY_MAX_TOKENS
    verify_system: str = VERIFY_SYSTEM

    # Heuristic bands. Anything above `medium_threshold` goes high in two-tier
    # mode: ambiguity resolves toward quality, never toward cost.
    medium_threshold: float = 1.0

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = {"cascade", "heuristic", "always_low", "always_high"}
        for name, value in (
            ("strategy", self.strategy),
            ("streaming_strategy", self.streaming_strategy),
        ):
            if value not in valid:
                raise ValueError(
                    f"{name}={value!r} is not one of {sorted(valid)}"
                )
        if self.streaming_strategy == "cascade":
            raise ValueError(
                "cascade cannot serve a streaming request: the verifier needs the "
                "complete cheap answer first, so nothing could be streamed until "
                "the whole response existed. Use a pre-request strategy."
            )
        if self.affinity_ttl_seconds < 0:
            raise ValueError("affinity_ttl_seconds must be >= 0")

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Self:
        env = environ if environ is not None else dict(os.environ)
        config = cls()
        if path := env.get("DMS_DISPATCH_CONFIG"):
            config = cls.from_file(Path(path))
        overrides: dict[str, Any] = {}
        if v := env.get("DMS_LOW_MODEL"):
            overrides["low_model"] = v
        if v := env.get("DMS_HIGH_MODEL"):
            overrides["high_model"] = v
        if v := env.get("DMS_STRATEGY"):
            overrides["strategy"] = v
        if v := env.get("DMS_SESSION_AFFINITY"):
            overrides["session_affinity"] = v.lower() not in {"0", "false", "no"}
        return replace(config, **overrides) if overrides else config

    @classmethod
    def from_file(cls, path: Path) -> Self:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown config keys: {', '.join(sorted(unknown))}; "
                f"known: {', '.join(sorted(known))}"
            )
        return cls(**data)

    def model_for(self, tier: str) -> str:
        if tier not in {"low", "high"}:
            raise ValueError(f"tier must be 'low' or 'high', got {tier!r}")
        return self.low_model if tier == "low" else self.high_model

    def tier_of(self, model: str) -> str:
        return "low" if model == self.low_model else "high"
