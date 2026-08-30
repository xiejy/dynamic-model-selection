"""Token accounting for a single Claude API call.

The one thing everybody gets wrong: `usage.input_tokens` is the **uncached
remainder**, not the size of the prompt. A long-running agent can show
`input_tokens: 4000` while actually sending 400k tokens, the rest served from
cache. Always sum the three input fields.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Immutable token counts for one call. Add records to aggregate."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        """Total tokens sent, cached or not."""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Share of the prompt served from cache. 0.0 when nothing was sent."""
        if self.prompt_tokens == 0:
            return 0.0
        return self.cache_read_input_tokens / self.prompt_tokens

    @classmethod
    def from_response(cls, response: Any) -> Self:
        """Parse `response.usage`. Cache fields are absent on uncached calls."""
        usage = getattr(response, "usage", None)
        if usage is None:
            raise ValueError("response has no .usage -- cannot account for its cost")
        return cls(
            input_tokens=_field(usage, "input_tokens"),
            output_tokens=_field(usage, "output_tokens"),
            cache_creation_input_tokens=_field(usage, "cache_creation_input_tokens"),
            cache_read_input_tokens=_field(usage, "cache_read_input_tokens"),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            cache_creation_input_tokens=int(data.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(data.get("cache_read_input_tokens") or 0),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }

    def __add__(self, other: UsageRecord) -> UsageRecord:
        if not isinstance(other, UsageRecord):
            return NotImplemented
        return replace(
            self,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
        )

    def __radd__(self, other: UsageRecord | int) -> UsageRecord:
        # sum() seeds with 0, so the reflected op must absorb it.
        if other == 0:
            return self
        return self.__add__(other)  # type: ignore[arg-type]

    def __bool__(self) -> bool:
        return self.total_tokens > 0


def _field(usage: Any, name: str) -> int:
    """Read a usage field that may be absent or explicitly None."""
    return int(getattr(usage, name, 0) or 0)


EMPTY_USAGE = UsageRecord()
