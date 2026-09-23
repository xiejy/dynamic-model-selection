"""Usage ledger: one JSON line per request, and the low-vs-high cost comparison.

Each line records which tier served the request, its tokens, status and latency
-- never prompt text, never credentials. Cost is computed at report time from
the price book, so a price change is a report change, not a data migration.

The comparison: every routed request is re-priced as if the model it asked
for had served the same tokens. That is an estimate -- a different model would
not produce exactly the same tokens -- but it is the like-for-like
counterfactual, with one correction. Prompt caches are per model, so routing
between two models leaves each one cold where a single model would have been
warm. The counterfactual bills those tokens as the cache reads they would have
been, so a switch's cold write counts against routing instead of passing as a
saving. Two cases are credited:

* within a session, after a switch: the conversation the previous request
  already sent (conversations only grow);
* at a session's opening, on a model routing had left unused: the prefix every
  session shares (Claude Code's system prompt and tools), sized by the largest
  cache read any session opened with -- a measured lower bound, zero when
  sessions share nothing (Codex's per-session cache).

Neither applies past the longest cache lifetime, when one model would have
started cold too.

Untouched requests (other models the client asked for) are counted but left out
of the comparison: they would have cost the same either way.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Any

from dms.pricing import PriceBook
from dms.usage import UsageRecord

DEFAULT_LEDGER = Path.home() / ".dms" / "usage.jsonl"
# The longest prompt-cache lifetime either provider offers (Anthropic's 1h TTL).
# Past it, an unswitched run would have started cold too.
CACHE_LIFETIME = timedelta(hours=1)


@dataclass(frozen=True, slots=True)
class UsageEvent:
    ts: str
    family: str
    session: str | None
    requested_model: str
    routed_model: str
    tier: str            # low | high | untouched
    turn: str            # new | continuation | other
    why: str
    status: int
    latency_ms: float
    usage: UsageRecord | None

    def to_json(self) -> str:
        row = asdict(self)
        row["usage"] = self.usage.to_dict() if self.usage else None
        return json.dumps(row)

    @classmethod
    def from_json(cls, line: str) -> UsageEvent:
        row = json.loads(line)
        usage = row.pop("usage")
        return cls(**row, usage=UsageRecord.from_dict(usage) if usage else None)


class Ledger:
    """Append-only JSONL, safe to share across the server's threads."""

    def __init__(self, path: Path | str = DEFAULT_LEDGER) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def record(self, event: UsageEvent) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.to_json() + "\n")

    def read(self) -> list[UsageEvent]:
        if not self.path.is_file():
            return []
        return [
            UsageEvent.from_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


@dataclass(frozen=True, slots=True)
class Summary:
    high_model: str
    requests: int
    low: int
    high: int
    untouched: int
    no_usage: int
    switches: int
    actual_usd: Decimal          # routed requests, as served
    all_high_usd: Decimal        # routed requests, had each gone to the model it asked for
    untouched_usd: Decimal
    unpriced: set[str] = field(default_factory=set)
    sessions: dict[str, dict[str, int]] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def saved_usd(self) -> Decimal:
        return self.all_high_usd - self.actual_usd

    @property
    def saved_share(self) -> float:
        return float(self.saved_usd / self.all_high_usd) if self.all_high_usd else 0.0


def summarise(events: list[UsageEvent], book: PriceBook) -> Summary:
    counts: dict[str, int] = defaultdict(int)
    sessions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tokens: dict[str, int] = defaultdict(int)
    actual = all_high = untouched = Decimal(0)
    unpriced: set[str] = set()
    no_usage = 0
    one_model = _OneModelCache(events)

    for event in events:
        counts[event.tier] += 1
        sessions[event.session or "(none)"][event.tier] += 1
        if event.usage is None:
            no_usage += 1
            continue
        tokens[event.tier] += event.usage.total_tokens
        cost = _cost(book, event.usage, event.routed_model, unpriced)
        if event.tier == "untouched":
            untouched += cost or 0
            continue
        high_cost = _cost(book, one_model.usage(event), event.requested_model, unpriced)
        if cost is not None and high_cost is not None:
            actual += cost
            all_high += high_cost

    requested = sorted({e.requested_model for e in events if e.tier != "untouched"})
    return Summary(
        high_model=", ".join(requested) or "-",
        requests=len(events),
        low=counts["low"],
        high=counts["high"],
        untouched=counts["untouched"],
        no_usage=no_usage,
        switches=_switches(events),
        actual_usd=actual,
        all_high_usd=all_high,
        untouched_usd=untouched,
        unpriced=unpriced,
        sessions={k: dict(v) for k, v in sessions.items()},
        tokens=dict(tokens),
    )


class _OneModelCache:
    """Replays routed requests as if one model's cache had seen them all.

    `usage(event)` must be called in ledger order, once per routed event with usage.
    """

    def __init__(self, events: list[UsageEvent]) -> None:
        self._shared_prefix = _shared_prefix(events)
        self._last_in_session: dict[str, UsageEvent] = {}
        self._last_on_model: dict[str, str] = {}   # model -> ts
        self._last_ts: str | None = None

    def usage(self, event: UsageEvent) -> UsageRecord:
        usage = event.usage
        assert usage is not None
        warm = max(self._conversation_so_far(event), self._shared_opening(event))
        self._observe(event)
        return _as_cache_reads(usage, warm)

    def _conversation_so_far(self, event: UsageEvent) -> int:
        previous = self._last_in_session.get(event.session) if event.session else None
        if (previous is None or previous.routed_model == event.routed_model
                or not _within_cache_lifetime(previous.ts, event.ts)):
            return 0
        return previous.usage.prompt_tokens  # type: ignore[union-attr]

    def _shared_opening(self, event: UsageEvent) -> int:
        model_was_used = _within_cache_lifetime(self._last_on_model.get(event.routed_model),
                                                event.ts)
        anything_was_used = _within_cache_lifetime(self._last_ts, event.ts)
        return self._shared_prefix if anything_was_used and not model_was_used else 0

    def _observe(self, event: UsageEvent) -> None:
        if event.session:
            self._last_in_session[event.session] = event
        self._last_on_model[event.routed_model] = event.ts
        self._last_ts = event.ts


def _shared_prefix(events: list[UsageEvent]) -> int:
    """The largest cache read any session opened with: proof that at least that
    many tokens are the same in every session."""
    opened: set[str] = set()
    largest = 0
    for event in events:
        if event.tier == "untouched" or event.usage is None or not event.session:
            continue
        if event.session not in opened:
            opened.add(event.session)
            largest = max(largest, event.usage.cache_read_input_tokens)
    return largest


def _as_cache_reads(usage: UsageRecord, warm: int) -> UsageRecord:
    """Bill the first `warm` prompt tokens as cache reads; the rest as they were."""
    extra = max(0, warm - usage.cache_read_input_tokens)
    from_write = min(usage.cache_creation_input_tokens, extra)
    from_input = min(usage.input_tokens, extra - from_write)
    if not from_write and not from_input:
        return usage
    return replace(
        usage,
        cache_creation_input_tokens=usage.cache_creation_input_tokens - from_write,
        input_tokens=usage.input_tokens - from_input,
        cache_read_input_tokens=usage.cache_read_input_tokens + from_write + from_input,
    )


def _within_cache_lifetime(earlier: str | None, later: str) -> bool:
    if earlier is None:
        return False
    try:
        gap = datetime.fromisoformat(later) - datetime.fromisoformat(earlier)
    except (ValueError, TypeError):
        return True  # unreadable timestamps: assume warm, which understates savings
    return gap <= CACHE_LIFETIME


def _cost(book: PriceBook, usage: UsageRecord, model: str, unpriced: set[str]) -> Decimal | None:
    try:
        return book.cost_usd(usage, model)
    except KeyError:
        unpriced.add(model)
        return None


def _switches(events: list[UsageEvent]) -> int:
    """Times a session's routed model changed between consecutive routed requests."""
    last: dict[str, str] = {}
    switches = 0
    for event in events:
        if event.tier == "untouched":
            continue
        key = event.session or "(none)"
        if key in last and last[key] != event.routed_model:
            switches += 1
        last[key] = event.routed_model
    return switches


def render(s: Summary) -> str:
    routed = s.low + s.high
    share = lambda n: f"{n / routed:.0%}" if routed else "-"
    lines = [
        f"requests        {s.requests}",
        f"  routed low    {s.low:>5}  ({share(s.low)})   tokens {s.tokens.get('low', 0):,}",
        f"  routed high   {s.high:>5}  ({share(s.high)})   tokens {s.tokens.get('high', 0):,}",
        (f"  untouched     {s.untouched:>5}  (other models the client asked for, "
         f"${s.untouched_usd:.4f})"),
        f"  no usage      {s.no_usage:>5}  (errors or cut-off streams; not counted as free)",
        (f"model switches  {s.switches}  (each re-sends the conversation to a cold cache; "
         "charged to routing)"),
        "",
        "cost, routed requests (API-equivalent):",
        f"  actual        ${s.actual_usd:.4f}",
        (f"  all-high      ${s.all_high_usd:.4f}   (same tokens on {s.high_model}, "
         "no switches)"),
        f"  saved         ${s.saved_usd:.4f}   ({s.saved_share:.1%})",
    ]
    if s.unpriced:
        lines.append(f"unpriced models (cost not estimated): {', '.join(sorted(s.unpriced))}")
    if len(s.sessions) > 1:
        lines.append("")
        lines.append("per session:")
        for name, tiers in s.sessions.items():
            detail = "  ".join(f"{t} {n}" for t, n in sorted(tiers.items()))
            lines.append(f"  {name[:36]:<36}  {detail}")
    return "\n".join(lines)


def _as_dict(s: Summary) -> dict[str, Any]:
    return {
        "requests": s.requests, "low": s.low, "high": s.high, "untouched": s.untouched,
        "no_usage": s.no_usage, "switches": s.switches,
        "actual_usd": str(s.actual_usd), "all_high_usd": str(s.all_high_usd),
        "saved_usd": str(s.saved_usd), "saved_share": round(s.saved_share, 4),
        "untouched_usd": str(s.untouched_usd), "unpriced": sorted(s.unpriced),
        "sessions": s.sessions,
    }
