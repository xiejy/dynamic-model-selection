"""Read token usage out of a response while it streams through, unmodified.

The pass-through proxy never alters a response. It feeds each chunk it forwards
to a sniffer, which picks the usage out of the event stream (or the JSON body
for a non-streamed response):

* Anthropic: `message_start` carries the input side; `message_delta` carries
  the final, cumulative output count. Summing deltas would double count. A
  stream cut off before `message_delta` has no real output count (the start
  event's is a placeholder 1), so it reports no usage rather than a wrong one.
* OpenAI Responses: the terminal event -- `response.completed`, or
  `response.incomplete` / `response.failed`, which are billed too -- carries
  everything. Its `input_tokens` INCLUDES the cached portion, so the cached
  count is subtracted or those tokens would be billed twice.
"""
from __future__ import annotations

import json
from typing import Any

from dms.usage import UsageRecord

JSON_BODY_LIMIT = 16 * 1024 * 1024  # a non-streamed body larger than this is not parsed
_OPENAI_TERMINAL = {"response.completed", "response.incomplete", "response.failed"}


class UsageSniffer:
    """Incremental: call `feed()` with each forwarded chunk, then `result()`."""

    def __init__(self, family: str) -> None:
        if family not in {"anthropic", "openai"}:
            raise ValueError(f"family must be 'anthropic' or 'openai', got {family!r}")
        self.family = family
        self._mode: str | None = None  # "sse" or "json", decided by the first byte
        self._line = bytearray()
        self._body = bytearray()
        self._usage: dict[str, int] = {}
        self._final = False  # anthropic: the cumulative output count has arrived

    @property
    def buffered_bytes(self) -> int:
        return len(self._line) + len(self._body)

    def feed(self, chunk: bytes) -> None:
        if self._mode is None:
            head = chunk.lstrip()
            if not head:
                return
            self._mode = "json" if head[:1] in (b"{", b"[") else "sse"
        if self._mode == "json":
            if len(self._body) < JSON_BODY_LIMIT:
                self._body.extend(chunk)
            return
        self._line.extend(chunk)
        while (end := self._line.find(b"\n")) != -1:
            line = bytes(self._line[:end]).strip()
            del self._line[: end + 1]
            if line.startswith(b"data:"):
                self._event(_load(line[5:].strip()))

    def result(self) -> UsageRecord | None:
        if self._mode == "json":
            self._event(_load(bytes(self._body)))
        elif self._line.strip().startswith(b"data:"):
            self._event(_load(bytes(self._line).strip()[5:].strip()))
            self._line.clear()
        if not self._usage or (self.family == "anthropic" and not self._final):
            return None
        return UsageRecord.from_dict(self._usage)

    # ------------------------------------------------------------------ events

    def _event(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        if self.family == "anthropic":
            self._anthropic(data)
        else:
            self._openai(data)

    def _anthropic(self, data: dict[str, Any]) -> None:
        if data.get("type") == "message_start":
            self._merge((data.get("message") or {}).get("usage"))
        elif data.get("type") in ("message_delta", "message"):  # "message": non-streamed
            self._merge(data.get("usage"))
            self._final = self._final or isinstance(data.get("usage"), dict)

    def _openai(self, data: dict[str, Any]) -> None:
        response = data.get("response") if data.get("type") in _OPENAI_TERMINAL else data
        usage = (response or {}).get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict) or "input_tokens" not in usage:
            return
        cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        self._usage = {
            "input_tokens": max(0, int(usage.get("input_tokens") or 0) - cached),
            "cache_read_input_tokens": cached,
            "output_tokens": int(usage.get("output_tokens") or 0),
        }

    def _merge(self, usage: Any) -> None:
        """Later values replace earlier ones: message_delta's output count is
        cumulative, so the last one seen is the total."""
        if not isinstance(usage, dict):
            return
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            if isinstance(usage.get(key), int):
                self._usage[key] = usage[key]


def _load(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
