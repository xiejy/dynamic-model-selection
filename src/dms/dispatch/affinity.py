"""Session affinity: pin a session's model so the provider prompt cache survives.

This exists because of a measurement in this repo, not because it is a common
gateway feature. A model switch invalidates the Anthropic prompt cache
completely -- tools, system and messages tiers -- and unlike tool edits or
system-prompt edits it has no escape hatch, because caches are model-scoped.
Cache reads cost 0.10x input and writes cost 1.25x, so a reroute forfeits the
discount AND re-pays the write premium.

Consequence: on a multi-turn session, re-deciding the model every turn can cost
more than the tier difference it saves. The first decision is therefore sticky
for a TTL. LiteLLM ships the same mechanism for the same reason.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class Pin:
    model: str
    reason: str
    expires_at: float

    def alive(self, now: float) -> bool:
        return now < self.expires_at


class SessionAffinity:
    """Thread-safe TTL map from session id to a pinned model.

    In-process only. A multi-instance deployment needs a shared store (Redis or
    similar) or sticky load balancing, otherwise each instance pins
    independently and the cache benefit is diluted by the number of replicas.
    """

    def __init__(self, ttl_seconds: int = 3600, clock=time.monotonic) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._pins: dict[str, Pin] = {}
        self._lock = Lock()

    def get(self, session_id: str | None) -> Pin | None:
        if not session_id or self.ttl_seconds == 0:
            return None
        now = self._clock()
        with self._lock:
            pin = self._pins.get(session_id)
            if pin is None:
                return None
            if not pin.alive(now):
                del self._pins[session_id]
                return None
            return pin

    def set(self, session_id: str | None, model: str, reason: str) -> None:
        if not session_id or self.ttl_seconds == 0:
            return
        with self._lock:
            self._pins[session_id] = Pin(
                model=model,
                reason=reason,
                expires_at=self._clock() + self.ttl_seconds,
            )

    def release(self, session_id: str) -> None:
        with self._lock:
            self._pins.pop(session_id, None)

    def purge_expired(self) -> int:
        """Drop dead pins. Call periodically; the map is otherwise unbounded."""
        now = self._clock()
        with self._lock:
            dead = [key for key, pin in self._pins.items() if not pin.alive(now)]
            for key in dead:
                del self._pins[key]
        return len(dead)

    def __len__(self) -> int:
        return len(self._pins)
