"""Baselines. The bench is meaningless without them.

`AlwaysRouter` gives the two anchors: always-Opus is the quality ceiling and cost
ceiling, always-Haiku the cost floor.

`RandomRouter` is the one people skip, and it is the one that matters. RouteLLM's
own evaluation compares against random routing **at the same strong-model call
fraction**, because any router that sends 40% of traffic to the expensive model
will beat always-Haiku on quality and always-Opus on cost -- that comparison
proves nothing. The question is whether the router beats a coin weighted to the
same rate. Several published routers barely do.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from dms.client import ModelClient
from dms.routers.base import Decision, Router
from dms.workload import Task


@dataclass(frozen=True, slots=True)
class AlwaysRouter(Router):
    """Send everything to one model. Zero routing cost, zero cleverness."""

    model: str
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", f"always:{_short(self.model)}")

    def route(self, task: Task, client: ModelClient) -> Decision:
        return Decision(model=self.model, why="fixed model")


@dataclass(frozen=True, slots=True)
class RandomRouter(Router):
    """Route to the strong model with fixed probability, deterministically.

    Seeded off the task id so a given call fraction always produces the same
    assignment -- otherwise the baseline moves under you between runs and the
    comparison is noise.
    """

    strong_model: str
    weak_model: str
    strong_fraction: float
    name: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.strong_fraction <= 1.0:
            raise ValueError("strong_fraction must be within [0, 1]")
        if not self.name:
            object.__setattr__(
                self, "name", f"random@{self.strong_fraction:.0%}-strong"
            )

    def route(self, task: Task, client: ModelClient) -> Decision:
        # Uniform in [0,1) from a stable hash of the task id.
        digest = hashlib.sha256(task.id.encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64

        if draw < self.strong_fraction:
            return Decision(model=self.strong_model, why="random draw -> strong")
        return Decision(model=self.weak_model, why="random draw -> weak")


def _short(model: str) -> str:
    """claude-haiku-4-5 -> haiku"""
    parts = model.split("-")
    return parts[1] if len(parts) > 1 else model
