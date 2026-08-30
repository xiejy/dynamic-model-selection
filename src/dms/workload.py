"""The benchmark workload: mixed-difficulty developer tasks with checkable answers.

Difficulty spread is the whole point. If every task were easy, routing
everything to Haiku would look free; if every task were hard, routing would look
useless. Real traffic is mixed, and the mix ratio is the single biggest driver of
how much routing can possibly save -- more than the router algorithm.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

DEFAULT_WORKLOAD = Path(__file__).resolve().parents[2] / "tasks" / "workload.jsonl"
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True, slots=True)
class Task:
    """One benchmark item."""

    id: str
    difficulty: str
    kind: str
    prompt: str
    expected: str
    grader: str

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Self:
        missing = {"id", "difficulty", "kind", "prompt", "expected", "grader"} - data.keys()
        if missing:
            raise ValueError(f"task is missing fields: {', '.join(sorted(missing))}")
        if data["difficulty"] not in DIFFICULTIES:
            raise ValueError(
                f"task {data['id']}: difficulty {data['difficulty']!r} "
                f"must be one of {DIFFICULTIES}"
            )
        return cls(
            id=data["id"],
            difficulty=data["difficulty"],
            kind=data["kind"],
            prompt=data["prompt"],
            expected=data["expected"],
            grader=data["grader"],
        )


@dataclass(frozen=True, slots=True)
class Workload:
    """An immutable, ordered set of tasks."""

    tasks: tuple[Task, ...]

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        source = path or DEFAULT_WORKLOAD
        tasks = tuple(
            Task.from_dict(json.loads(line))
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not tasks:
            raise ValueError(f"{source} contains no tasks")
        _reject_duplicate_ids(tasks)
        return cls(tasks=tasks)

    def by_difficulty(self, difficulty: str) -> tuple[Task, ...]:
        return tuple(task for task in self.tasks if task.difficulty == difficulty)

    def mix(self) -> dict[str, int]:
        return {level: len(self.by_difficulty(level)) for level in DIFFICULTIES}

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)


def _reject_duplicate_ids(tasks: tuple[Task, ...]) -> None:
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise ValueError(f"duplicate task id: {task.id}")
        seen.add(task.id)
