"""Content-addressed fixture store.

Every live call is recorded to `fixtures/<hash>.json`, keyed on the request
content. The talk demo then replays offline: no key, no network, no spend, and
identical numbers every run. If the conference wifi dies, the demo still works.

The key must be content-addressed rather than order-addressed -- `json.dumps`
without `sort_keys` produces a different string for the same dict on a different
run, which would miss every fixture. This is the same class of bug that silently
destroys prompt-cache hit rates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dms.client import Call


class FixtureStore:
    """Reads and writes recorded calls under a directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def key(
        self,
        *,
        model: str,
        system: str | None,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        payload = json.dumps(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "options": options or {},
            },
            sort_keys=True,  # order-independence: see module docstring
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Call | None:
        from dms.client import Call

        path = self.path_for(key)
        if not path.is_file():
            return None
        return Call.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def put(self, key: str, call: Call) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path_for(key).write_text(
            json.dumps(call.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def count(self) -> int:
        return len(list(self.root.glob("*.json"))) if self.root.is_dir() else 0
