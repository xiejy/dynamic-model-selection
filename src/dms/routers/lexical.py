"""Nearest-exemplar routing over TF-IDF vectors. Local, stdlib, ~0.1 ms.

This stands in for the embedding-similarity family (Aurelio's semantic-router,
LiteLLM's semantic keyword matching). It is deliberately *not* a neural embedder:
LLMRouterBench's ablation found a 22.7M-parameter embedder performed comparably
to much larger ones, which is the same message one step further down -- most of
the signal is lexical. Swapping in real embeddings is a one-function change and
would let the talk show whether the extra dependency buys anything.

Two details that matter and are easy to get wrong:

* **MAX aggregation, not MEAN.** Similarity to a tier is the best-matching
  exemplar, not the average. Averaging dilutes one strong match against seven
  unrelated ones and collapses the tiers together.
* **Exemplars are a separate file from the workload.** Fitting the router on the
  tasks it is scored against is train-on-test, and it is exactly how routing
  demos manufacture numbers that do not survive contact with real traffic.
  FrugalGPT's own limitations section says the quiet part: training examples must
  come from the same distribution as the test set.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from dms.client import ModelClient
from dms.routers.base import TIER_MODELS, Decision, Router
from dms.workload import Task

DEFAULT_EXEMPLARS = Path(__file__).resolve().parents[3] / "tasks" / "exemplars.jsonl"
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def tokenise(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True, slots=True)
class Exemplar:
    tier: str
    text: str
    vector: dict[str, float]


class LexicalRouter(Router):
    """Route to the tier whose closest exemplar is most similar to the prompt."""

    name = "lexical"

    def __init__(
        self,
        exemplars_path: Path | None = None,
        tiers: dict[str, str] | None = None,
        min_similarity: float = 0.05,
    ) -> None:
        self.tiers = dict(tiers or TIER_MODELS)
        self.min_similarity = min_similarity
        self._idf: dict[str, float] = {}
        self.exemplars: tuple[Exemplar, ...] = ()
        self._fit(exemplars_path or DEFAULT_EXEMPLARS)

    # -------------------------------------------------------------------- fitting

    def _fit(self, path: Path) -> None:
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"{path} contains no exemplars")

        documents = [tokenise(row["text"]) for row in rows]
        total = len(documents)
        appearances = Counter(term for doc in documents for term in set(doc))
        # Smoothed IDF; +1 keeps a term that appears everywhere at a small but
        # non-zero weight instead of annihilating it.
        self._idf = {
            term: math.log(total / (1 + count)) + 1.0
            for term, count in appearances.items()
        }
        self.exemplars = tuple(
            Exemplar(tier=row["tier"], text=row["text"], vector=self._vectorise(doc))
            for row, doc in zip(rows, documents, strict=True)
        )

    def _vectorise(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        weights = {
            # Unseen terms get idf 0 -- they carry no discriminative signal.
            term: (count / len(tokens)) * self._idf.get(term, 0.0)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        return {term: weight / norm for term, weight in weights.items()} if norm else {}

    # -------------------------------------------------------------------- routing

    def similarities(self, prompt: str) -> dict[str, float]:
        """Best-matching exemplar per tier (MAX aggregation)."""
        query = self._vectorise(tokenise(prompt))
        scores = {tier: 0.0 for tier in self.tiers}
        for exemplar in self.exemplars:
            score = _cosine(query, exemplar.vector)
            if score > scores.get(exemplar.tier, 0.0):
                scores[exemplar.tier] = score
        return scores

    def route(self, task: Task, client: ModelClient) -> Decision:
        scores = self.similarities(task.prompt)
        tier, best = max(scores.items(), key=lambda item: item[1])

        # No exemplar resembles this prompt: refusing to guess and falling back
        # to the capable model is the safe failure. A router that silently
        # guesses on out-of-distribution input is how quality regressions ship.
        if best < self.min_similarity:
            return Decision(
                model=self.tiers["complex"],
                why=f"no confident match (best={best:.3f}) -> fallback to complex",
            )

        runner_up = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
        return Decision(
            model=self.tiers[tier],
            why=f"{tier} sim={best:.3f} (margin={best - runner_up:+.3f})",
            spends=(),  # local vectors only: no tokens spent
        )

    @classmethod
    def from_path(cls, path: Path) -> Self:
        return cls(exemplars_path=path)


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Both vectors are pre-normalised, so the dot product is the cosine."""
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())
