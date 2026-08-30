"""Shared helpers for the numbered examples.

Every example runs offline by default. Set DMS_MODE=record with a real
ANTHROPIC_API_KEY to measure instead of estimate.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dms.client import Mode, ModelClient  # noqa: E402
from dms.pricing import PriceBook  # noqa: E402
from dms.replay import FixtureStore  # noqa: E402

MODE = Mode(os.environ.get("DMS_MODE", "simulate"))
ROOT = Path(__file__).resolve().parents[1]


def client(max_spend_usd: str | None = "2.00") -> ModelClient:
    return ModelClient(
        mode=MODE,
        book=PriceBook.load(),
        store=FixtureStore(ROOT / "fixtures"),
        max_spend_usd=max_spend_usd,
    )


def banner(title: str, *, exact: bool = False) -> None:
    """`exact=True` marks an example that is pure arithmetic over published
    rates -- no model is called, so the numbers hold regardless of mode."""
    if exact:
        print(f"\n=== {title}  [EXACT: arithmetic over published rates] ===\n")
        return
    tag = "ESTIMATED (offline)" if MODE is Mode.SIMULATE else MODE.value.upper()
    print(f"\n=== {title}  [{tag}] ===")
    if MODE is Mode.SIMULATE:
        print("    numbers below are modelled, not measured -- DMS_MODE=record to measure\n")


def usd(value) -> str:
    return f"${value:.6f}"
