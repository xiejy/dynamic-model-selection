"""Compare two identical bench runs. The gap between them IS the noise floor.

Same workload, same strategies, same models, same code -- only the model's own
nondeterminism differs between them. Whatever moves is what no conclusion at
this sample size may rest on.

    uv run python scripts/compare_runs.py out/twotier-run1.json out/twotier-run2.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main(a_path: str, b_path: str) -> int:
    a, b = load(a_path), load(b_path)
    ax = {s["name"]: s for s in a["strategies"]}
    bx = {s["name"]: s for s in b["strategies"]}
    shared = [n for n in ax if n in bx]

    print(f"run 1: {a['calls_made']} calls, ${float(a['total_spend_usd']):.4f}")
    print(f"run 2: {b['calls_made']} calls, ${float(b['total_spend_usd']):.4f}\n")

    print(f"{'strategy':<16}{'acc 1':>8}{'acc 2':>8}{'swing':>9}"
          f"{'cost 1':>10}{'cost 2':>10}{'swing':>9}")
    print("-" * 70)
    swings = []
    for name in shared:
        a1, a2 = ax[name]["accuracy"], bx[name]["accuracy"]
        c1, c2 = float(ax[name]["cost_usd"]), float(bx[name]["cost_usd"])
        swings.append(abs(a2 - a1))
        print(f"{name:<16}{a1:>7.1%}{a2:>8.1%}{a2 - a1:>+9.1%}"
              f"{c1:>10.4f}{c2:>10.4f}{(c2 - c1) / c1:>+8.1%}")

    print(f"\n  largest accuracy swing between identical runs : {max(swings):+.1%}")
    print(f"  mean absolute swing                           : {sum(swings)/len(swings):.1%}")

    def order(x: dict, key) -> list[str]:
        return [n for n, _ in sorted(x.items(), key=key)]

    by_acc = lambda kv: -kv[1]["accuracy"]  # noqa: E731
    by_cost = lambda kv: float(kv[1]["cost_usd"])  # noqa: E731

    r1, r2 = order(ax, by_acc), order(bx, by_acc)
    print(f"\n  accuracy ranking run 1: {' > '.join(r1[:4])}")
    print(f"  accuracy ranking run 2: {' > '.join(r2[:4])}")
    print(f"  accuracy ranking stable? {'YES' if r1 == r2 else 'NO -- the order changed'}")

    k1, k2 = order(ax, by_cost), order(bx, by_cost)
    print(f"  cost ranking stable?     {'YES' if k1 == k2 else 'NO'}")

    print("\n  per-task verdict flips (same strategy, same task, different answer):")
    total = 0
    for name in sorted(set(a.get("per_task", {})) & set(b.get("per_task", {}))):
        pa, pb = a["per_task"][name], b["per_task"][name]
        flips = sorted(t for t in pa if t in pb and pa[t]["correct"] != pb[t]["correct"])
        total += len(flips)
        if flips:
            print(f"    {name:<16} {len(flips):>2}: {', '.join(flips)}")
    print(f"    {'TOTAL':<16} {total:>2} flips across identical reruns")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:3] or ["out/twotier-run1.json", "out/twotier-run2.json"]
    raise SystemExit(main(*args))
