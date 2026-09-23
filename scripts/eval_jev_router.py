"""Evaluate Jev as the model selector, against the pre-registered bar.

    uv run python scripts/eval_jev_router.py          # plan: reads no key, sends nothing
    source ~/.config/typesafe/env && \\
    uv run python scripts/eval_jev_router.py --live   # 36 requests to api.typesafe.ai

Only Jev is called. Claude outcomes are REPLAYED from the three recorded runs: a
LOW verdict takes Haiku's recorded correctness, HIGH takes Opus's. Responses are
cached on (model + request), so every number re-derives for free.

Decision rule and pass bar: docs/jev-router-prereg.md (written before any Jev
response existed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dms.client import Mode, ModelClient  # noqa: E402
from dms.pricing import PriceBook  # noqa: E402
from dms.replay import FixtureStore  # noqa: E402
from dms.routers.base import ANSWER_EFFORT, ANSWER_MAX_TOKENS, ANSWER_SYSTEM  # noqa: E402
from dms.routers.heuristic import HeuristicRouter  # noqa: E402
from dms.routers.jev import HIGH_LEVEL, JEV_MODEL, JevRouter, default_asker, nearest_level  # noqa: E402
from dms.twotier import HIGH_MODEL, LOW_MODEL, two_tier_map  # noqa: E402
from dms.workload import Workload  # noqa: E402

CACHE = ROOT / "out" / "jev_cache.jsonl"
LABELS = ROOT / "out" / "routing_labels.json"
RUNS = [ROOT / "out" / f"twotier-run{i}.json" for i in (1, 2, 3)]
EST_CHARS_PER_TOKEN = 3.6
EST_OVERHEAD_TOKENS = 200

BAR_QUALITY_SLACK = 1 / 36   # one task
BAR_MAX_HIGH_SHARE = 0.30


# ---------------------------------------------------------------------- cache


def cache_key(state: str, questions: dict, model: str) -> str:
    body = json.dumps({"model": model, "state": state, "questions": questions}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def load_cache() -> dict[str, dict]:
    if not CACHE.is_file():
        return {}
    rows = (json.loads(line) for line in CACHE.read_text().splitlines() if line.strip())
    return {row["key"]: row["response"] for row in rows}


def save_to_cache(key: str, response: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a") as handle:
        handle.write(json.dumps({"key": key, "response": response}) + "\n")


# ----------------------------------------------------------------------- plan


def plan(tasks, router: JevRouter) -> None:
    questions = router.questions()
    q_chars = len(json.dumps(questions))
    est = [EST_OVERHEAD_TOKENS + (q_chars + len(t.prompt)) / EST_CHARS_PER_TOKEN for t in tasks]
    tokens = sum(est)
    cost = Decimal(str(tokens)) * Decimal("0.042") / Decimal(1_000_000)

    print("PLAN — no key read, nothing sent\n")
    print(f"  destination : POST https://api.typesafe.ai/v1/systemone (TypeSafe AI)")
    print(f"  requests    : {len(tasks)}, one per benchmark task, model {JEV_MODEL}")
    print(f"  state       : the task prompt text ONLY -- no expected answer, no label")
    print(f"  questions   : 1 Score (decides) + 2 Nouls (diagnostic), identical every request")
    print(f"  data origin : tasks/workload.jsonl -- already public in this GitHub repo")
    print(f"  est. tokens : ~{tokens:,.0f} input (output tokens are free)")
    print(f"  est. cost   : ~${cost:.4f}   (a 2-item smoke test will refit this)")
    print(f"\n  example request (task {tasks[0].id}):")
    example = {"model": JEV_MODEL, "state": tasks[0].prompt, "questions": questions}
    print("   ", json.dumps(example)[:420] + " ...")


# ----------------------------------------------------------------------- live


def run_live(tasks, router: JevRouter, limit: int | None) -> dict[str, dict]:
    cache = load_cache()
    questions = router.questions()
    results: dict[str, dict] = {}
    latencies: list[float] = []
    for task in tasks[:limit] if limit else tasks:
        key = cache_key(task.prompt, questions, JEV_MODEL)
        if key in cache:
            results[task.id] = cache[key]
            continue
        started = time.perf_counter()
        response = default_asker(task.prompt, questions, JEV_MODEL)
        latencies.append((time.perf_counter() - started) * 1000)
        if response.get("model") != JEV_MODEL:
            print(f"  WARNING: asked for {JEV_MODEL}, served {response.get('model')}")
        save_to_cache(key, response)
        results[task.id] = response
    if latencies:
        print(f"  live calls: {len(latencies)}, latency median "
              f"{statistics.median(latencies):.0f} ms, max {max(latencies):.0f} ms")
    return results


# ------------------------------------------------------------------- evaluate


def auc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def per_task_cost(book: PriceBook, tasks) -> dict[str, dict[str, Decimal]]:
    client = ModelClient(mode=Mode.REPLAY, book=book, store=FixtureStore(ROOT / "fixtures"))
    costs: dict[str, dict[str, Decimal]] = {}
    for task in tasks:
        costs[task.id] = {}
        for model in (LOW_MODEL, HIGH_MODEL):
            key = client.fixture_key(model=model, prompt=task.prompt, system=ANSWER_SYSTEM,
                                     max_tokens=ANSWER_MAX_TOKENS, effort=ANSWER_EFFORT)
            call = client.store.get(key)
            costs[task.id][model] = book.cost_usd(call.usage, model) if call else Decimal(0)
    return costs


def replay(decisions: dict[str, bool], runs, costs, router_cost: Decimal) -> dict:
    """Accuracy and cost if these high/low decisions had been made."""
    accuracy = statistics.mean(
        statistics.mean(
            run["per_task"]["always:opus" if decisions[t] else "always:haiku"][t]["correct"]
            for t in decisions
        )
        for run in runs
    )
    claude = sum(costs[t][HIGH_MODEL if decisions[t] else LOW_MODEL] for t in decisions)
    return {
        "accuracy": accuracy,
        "high_share": sum(decisions.values()) / len(decisions),
        "cost": claude + router_cost,
    }


def evaluate(tasks, responses: dict[str, dict]) -> int:
    book = PriceBook.load()
    runs = [json.loads(p.read_text()) for p in RUNS]
    labels = json.loads(LABELS.read_text())
    costs = per_task_cost(book, tasks)
    need = {t: v["haiku"] <= 1 for t, v in labels.items()}
    routable = {t for t, v in labels.items() if v["haiku"] <= 1 and v["opus"] >= 2}

    score = {t: r["answers"]["route"]["score"] for t, r in responses.items()}
    conf = {t: r["answers"]["route"].get("confidence", 0.0) for t, r in responses.items()}
    jev_high = {t: nearest_level(s, 4) >= HIGH_LEVEL for t, s in score.items()}
    jev_tokens = sum(r["usage"]["input_tokens"] for r in responses.values())
    jev_cost = book.cost_usd(type("U", (), {"input_tokens": jev_tokens, "output_tokens": 0,
                                           "cache_creation_input_tokens": 0,
                                           "cache_read_input_tokens": 0})(), JEV_MODEL)

    heuristic = HeuristicRouter(tiers=two_tier_map())
    client = ModelClient(mode=Mode.SIMULATE, book=book, store=FixtureStore("/tmp/unused"))
    heur_high = {t.id: heuristic.route(t, client).model == HIGH_MODEL for t in tasks}

    rows = {
        "always: haiku": replay({t: False for t in score}, runs, costs, Decimal(0)),
        "heuristic": replay(heur_high, runs, costs, Decimal(0)),
        "jev": replay(jev_high, runs, costs, jev_cost),
        "always: opus": replay({t: True for t in score}, runs, costs, Decimal(0)),
    }

    print_bar()
    print("\nREPLAY on recorded outcomes (same method for every row)\n")
    print(f"  {'router':<14}{'accuracy':>10}{'% high':>9}{'cost':>12}")
    for name, r in rows.items():
        print(f"  {name:<14}{r['accuracy']:>9.1%}{r['high_share']:>9.0%}{r['cost']:>12.6f}")
    print(f"\n  jev router spend: {jev_tokens:,} input tokens = ${jev_cost:.6f} "
          f"for {len(responses)} decisions")

    return verdict(rows, score, conf, need, routable, jev_high)


def print_bar() -> None:
    print("\nPRE-REGISTERED BAR (docs/jev-router-prereg.md) — Jev passes only if ALL hold:")
    print(f"  1. accuracy  >= heuristic - {BAR_QUALITY_SLACK:.1%} (one task)")
    print(f"  2. sends     <= {BAR_MAX_HIGH_SHARE:.0%} of tasks high")
    print(f"  3. total cost < heuristic's")


def verdict(rows, score, conf, need, routable, jev_high) -> int:
    jev, heur = rows["jev"], rows["heuristic"]
    checks = [
        ("quality", jev["accuracy"] >= heur["accuracy"] - BAR_QUALITY_SLACK),
        ("selectivity", jev["high_share"] <= BAR_MAX_HIGH_SHARE),
        ("cost", jev["cost"] < heur["cost"]),
    ]
    ids = sorted(score)
    ranking = auc([score[t] for t in ids], [need[t] for t in ids])
    caught = [t for t in sorted(routable) if jev_high[t]]
    agree = [conf[t] for t in ids if jev_high[t] == need[t]]
    disagree = [conf[t] for t in ids if jev_high[t] != need[t]]

    print("\nVERDICT")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  => Jev {'PASSES' if all(ok for _, ok in checks) else 'DOES NOT PASS'} the bar")
    print("\nDIAGNOSTICS (reported, never the verdict)")
    print(f"  AUC vs NEEDS HIGH      : {ranking:.3f}   (6 positives -- wide interval)")
    print(f"  recall, routable (5)   : {len(caught)}/{len(routable)}  {caught}")
    print(f"  mean confidence        : {statistics.mean(agree) if agree else 0:.2f} where right, "
          f"{statistics.mean(disagree) if disagree else 0:.2f} where wrong")
    print("\n  per task (score, level, decision, label):")
    for t in ids:
        mark = "" if jev_high[t] == need[t] else "   <-- disagrees with label"
        print(f"    {t}  {score[t]:.2f}  L{nearest_level(score[t], 4)}  "
              f"{'HIGH' if jev_high[t] else 'low '}  need={'HIGH' if need[t] else 'low '}{mark}")
    return 0 if all(ok for _, ok in checks) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--live", action="store_true", help="send requests (needs TYPESAFE_API_KEY)")
    parser.add_argument("--limit", type=int, default=None, help="smoke-test the first N tasks")
    args = parser.parse_args()

    tasks = list(Workload.load())
    router = JevRouter(tiers=two_tier_map())
    if not args.live:
        plan(tasks, router)
        cached = load_cache()
        hit = [t for t in tasks if cache_key(t.prompt, router.questions(), JEV_MODEL) in cached]
        if len(hit) == len(tasks):
            print("\n  every task is cached -- evaluating from cache, nothing sent:")
            return evaluate(tasks, {t.id: cached[cache_key(t.prompt, router.questions(),
                                                           JEV_MODEL)] for t in tasks})
        print(f"\n  {len(hit)}/{len(tasks)} cached. Re-run with --live to send the rest.")
        return 0

    responses = run_live(tasks, router, args.limit)
    if len(responses) < len(tasks):
        for tid, r in responses.items():
            print(f"  {tid}: score {r['answers']['route']['score']:.2f}, "
                  f"{r['usage']['input_tokens']} input tokens")
        return 0
    return evaluate(tasks, responses)


if __name__ == "__main__":
    raise SystemExit(main())
