"""Command line entry point: `dms bench`, `dms levers`, `dms tasks`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dms.bench import run_bench, two_tier_strategies
from dms.client import Mode, ModelClient, MissingCredential, SpendLimitExceeded
from dms.pricing import PriceBook
from dms.replay import FixtureStore
from dms.report import dumps, render, render_levers
from dms.workload import Workload

DEFAULT_OUT = Path("out")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dms",
        description="Measure what LLM cost levers actually save.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="run every strategy over the workload")
    bench.add_argument(
        "--mode",
        choices=[mode.value for mode in Mode],
        default=Mode.SIMULATE.value,
        help="simulate (no key, estimates) | record (live, writes fixtures) | "
        "replay (fixtures only) | live (no fixtures)",
    )
    bench.add_argument("--max-spend", default=None, help="hard USD cap, e.g. 5.00")
    bench.add_argument("--workload", type=Path, default=None)
    bench.add_argument("--json", type=Path, default=None, help="write results JSON here")
    bench.add_argument(
        "--no-random-curve",
        action="store_true",
        help="skip the random-routing baseline (faster, but removes the honest comparison)",
    )

    bench.add_argument(
        "--two-tier",
        action="store_true",
        help="binary high/low dispatch only -- is a dispatcher worth building?",
    )

    proxy = sub.add_parser(
        "proxy", help="run the dispatching proxy (Anthropic + OpenAI ingress)"
    )
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=int, default=8787)
    proxy.add_argument(
        "--strategy", default=None,
        choices=["cascade", "heuristic", "always_low", "always_high"],
        help="default: cascade (measured 59%% cheaper than always-high)",
    )
    proxy.add_argument("--low", default=None, help="low tier model id")
    proxy.add_argument("--high", default=None, help="high tier model id")
    proxy.add_argument(
        "--no-affinity", action="store_true",
        help="re-decide every turn (forfeits the prompt cache; usually a loss)",
    )

    sub.add_parser("tasks", help="show the workload mix")
    sub.add_parser(
        "levers",
        help="rank every cost lever by exact arithmetic (no API key needed)",
    )

    args = parser.parse_args(argv)

    # The dispatch verdict is DEFINED as "better than a coin at the same call
    # fraction". Without the random curve there is no verdict -- and silently
    # printing the rest would produce a table that looks authoritative while
    # omitting the only comparison that decides whether a dispatcher earned its
    # existence. Refuse the combination instead.
    if getattr(args, "two_tier", False) and getattr(args, "no_random_curve", False):
        parser.error(
            "--two-tier needs the random-routing baseline: the verdict is "
            "'better than a coin at the same %high', which cannot be computed "
            "without it. Drop --no-random-curve."
        )

    match args.command:
        case "tasks":
            return _tasks(args)
        case "levers":
            return _levers(args)
        case "proxy":
            return _proxy(args)
        case _:
            return _bench(args)


def _bench(args: argparse.Namespace) -> int:
    workload = Workload.load(args.workload)
    book = PriceBook.load()
    client = ModelClient(
        mode=Mode(args.mode),
        book=book,
        store=FixtureStore(Path(__file__).resolve().parents[2] / "fixtures"),
        max_spend_usd=args.max_spend,
    )

    try:
        report = run_bench(
            workload,
            client,
            two_tier_strategies() if args.two_tier else None,
            include_random_curve=not args.no_random_curve,
        )
    except MissingCredential as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SpendLimitExceeded as exc:
        print(f"stopped by spend guard: {exc}", file=sys.stderr)
        return 3
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    print(render(report))
    if args.two_tier:
        from dms.report import render_dispatch_value

        print()
        print(render_dispatch_value(report))

    destination = args.json or (DEFAULT_OUT / "results.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dumps(report) + "\n", encoding="utf-8")
    print(f"\nwrote {destination}")
    return 0


def _proxy(args: argparse.Namespace) -> int:
    from dataclasses import replace as _replace

    from dms.dispatch.config import DispatchConfig
    from dms.dispatch.server import serve

    config = DispatchConfig.from_env()
    overrides = {}
    if args.strategy:
        overrides["strategy"] = args.strategy
    if args.low:
        overrides["low_model"] = args.low
    if args.high:
        overrides["high_model"] = args.high
    if args.no_affinity:
        overrides["session_affinity"] = False
    if overrides:
        config = _replace(config, **overrides)

    return serve(args.host, args.port, config)


def _levers(_: argparse.Namespace) -> int:
    print(render_levers(PriceBook.load()))
    return 0


def _tasks(_: argparse.Namespace) -> int:
    workload = Workload.load()
    print(json.dumps({"total": len(workload), "mix": workload.mix()}, indent=2))
    for task in workload:
        print(f"  {task.id:<5} {task.difficulty:<7} {task.kind:<9} {task.grader}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
