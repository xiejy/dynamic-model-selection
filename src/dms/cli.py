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

    passthrough = sub.add_parser(
        "passthrough",
        help="choose low/high per message, forward everything else to your router",
    )
    passthrough.add_argument("--family", required=True, choices=["openai", "anthropic"],
                             help="openai for Codex, anthropic for Claude Code")
    passthrough.add_argument("--upstream", required=True,
                             help="the router the client normally uses, e.g. http://127.0.0.1:18790")
    passthrough.add_argument("--high", required=True,
                             help="the main model exactly as the client sends it")
    passthrough.add_argument("--low", required=True, help="the cheaper model to route down to")
    passthrough.add_argument("--port", type=int, default=8789)
    passthrough.add_argument("--host", default="127.0.0.1")
    passthrough.add_argument("--usage-log", type=Path, default=None,
                             help="JSONL ledger (default ~/.dms/usage.jsonl)")

    usage = sub.add_parser("usage", help="low/high split and cost from the pass-through ledger")
    usage.add_argument("--usage-log", type=Path, default=None)
    usage.add_argument("--json", action="store_true")

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
        case "passthrough":
            return _passthrough(args)
        case "usage":
            return _usage(args)
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


def _passthrough(args: argparse.Namespace) -> int:
    import errno
    import logging

    from dms.passthrough.ledger import DEFAULT_LEDGER, Ledger
    from dms.passthrough.select import PassthroughSelector, downgrade_warning
    from dms.passthrough.server import build_passthrough_server

    book = PriceBook.load()
    low_context = book.context_window.get(book.resolve(args.low))
    selector = PassthroughSelector(
        family=args.family, low=args.low, high=args.high, low_context_tokens=low_context
    )
    ledger = Ledger(args.usage_log or DEFAULT_LEDGER)
    try:
        server = build_passthrough_server(
            args.host, args.port, family=args.family, upstream=args.upstream,
            selector=selector, ledger=ledger, book=book,
        )
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        print(f"error: {args.host}:{args.port} is already in use; pick another with --port",
              file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base = f"http://{args.host}:{args.port}"
    print(f"dms passthrough ({args.family}) on {base} -> {args.upstream}")
    print(f"  high  {args.high}   (the only model it ever changes)")
    print(f"  low   {args.low}" + (f"   (context {low_context:,} tokens)" if low_context else ""))
    print(f"  usage {ledger.path}   ·   live totals: {base}/_dms/stats")
    if warning := downgrade_warning(args.family, args.low):
        print(f"  WARNING: {warning}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped -- run `dms usage` for the report")
    finally:
        server.server_close()
    return 0


def _usage(args: argparse.Namespace) -> int:
    from dms.passthrough.ledger import DEFAULT_LEDGER, Ledger, _as_dict, render, summarise

    ledger = Ledger(args.usage_log or DEFAULT_LEDGER)
    events = ledger.read()
    if not events:
        print(f"no requests recorded in {ledger.path}")
        return 0

    book = PriceBook.load()
    reports = {
        family: summarise([e for e in events if e.family == family], book)
        for family in sorted({e.family for e in events})
    }

    if args.json:
        print(json.dumps({f: _as_dict(s) for f, s in reports.items()}, indent=2))
        return 0
    print(f"ledger {ledger.path}  ·  {events[0].ts} .. {events[-1].ts}")
    for family, summary in reports.items():
        print(f"\n== {family} ==")
        print(render(summary))
    return 0


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
