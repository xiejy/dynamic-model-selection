"""Reporting and CLI. The banner assertions are an honesty guard, not cosmetics."""
import json

import pytest

from dms.bench import run_bench
from dms.cli import main
from dms.client import Mode, ModelClient
from dms.pricing import PriceBook
from dms.replay import FixtureStore
from dms.report import dumps, render, render_levers, to_dict
from dms.routers.baseline import AlwaysRouter
from dms.strategies import RoutedStrategy
from dms.workload import Workload


@pytest.fixture(scope="module")
def report():
    book = PriceBook.load()
    client = ModelClient(mode=Mode.SIMULATE, book=book, store=FixtureStore("/tmp/unused"))
    strategies = (
        RoutedStrategy(AlwaysRouter(model="claude-haiku-4-5")),
        RoutedStrategy(AlwaysRouter(model="claude-opus-5")),
    )
    return run_bench(Workload.load(), client, strategies, include_random_curve=False)


# --------------------------------------------------------------------------- text


def test_a_simulated_run_says_so_loudly(report) -> None:
    """If this ever silently flips, a talk quotes estimates as measurements."""
    text = render(report)

    assert "SIMULATED RUN" in text
    assert "ESTIMATES" in text


def test_report_contains_every_table(report) -> None:
    text = render(report)

    for heading in (
        "COST vs QUALITY",
        "DOES THE ROUTER EARN ITS KEEP?",
        "PARETO FRONTIER",
        "ACCURACY BY DIFFICULTY",
    ):
        assert heading in text


def test_levers_report_states_it_is_exact_arithmetic() -> None:
    text = render_levers(PriceBook.load())

    assert "EXACT ARITHMETIC" in text
    assert "CROSSOVER" in text
    assert "SILENT NO-OP" in text


def test_levers_report_refuses_to_invent_an_effort_number() -> None:
    text = render_levers(PriceBook.load())

    assert "effort" in text
    assert "no per-level token ratio" in text


# --------------------------------------------------------------------------- json


def test_json_export_is_serialisable_and_flags_simulation(report) -> None:
    payload = json.loads(dumps(report))

    assert payload["simulated"] is True
    assert payload["mode"] == "simulate"
    assert len(payload["strategies"]) == 2


def test_json_keeps_money_as_strings_not_floats(report) -> None:
    """Decimal -> float would reintroduce the drift the price book avoids."""
    payload = to_dict(report)

    assert isinstance(payload["strategies"][0]["cost_usd"], str)
    assert isinstance(payload["total_spend_usd"], str)


def test_json_records_the_router_and_waste_split(report) -> None:
    strategy = to_dict(report)["strategies"][0]

    assert "router_cost_usd" in strategy
    assert "wasted_cost_usd" in strategy
    assert "vs_random" in strategy


# ---------------------------------------------------------------------------- cli


def test_cli_bench_runs_offline_and_writes_json(tmp_path, capsys) -> None:
    destination = tmp_path / "results.json"

    code = main(
        ["bench", "--mode", "simulate", "--json", str(destination), "--no-random-curve"]
    )

    assert code == 0
    assert destination.is_file()
    assert "SIMULATED RUN" in capsys.readouterr().out


def test_cli_replay_without_fixtures_fails_cleanly(tmp_path, capsys) -> None:
    """A missing fixture must be an actionable message, not a traceback."""
    code = main(["bench", "--mode", "replay", "--json", str(tmp_path / "x.json")])

    assert code == 4
    assert "no fixture" in capsys.readouterr().err


def test_cli_spend_guard_stops_the_run(tmp_path, capsys) -> None:
    code = main(
        [
            "bench", "--mode", "simulate", "--max-spend", "0.0000001",
            "--json", str(tmp_path / "x.json"),
        ]
    )

    assert code == 3
    assert "spend guard" in capsys.readouterr().err


def test_cli_levers_needs_no_credential(capsys) -> None:
    assert main(["levers"]) == 0
    assert "CROSSOVER" in capsys.readouterr().out


def test_cli_tasks_lists_the_mix(capsys) -> None:
    assert main(["tasks"]) == 0

    out = capsys.readouterr().out

    assert "easy" in out and "hard" in out


def test_two_tier_refuses_to_run_without_the_random_baseline(capsys) -> None:
    """The dispatch verdict is defined by the coin comparison. Producing the
    table without it would look authoritative while omitting the only number
    that decides whether the routing logic earned its existence."""
    with pytest.raises(SystemExit) as exit_info:
        main(["bench", "--mode", "simulate", "--two-tier", "--no-random-curve"])

    assert exit_info.value.code != 0
    assert "random-routing baseline" in capsys.readouterr().err


def test_two_tier_runs_with_the_baseline(tmp_path, capsys) -> None:
    code = main(
        ["bench", "--mode", "simulate", "--two-tier", "--json", str(tmp_path / "t.json")]
    )

    out = capsys.readouterr().out

    assert code == 0
    assert "IS THE DISPATCHER WORTH IT?" in out
    assert "AGAINST THE TWO TRIVIAL ANSWERS" in out
