"""The usage ledger: every request's tier, tokens and cost, and the comparison."""
import json
from decimal import Decimal

from dms.passthrough.ledger import Ledger, UsageEvent, render, summarise
from dms.pricing import PriceBook
from dms.usage import UsageRecord

LOW, HIGH = "gpt-5.6-luna@personal", "gpt-5.6-sol@personal"
MILLION = UsageRecord(input_tokens=1_000_000)


def _event(tier: str, model: str, usage=MILLION, session="s1", status=200) -> UsageEvent:
    return UsageEvent(
        ts="2026-09-24T10:00:00Z", family="openai", session=session,
        requested_model=HIGH, routed_model=model, tier=tier, turn="new",
        why="test", status=status, latency_ms=100.0, usage=usage,
    )


def test_counts_split_by_tier() -> None:
    events = [_event("low", LOW), _event("low", LOW), _event("high", HIGH),
              _event("untouched", "gpt-5.5@personal")]

    s = summarise(events, PriceBook.load())

    assert (s.requests, s.low, s.high, s.untouched) == (4, 2, 1, 1)


def test_cost_is_compared_against_sending_everything_high() -> None:
    """Luna $1/M input, Sol $5/M: two low requests cost $2 but would have cost
    $10 on Sol, and the high request is $5 either way. (Three conversations:
    within one, a switch changes what is cached -- tested below.)"""
    events = [_event("low", LOW, session="a"), _event("low", LOW, session="b"),
              _event("high", HIGH, session="c")]

    s = summarise(events, PriceBook.load())

    assert s.actual_usd == Decimal(7)
    assert s.all_high_usd == Decimal(15)
    assert s.saved_usd == Decimal(8)
    assert round(s.saved_share, 4) == round(8 / 15, 4)


def test_untouched_requests_are_counted_but_not_in_the_comparison() -> None:
    """They would have cost the same either way; including them dilutes the
    saving the router is responsible for."""
    events = [_event("low", LOW), _event("untouched", "gpt-5.5@personal")]

    s = summarise(events, PriceBook.load())

    assert s.all_high_usd == Decimal(5)
    assert s.untouched_usd > 0


def test_a_request_with_no_usage_is_counted_separately_not_as_free() -> None:
    events = [_event("low", LOW), _event("low", LOW, usage=None, status=500)]

    s = summarise(events, PriceBook.load())

    assert s.no_usage == 1
    assert s.actual_usd == Decimal(1)


def test_unpriced_models_are_reported_not_guessed() -> None:
    events = [_event("untouched", "gpt-6-astra@personal")]

    s = summarise(events, PriceBook.load())

    assert s.unpriced == {"gpt-6-astra@personal"}


def test_per_session_breakdown() -> None:
    events = [_event("low", LOW, session="a"), _event("high", HIGH, session="b")]

    s = summarise(events, PriceBook.load())

    assert s.sessions["a"]["low"] == 1 and s.sessions["b"]["high"] == 1


def test_model_switches_within_a_session_are_counted() -> None:
    """Each switch re-reads the conversation on a cold cache -- worth seeing."""
    events = [_event("low", LOW), _event("high", HIGH), _event("high", HIGH),
              _event("low", LOW)]

    assert summarise(events, PriceBook.load()).switches == 2


def test_the_ledger_persists_as_jsonl_and_reads_back(tmp_path) -> None:
    ledger = Ledger(tmp_path / "usage.jsonl")
    ledger.record(_event("low", LOW))
    ledger.record(_event("high", HIGH, usage=None))

    events = Ledger(tmp_path / "usage.jsonl").read()

    assert [e.tier for e in events] == ["low", "high"]
    assert events[0].usage == MILLION and events[1].usage is None


def test_the_ledger_never_stores_prompts_or_credentials(tmp_path) -> None:
    ledger = Ledger(tmp_path / "usage.jsonl")
    ledger.record(_event("low", LOW))

    row = json.loads((tmp_path / "usage.jsonl").read_text())

    assert set(row) == {
        "ts", "family", "session", "requested_model", "routed_model", "tier",
        "turn", "why", "status", "latency_ms", "usage",
    }


def test_the_report_shows_the_split_and_the_saving() -> None:
    events = [_event("low", LOW), _event("high", HIGH)]

    text = render(summarise(events, PriceBook.load()))

    for fragment in ("low", "high", "all-high", "saved"):
        assert fragment in text.lower()


# ------------------------------------------------------------------------- cli


def test_the_usage_command_reports_each_family_separately(tmp_path, capsys) -> None:
    from dms.cli import main

    ledger = Ledger(tmp_path / "usage.jsonl")
    ledger.record(_event("low", LOW))
    ledger.record(_event("high", HIGH))
    ledger.record(UsageEvent(
        ts="2026-09-24T10:01:00Z", family="anthropic", session="c1",
        requested_model="claude-opus-5", routed_model="claude-sonnet-5", tier="low",
        turn="new", why="t", status=200, latency_ms=1.0, usage=MILLION))

    assert main(["usage", "--usage-log", str(tmp_path / "usage.jsonl")]) == 0

    out = capsys.readouterr().out
    assert "== anthropic ==" in out and "== openai ==" in out
    assert "claude-opus-5" in out  # the counterfactual is the family's own high model


def test_the_usage_command_handles_an_empty_ledger(tmp_path, capsys) -> None:
    from dms.cli import main

    assert main(["usage", "--usage-log", str(tmp_path / "none.jsonl")]) == 0
    assert "no requests recorded" in capsys.readouterr().out


# ------------------------------------------------------------- review findings

OPUS, SONNET = "claude-opus-5", "claude-sonnet-5"
PREFIX = 66_000


def _claude_event(model: str, usage: UsageRecord, tier: str, ts: str,
                  session: str = "c1") -> UsageEvent:
    return UsageEvent(
        ts=ts, family="anthropic", session=session, requested_model=OPUS,
        routed_model=model, tier=tier, turn="new", why="t", status=200,
        latency_ms=1.0, usage=usage,
    )


def test_a_switch_is_not_counted_as_a_saving() -> None:
    """Caches are per model. Opus writes a 66k prefix; the next message goes to
    Sonnet, which must write it again; the one after returns to Opus and reads
    it. An all-Opus run would have read the prefix on the middle request too.
    Re-pricing Sonnet's cold write at Opus's write rate made routing look
    $0.17 cheaper when it cost $0.21 more."""
    events = [
        _claude_event(OPUS, UsageRecord(cache_creation_input_tokens=PREFIX), "high",
                      "2026-09-24T10:00:00+00:00"),
        _claude_event(SONNET, UsageRecord(cache_creation_input_tokens=PREFIX), "low",
                      "2026-09-24T10:01:00+00:00"),
        _claude_event(OPUS, UsageRecord(cache_read_input_tokens=PREFIX), "high",
                      "2026-09-24T10:02:00+00:00"),
    ]

    s = summarise(events, PriceBook.load())

    # all-Opus: one 66k write at $6.25/M, then two 66k reads at $0.50/M
    assert s.all_high_usd == Decimal("0.4125") + 2 * Decimal("0.033")
    assert s.actual_usd > s.all_high_usd
    assert s.saved_usd < 0


def test_a_switch_after_the_cache_would_have_expired_is_a_cold_start_either_way() -> None:
    events = [
        _claude_event(OPUS, UsageRecord(cache_creation_input_tokens=PREFIX), "high",
                      "2026-09-24T10:00:00+00:00"),
        _claude_event(SONNET, UsageRecord(cache_creation_input_tokens=PREFIX), "low",
                      "2026-09-24T12:00:00+00:00"),
    ]

    s = summarise(events, PriceBook.load())

    assert s.all_high_usd == 2 * Decimal("0.4125")


def test_only_the_prefix_already_sent_would_have_been_a_cache_read() -> None:
    """The new tail of the switched request is a write in either world."""
    events = [
        _claude_event(OPUS, UsageRecord(cache_creation_input_tokens=PREFIX), "high",
                      "2026-09-24T10:00:00+00:00"),
        _claude_event(SONNET, UsageRecord(cache_creation_input_tokens=PREFIX + 4_000), "low",
                      "2026-09-24T10:01:00+00:00"),
    ]

    s = summarise(events, PriceBook.load())

    assert s.all_high_usd == Decimal("0.4125") + Decimal("0.033") + Decimal("0.025")


def test_each_request_is_compared_with_the_model_it_asked_for() -> None:
    """The ledger spans runs. Requests from a run whose high model was Terra
    must be compared with Terra, not with the most common high model."""
    terra = "gpt-5.6-terra@personal"
    events = [_event("low", LOW), _event("low", LOW), _event("low", LOW),
              UsageEvent(ts="2026-09-24T11:00:00Z", family="openai", session="s2",
                         requested_model=terra, routed_model=terra, tier="high",
                         turn="new", why="t", status=200, latency_ms=1.0, usage=MILLION),
              UsageEvent(ts="2026-09-24T11:00:00Z", family="openai", session="s2",
                         requested_model=terra, routed_model=terra, tier="high",
                         turn="new", why="t", status=200, latency_ms=1.0, usage=MILLION)]

    s = summarise(events, PriceBook.load())

    assert s.saved_usd == 3 * (Decimal(5) - Decimal(1))
    assert HIGH in s.high_model and terra in s.high_model


def _opening(model: str, session: str, ts: str, *, write: int, read: int) -> UsageEvent:
    tier = "high" if model == OPUS else "low"
    return _claude_event(model, UsageRecord(cache_creation_input_tokens=write,
                                            cache_read_input_tokens=read), tier, ts, session)


def test_the_prefix_shared_across_sessions_counts_against_a_cold_model() -> None:
    """Claude Code's system prompt and tools are the same in every session, so
    a second session opens on a cache read (measured: 49k and 52k of 66k). An
    Opus request that opens cold only because routing left Opus unused would,
    in an all-Opus run, have read that shared prefix too."""
    shared, tail = 60_000, 6_000
    events = [
        _opening(SONNET, "s1", "2026-09-24T10:00:00+00:00", write=shared + tail, read=0),
        _opening(SONNET, "s2", "2026-09-24T10:00:30+00:00", write=tail, read=shared),
        _opening(OPUS, "s3", "2026-09-24T10:01:00+00:00", write=shared + tail, read=0),
    ]

    s = summarise(events, PriceBook.load())

    one_opening_cold = Decimal("0.4125")                          # 66k written at $6.25/M
    one_opening_warm = Decimal("0.03") + Decimal("0.0375")        # 60k read + 6k written
    assert s.all_high_usd == one_opening_cold + 2 * one_opening_warm


def test_no_cross_session_credit_without_a_measured_shared_prefix() -> None:
    """Codex's cache is keyed per session: every opening read 0 in the live
    run, so nothing is assumed shared."""
    events = [
        _opening(SONNET, "s1", "2026-09-24T10:00:00+00:00", write=66_000, read=0),
        _opening(OPUS, "s2", "2026-09-24T10:00:30+00:00", write=66_000, read=0),
    ]

    assert summarise(events, PriceBook.load()).all_high_usd == 2 * Decimal("0.4125")


def test_no_cross_session_credit_when_the_model_was_warm_anyway() -> None:
    """A cold opening on a model that routing had just used is a miss either way."""
    shared = 60_000
    events = [
        _opening(OPUS, "s1", "2026-09-24T10:00:00+00:00", write=shared, read=0),
        _opening(SONNET, "s2", "2026-09-24T10:00:10+00:00", write=0, read=shared),
        _opening(OPUS, "s3", "2026-09-24T10:00:20+00:00", write=shared, read=0),
    ]

    s = summarise(events, PriceBook.load())

    assert s.all_high_usd == 2 * Decimal("0.375") + Decimal("0.03")
