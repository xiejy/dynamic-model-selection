"""Client modes: record / replay / simulate, and the spend guard."""
import json

import pytest

from dms.client import Call, ModelClient, Mode, SpendLimitExceeded
from dms.pricing import PriceBook
from dms.replay import FixtureStore
from dms.usage import UsageRecord


@pytest.fixture
def store(tmp_path) -> FixtureStore:
    return FixtureStore(tmp_path / "fixtures")


@pytest.fixture
def book() -> PriceBook:
    return PriceBook.load()


# --------------------------------------------------------------------------- fixtures


def test_fixture_key_is_stable_across_dict_ordering(store: FixtureStore) -> None:
    """Key must be content-addressed, not order-addressed, or every rerun misses."""
    a = store.key(model="m", system="s", prompt="p", options={"x": 1, "y": 2})
    b = store.key(model="m", system="s", prompt="p", options={"y": 2, "x": 1})

    assert a == b


def test_fixture_key_changes_when_the_model_changes(store: FixtureStore) -> None:
    a = store.key(model="claude-opus-5", system="s", prompt="p", options={})
    b = store.key(model="claude-haiku-4-5", system="s", prompt="p", options={})

    assert a != b


def test_round_trips_a_recorded_call(store: FixtureStore) -> None:
    key = store.key(model="m", system=None, prompt="p", options={})
    call = Call(
        model="m",
        text="hello",
        usage=UsageRecord(input_tokens=10, output_tokens=3),
        latency_ms=42.0,
    )

    store.put(key, call)

    assert store.get(key) == call


def test_missing_fixture_returns_none(store: FixtureStore) -> None:
    assert store.get("nope") is None


def test_fixtures_persist_to_disk_as_readable_json(store: FixtureStore, tmp_path) -> None:
    key = store.key(model="m", system=None, prompt="p", options={})
    store.put(key, Call(model="m", text="hi", usage=UsageRecord(output_tokens=1)))

    written = list((tmp_path / "fixtures").glob("*.json"))

    assert len(written) == 1
    assert json.loads(written[0].read_text())["text"] == "hi"


# --------------------------------------------------------------------------- modes


def test_replay_mode_never_touches_the_network(store: FixtureStore, book: PriceBook) -> None:
    client = ModelClient(mode=Mode.REPLAY, store=store, book=book, api=_ExplodingAPI())
    key = client.fixture_key(model="claude-opus-5", prompt="ping")
    store.put(key, Call(model="claude-opus-5", text="pong", usage=UsageRecord(output_tokens=1)))

    call = client.complete(model="claude-opus-5", prompt="ping")

    assert call.text == "pong"


def test_replay_mode_fails_loudly_on_a_missing_fixture(
    store: FixtureStore, book: PriceBook
) -> None:
    client = ModelClient(mode=Mode.REPLAY, store=store, book=book, api=_ExplodingAPI())

    with pytest.raises(LookupError, match="no fixture"):
        client.complete(model="claude-opus-5", prompt="unrecorded")


def test_simulate_mode_is_deterministic(book: PriceBook, store: FixtureStore) -> None:
    """Two runs of the same prompt must agree, or the bench is not reproducible."""
    first = ModelClient(mode=Mode.SIMULATE, store=store, book=book).complete(
        model="claude-haiku-4-5", prompt="classify this log line"
    )
    second = ModelClient(mode=Mode.SIMULATE, store=store, book=book).complete(
        model="claude-haiku-4-5", prompt="classify this log line"
    )

    assert first == second


def test_simulate_marks_its_calls_as_not_measured(book: PriceBook, store: FixtureStore) -> None:
    call = ModelClient(mode=Mode.SIMULATE, store=store, book=book).complete(
        model="claude-opus-5", prompt="hello"
    )

    assert call.simulated is True


def test_recorded_calls_are_marked_as_measured(store: FixtureStore, book: PriceBook) -> None:
    client = ModelClient(mode=Mode.REPLAY, store=store, book=book, api=_ExplodingAPI())
    key = client.fixture_key(model="claude-opus-5", prompt="ping")
    store.put(key, Call(model="claude-opus-5", text="pong", usage=UsageRecord(output_tokens=1)))

    assert client.complete(model="claude-opus-5", prompt="ping").simulated is False


def test_longer_prompts_simulate_more_input_tokens(book: PriceBook, store: FixtureStore) -> None:
    client = ModelClient(mode=Mode.SIMULATE, store=store, book=book)

    short = client.complete(model="claude-haiku-4-5", prompt="hi")
    long = client.complete(model="claude-haiku-4-5", prompt="word " * 500)

    assert long.usage.input_tokens > short.usage.input_tokens


# --------------------------------------------------------------------------- spend


def test_spend_guard_trips_before_exceeding_the_budget(
    book: PriceBook, store: FixtureStore
) -> None:
    """The cap applies in every mode, so a spend plan can be dry-run offline."""
    client = ModelClient(mode=Mode.SIMULATE, store=store, book=book, max_spend_usd="0.0000001")

    with pytest.raises(SpendLimitExceeded):
        for _ in range(50):
            client.complete(model="claude-opus-5", prompt="expensive " * 200)


def test_client_accumulates_spend_and_usage(book: PriceBook, store: FixtureStore) -> None:
    client = ModelClient(mode=Mode.SIMULATE, store=store, book=book)

    client.complete(model="claude-haiku-4-5", prompt="one")
    client.complete(model="claude-haiku-4-5", prompt="two")

    assert client.calls_made == 2
    assert client.total_usage.output_tokens > 0
    assert client.total_spend_usd > 0


class _ExplodingAPI:
    """Any network use in replay mode is a bug."""

    def __getattr__(self, name: str):  # pragma: no cover - defensive
        raise AssertionError(f"replay mode touched the network via .{name}")
