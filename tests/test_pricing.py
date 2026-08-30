"""Money layer: usage parsing and exact cost arithmetic."""
from decimal import Decimal

import pytest

from dms.pricing import PriceBook, CacheTTL
from dms.usage import UsageRecord


# --------------------------------------------------------------------------- usage


def test_prompt_tokens_sums_all_three_input_fields() -> None:
    # Arrange -- input_tokens is the UNCACHED REMAINDER only, not the whole prompt.
    usage = UsageRecord(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=2000,
        cache_read_input_tokens=8000,
    )

    # Act / Assert
    assert usage.prompt_tokens == 10100


def test_usage_parses_from_an_sdk_response_shape() -> None:
    class FakeUsage:
        input_tokens = 7
        output_tokens = 3
        cache_creation_input_tokens = 11
        cache_read_input_tokens = 13

    class FakeResponse:
        usage = FakeUsage()

    record = UsageRecord.from_response(FakeResponse())

    assert (record.input_tokens, record.output_tokens) == (7, 3)
    assert (record.cache_creation_input_tokens, record.cache_read_input_tokens) == (11, 13)


def test_usage_tolerates_missing_cache_fields() -> None:
    """Responses without caching omit the cache fields entirely."""

    class FakeUsage:
        input_tokens = 5
        output_tokens = 2

    class FakeResponse:
        usage = FakeUsage()

    record = UsageRecord.from_response(FakeResponse())

    assert record.cache_creation_input_tokens == 0
    assert record.cache_read_input_tokens == 0
    assert record.prompt_tokens == 5


def test_usage_records_add() -> None:
    a = UsageRecord(input_tokens=1, output_tokens=2, cache_read_input_tokens=3)
    b = UsageRecord(input_tokens=10, output_tokens=20, cache_creation_input_tokens=30)

    total = a + b

    assert total == UsageRecord(
        input_tokens=11,
        output_tokens=22,
        cache_creation_input_tokens=30,
        cache_read_input_tokens=3,
    )


def test_usage_is_immutable() -> None:
    usage = UsageRecord(input_tokens=1, output_tokens=1)

    with pytest.raises(Exception):
        usage.input_tokens = 999  # type: ignore[misc]


# --------------------------------------------------------------------------- prices


@pytest.fixture
def book() -> PriceBook:
    return PriceBook.load()


def test_intro_snapshot_applies_before_the_expiry(book: PriceBook) -> None:
    rate = book.rate("claude-sonnet-5", at="2026-08-24T00:00:00Z")

    assert rate.input_usd_per_million == Decimal("2.00")
    assert rate.output_usd_per_million == Decimal("10.00")


def test_sticker_snapshot_applies_after_the_intro_expires(book: PriceBook) -> None:
    """The Sonnet 5 introductory rate lapses 2026-08-31 -- a cost model built this
    week silently under-reports from September onward unless snapshots are dated."""
    rate = book.rate("claude-sonnet-5", at="2026-09-01T00:00:00Z")

    assert rate.input_usd_per_million == Decimal("3.00")
    assert rate.output_usd_per_million == Decimal("15.00")


def test_alias_resolves_to_full_model_id(book: PriceBook) -> None:
    assert book.resolve("opus") == "claude-opus-5"
    assert book.resolve("claude-opus-5") == "claude-opus-5"


def test_unknown_model_is_rejected_loudly(book: PriceBook) -> None:
    with pytest.raises(KeyError, match="gpt-4"):
        book.rate("gpt-4", at="2026-08-24T00:00:00Z")


def test_plain_cost_is_input_plus_output(book: PriceBook) -> None:
    # Arrange -- 1M input + 1M output on Opus 5 ($5 / $25).
    usage = UsageRecord(input_tokens=1_000_000, output_tokens=1_000_000)

    # Act
    cost = book.cost_usd(usage, "claude-opus-5", at="2026-08-24T00:00:00Z")

    # Assert
    assert cost == Decimal("30.00")


def test_cache_read_is_a_tenth_of_input_price(book: PriceBook) -> None:
    usage = UsageRecord(input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000)

    cost = book.cost_usd(usage, "claude-opus-5", at="2026-08-24T00:00:00Z")

    assert cost == Decimal("0.50")  # 5.00 * 0.10


def test_cache_write_carries_a_premium_that_differs_by_ttl(book: PriceBook) -> None:
    usage = UsageRecord(
        input_tokens=0, output_tokens=0, cache_creation_input_tokens=1_000_000
    )

    five_min = book.cost_usd(
        usage, "claude-opus-5", at="2026-08-24T00:00:00Z", cache_ttl=CacheTTL.FIVE_MINUTES
    )
    one_hour = book.cost_usd(
        usage, "claude-opus-5", at="2026-08-24T00:00:00Z", cache_ttl=CacheTTL.ONE_HOUR
    )

    assert five_min == Decimal("6.25")  # 5.00 * 1.25
    assert one_hour == Decimal("10.00")  # 5.00 * 2.00


def test_batch_halves_everything(book: PriceBook) -> None:
    usage = UsageRecord(input_tokens=1_000_000, output_tokens=1_000_000)

    cost = book.cost_usd(usage, "claude-opus-5", at="2026-08-24T00:00:00Z", batch=True)

    assert cost == Decimal("15.00")


def test_cost_uses_exact_decimal_not_binary_float(book: PriceBook) -> None:
    """0.1 + 0.2 != 0.3 in binary float; money math must not drift."""
    usage = UsageRecord(input_tokens=1, output_tokens=0)

    cost = book.cost_usd(usage, "claude-opus-5", at="2026-08-24T00:00:00Z")

    assert isinstance(cost, Decimal)
    assert cost == Decimal("5") / Decimal("1000000")


# ------------------------------------------------------- the routing/caching tension


def test_haiku_has_an_eight_times_higher_cache_floor_than_opus(book: PriceBook) -> None:
    """The cheap model is the HARDER one to cache. This asymmetry is what makes
    routing down to Haiku net-negative on a cache-warm workload."""
    assert book.min_cache_prefix_tokens("claude-opus-5") == 512
    assert book.min_cache_prefix_tokens("claude-haiku-4-5") == 4096


@pytest.mark.parametrize(
    "model,prefix_tokens,expected",
    [
        ("claude-opus-5", 3000, True),
        ("claude-haiku-4-5", 3000, False),  # silently does not cache
        ("claude-haiku-4-5", 5000, True),
        ("claude-sonnet-5", 1024, True),
        ("claude-sonnet-5", 1023, False),
    ],
)
def test_will_cache_predicts_the_silent_no_op(
    book: PriceBook, model: str, prefix_tokens: int, expected: bool
) -> None:
    assert book.will_cache(model, prefix_tokens) is expected


# ------------------------------------------------------ multi-provider pricing


def test_openai_models_use_their_own_stated_cached_rate(book: PriceBook) -> None:
    """Applying Anthropic's 0.10x multiplier to a GPT model would be wrong --
    OpenAI publishes the cached-input rate outright."""
    usage = UsageRecord(cache_read_input_tokens=1_000_000)

    cost = book.cost_usd(usage, "gpt-5.6-luna", at="2026-08-24T00:00:00Z")

    assert cost == Decimal("0.1")  # the stated rate, not 1.00 * 0.10


def test_openai_cache_writes_carry_no_premium(book: PriceBook) -> None:
    """OpenAI caching is automatic: a write is billed as ordinary input, with no
    1.25x/2x write premium the way Anthropic charges."""
    usage = UsageRecord(cache_creation_input_tokens=1_000_000)

    cost = book.cost_usd(usage, "gpt-5.6-luna", at="2026-08-24T00:00:00Z")

    assert cost == Decimal("1")  # plain input rate


def test_anthropic_still_uses_the_multiplier(book: PriceBook) -> None:
    """Regression guard: adding a provider must not change Claude's arithmetic."""
    usage = UsageRecord(cache_read_input_tokens=1_000_000)

    assert book.cost_usd(usage, "claude-opus-5", at="2026-08-24T00:00:00Z") == Decimal("0.50")


def test_codex_alias_resolves(book: PriceBook) -> None:
    assert book.resolve("codex") == "gpt-5.6-sol"


def test_automatic_caching_providers_report_no_cache_floor(book: PriceBook) -> None:
    """Claude models have a minimum cacheable prefix; OpenAI's is not a concept."""
    assert book.min_cache_prefix_tokens("gpt-5.6-luna") == 0
    assert book.will_cache("gpt-5.6-luna", 10) is True
