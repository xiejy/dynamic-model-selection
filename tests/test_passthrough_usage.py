"""Reading token usage out of a response as it streams past, in both dialects."""
import json

import pytest

from dms.passthrough.usage import UsageSniffer


def _sse(events: list[tuple[str, dict]]) -> bytes:
    return b"".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events
    )


def _feed(sniffer: UsageSniffer, payload: bytes, chunk: int = 7) -> UsageSniffer:
    """Feed in small uneven chunks: SSE lines arrive split across reads."""
    for i in range(0, len(payload), chunk):
        sniffer.feed(payload[i:i + chunk])
    return sniffer


# ------------------------------------------------------------------ anthropic


ANTHROPIC_STREAM = _sse([
    ("message_start", {"type": "message_start", "message": {
        "model": "claude-opus-5",
        "usage": {"input_tokens": 12, "cache_creation_input_tokens": 3000,
                  "cache_read_input_tokens": 40000, "output_tokens": 1}}}),
    ("content_block_delta", {"type": "content_block_delta",
                             "delta": {"type": "text_delta", "text": "hi"}}),
    ("message_delta", {"type": "message_delta", "usage": {"output_tokens": 57}}),
    ("message_stop", {"type": "message_stop"}),
])


def test_anthropic_stream_takes_input_from_start_and_output_from_the_final_delta() -> None:
    """message_start carries the input side; message_delta carries the final,
    cumulative output count. Summing deltas would double count."""
    usage = _feed(UsageSniffer("anthropic"), ANTHROPIC_STREAM).result()

    assert usage.input_tokens == 12
    assert usage.cache_creation_input_tokens == 3000
    assert usage.cache_read_input_tokens == 40000
    assert usage.output_tokens == 57


def test_anthropic_json_response() -> None:
    body = json.dumps({"type": "message", "usage": {
        "input_tokens": 5, "output_tokens": 9,
        "cache_read_input_tokens": 100, "cache_creation_input_tokens": 0}}).encode()

    usage = _feed(UsageSniffer("anthropic"), body).result()

    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens) == (5, 9, 100)


# ------------------------------------------------------------------ openai responses


RESPONSES_STREAM = _sse([
    ("response.created", {"type": "response.created", "response": {"usage": None}}),
    ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "ok"}),
    ("response.completed", {"type": "response.completed", "response": {"usage": {
        "input_tokens": 16862, "input_tokens_details": {"cached_tokens": 11008},
        "output_tokens": 79, "output_tokens_details": {"reasoning_tokens": 40}}}}),
])


def test_responses_stream_subtracts_cached_tokens_from_input() -> None:
    """OpenAI's input_tokens INCLUDES the cached portion. Not subtracting it bills
    the cached tokens twice."""
    usage = _feed(UsageSniffer("openai"), RESPONSES_STREAM).result()

    assert usage.input_tokens == 16862 - 11008
    assert usage.cache_read_input_tokens == 11008
    assert usage.output_tokens == 79          # reasoning is a subset, not added
    assert usage.prompt_tokens == 16862


def test_responses_json_response() -> None:
    body = json.dumps({"object": "response", "usage": {
        "input_tokens": 100, "input_tokens_details": {"cached_tokens": 60},
        "output_tokens": 7}}).encode()

    usage = _feed(UsageSniffer("openai"), body).result()

    assert (usage.input_tokens, usage.cache_read_input_tokens, usage.output_tokens) == (40, 60, 7)


# ------------------------------------------------------------------ robustness


def test_no_usage_reported_gives_none_not_zero() -> None:
    """An error body or a cut-off stream must not be logged as a free request."""
    assert _feed(UsageSniffer("anthropic"), b'{"type":"error"}').result() is None
    assert _feed(UsageSniffer("openai"), b"data: [DONE]\n\n").result() is None


def test_garbage_and_keepalives_are_ignored() -> None:
    payload = b": ping\n\ndata: {not json\n\n" + RESPONSES_STREAM

    assert _feed(UsageSniffer("openai"), payload).result().output_tokens == 79


def test_the_body_is_never_retained_beyond_one_line() -> None:
    """A long stream must not accumulate in memory."""
    sniffer = UsageSniffer("anthropic")
    _feed(sniffer, ANTHROPIC_STREAM * 50)

    assert sniffer.buffered_bytes < 4096


# ------------------------------------------------------------- review findings


def test_anthropic_stream_cut_off_before_its_final_count_gives_none() -> None:
    """message_start's output_tokens is a placeholder 1. A stream that stops
    before message_delta has no real count; it is 'no usage', not 1 token."""
    cut = ANTHROPIC_STREAM[: ANTHROPIC_STREAM.index(b"event: message_delta")]

    assert _feed(UsageSniffer("anthropic"), cut).result() is None


@pytest.mark.parametrize("kind", ["response.incomplete", "response.failed"])
def test_responses_that_end_early_still_report_their_usage(kind) -> None:
    """A response cut at max_output_tokens is still billed."""
    stream = _sse([(kind, {"type": kind, "response": {"usage": {
        "input_tokens": 100, "input_tokens_details": {"cached_tokens": 40},
        "output_tokens": 7}}})])

    usage = _feed(UsageSniffer("openai"), stream).result()

    assert (usage.input_tokens, usage.cache_read_input_tokens, usage.output_tokens) == (60, 40, 7)
