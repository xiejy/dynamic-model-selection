"""Unit coverage for the proxy's edges: config, wire formats, provider payloads.

These are the parts a fake-provider integration test flies straight past -- the
translation layers where a silent mistake becomes a wrong bill or a malformed
response rather than an exception.
"""
import json
from dataclasses import replace
from decimal import Decimal

import pytest

from dms.dispatch import wire
from dms.dispatch.config import DispatchConfig
from dms.dispatch.core import DispatchResult, Leg
from dms.dispatch.providers import (
    AnthropicProvider,
    OpenAIProvider,
    ProviderError,
    Request,
    _rejects_sampling,
)
from dms.usage import UsageRecord

LOW = "claude-haiku-4-5"
HIGH = "claude-opus-5"


# --------------------------------------------------------------------- config


def test_tier_lookups_round_trip() -> None:
    config = DispatchConfig(low_model=LOW, high_model=HIGH)

    assert config.model_for("low") == LOW
    assert config.tier_of(HIGH) == "high"


def test_unknown_tier_is_rejected() -> None:
    with pytest.raises(ValueError, match="tier must be"):
        DispatchConfig().model_for("medium")


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="not one of"):
        DispatchConfig(strategy="vibes")


def test_negative_ttl_is_rejected() -> None:
    with pytest.raises(ValueError, match="affinity_ttl_seconds"):
        DispatchConfig(affinity_ttl_seconds=-1)


def test_env_overrides_apply() -> None:
    config = DispatchConfig.from_env(
        {"DMS_LOW_MODEL": "gpt-5.6-luna", "DMS_HIGH_MODEL": HIGH,
         "DMS_STRATEGY": "heuristic", "DMS_SESSION_AFFINITY": "false"}
    )

    assert config.low_model == "gpt-5.6-luna"
    assert config.strategy == "heuristic"
    assert config.session_affinity is False


def test_env_with_nothing_set_gives_defaults() -> None:
    assert DispatchConfig.from_env({}).strategy == "cascade"


def test_config_file_round_trips(tmp_path) -> None:
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"strategy": "heuristic", "low_model": "gpt-5"}))

    config = DispatchConfig.from_file(path)

    assert config.strategy == "heuristic"
    assert config.low_model == "gpt-5"


def test_config_file_rejects_unknown_keys(tmp_path) -> None:
    """A typo in a deployed config must fail loudly, not be ignored."""
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"stratergy": "heuristic"}))

    with pytest.raises(ValueError, match="unknown config keys"):
        DispatchConfig.from_file(path)


def test_env_can_point_at_a_config_file(tmp_path) -> None:
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"strategy": "always_high"}))

    config = DispatchConfig.from_env({"DMS_DISPATCH_CONFIG": str(path)})

    assert config.strategy == "always_high"


# ----------------------------------------------------------------------- wire


def test_anthropic_parse_flattens_block_system() -> None:
    request = wire.parse_anthropic(
        {
            "system": [{"type": "text", "text": "you are terse"}],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 99,
        }
    )

    assert request.system == "you are terse"
    assert request.max_tokens == 99


def test_openai_parse_accepts_either_max_tokens_spelling() -> None:
    base = {"messages": [{"role": "user", "content": "hi"}]}

    assert wire.parse_openai({**base, "max_completion_tokens": 11}).max_tokens == 11
    assert wire.parse_openai({**base, "max_tokens": 22}).max_tokens == 22
    assert wire.parse_openai(base).max_tokens == 4096


def test_openai_parse_normalises_a_bare_stop_string() -> None:
    request = wire.parse_openai(
        {"messages": [{"role": "user", "content": "hi"}], "stop": "END"}
    )

    assert request.stop_sequences == ("END",)


def test_openai_parse_merges_multiple_system_messages() -> None:
    request = wire.parse_openai(
        {"messages": [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "hi"},
        ]}
    )

    assert request.system == "a\nb"
    assert len(request.messages) == 1


def _result(**kw) -> DispatchResult:
    defaults = dict(
        text="hello",
        model=LOW,
        why="because",
        legs=(
            Leg(model=LOW, role="answer", usage=UsageRecord(input_tokens=10, output_tokens=5),
                cost_usd=Decimal("0.001"), latency_ms=12.0),
        ),
        request_id="abc123",
        strategy="cascade",
    )
    return DispatchResult(**{**defaults, **kw})


def test_anthropic_render_has_the_required_shape() -> None:
    body = wire.render_anthropic(_result())

    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "hello"}]
    assert body["usage"]["output_tokens"] == 5
    assert body["dms_dispatch"]["why"] == "because"


def test_openai_render_has_the_required_shape() -> None:
    body = wire.render_openai(_result())

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["usage"]["total_tokens"] == 15


@pytest.mark.parametrize(
    "internal,openai_finish",
    [("end_turn", "stop"), ("max_tokens", "length"),
     ("tool_use", "tool_calls"), ("refusal", "content_filter")],
)
def test_stop_reasons_map_across_dialects(internal, openai_finish) -> None:
    body = wire.render_openai(_result(stop_reason=internal))

    assert body["choices"][0]["finish_reason"] == openai_finish


def test_an_unknown_stop_reason_degrades_to_end_turn() -> None:
    assert wire.render_anthropic(_result(stop_reason="weird"))["stop_reason"] == "end_turn"


def test_openai_stream_chunks_are_valid_sse() -> None:
    chunk = wire.openai_stream_chunk(LOW, "tok", "rid")

    assert chunk.startswith("data: ")
    assert chunk.endswith("\n\n")
    assert json.loads(chunk[6:])["choices"][0]["delta"]["content"] == "tok"


def test_openai_stream_terminates_with_done() -> None:
    assert wire.openai_stream_done(LOW, "rid").endswith("data: [DONE]\n\n")


def test_anthropic_stream_emits_the_full_event_sequence() -> None:
    events = list(wire.anthropic_stream_events(LOW, ["a", "b"], "rid"))
    names = [line.split("event: ")[1].split("\n")[0] for line in events]

    assert names == [
        "message_start", "content_block_start",
        "content_block_delta", "content_block_delta",
        "content_block_stop", "message_delta", "message_stop",
    ]


def test_error_body_shape() -> None:
    body = wire.error_body("nope", kind="not_found_error")

    assert body == {"type": "error", "error": {"type": "not_found_error", "message": "nope"}}


# ------------------------------------------------------------------ providers


def test_anthropic_payload_lifts_system_into_a_block() -> None:
    kwargs = AnthropicProvider()._kwargs(
        HIGH, Request(messages=({"role": "user", "content": "hi"},), system="terse")
    )

    assert kwargs["system"] == [{"type": "text", "text": "terse"}]
    assert kwargs["model"] == HIGH


@pytest.mark.parametrize(
    "model,rejects",
    [(HIGH, True), ("claude-sonnet-5", True), ("claude-fable-5", True), (LOW, False)],
)
def test_sampling_params_are_dropped_where_the_model_rejects_them(model, rejects) -> None:
    """temperature is a 400 on Opus 4.7+/Sonnet 5/Fable 5. Passing it through
    from a caller would turn a working request into an error."""
    assert _rejects_sampling(model) is rejects

    kwargs = AnthropicProvider()._kwargs(
        model, Request(messages=({"role": "user", "content": "hi"},), temperature=0.5)
    )

    assert ("temperature" in kwargs) is not rejects


def test_anthropic_provider_claims_only_claude_models() -> None:
    provider = AnthropicProvider()

    assert provider.handles(HIGH)
    assert not provider.handles("gpt-5")


def test_openai_provider_claims_gpt_and_codex() -> None:
    provider = OpenAIProvider()

    assert provider.handles("gpt-5.6-sol")
    assert provider.handles("codex-mini")
    assert not provider.handles(HIGH)


def test_openai_payload_uses_max_completion_tokens_and_hoists_system() -> None:
    payload = OpenAIProvider(api_key="k")._payload(
        "gpt-5",
        Request(
            messages=({"role": "user", "content": "hi"},),
            system="terse",
            max_tokens=64,
            stop_sequences=("STOP",),
        ),
    )

    assert payload["max_completion_tokens"] == 64
    assert payload["messages"][0] == {"role": "system", "content": "terse"}
    assert payload["stop"] == ["STOP"]


def test_missing_openai_key_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        _ = OpenAIProvider().api_key


def test_missing_anthropic_key_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        _ = AnthropicProvider().client


def test_openai_response_parsing(monkeypatch) -> None:
    """Parse a real Chat Completions body without touching the network."""
    provider = OpenAIProvider(api_key="k")
    body = {
        "model": "gpt-5.6-sol",
        "choices": [{"message": {"content": " hi "}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 7,
                  "prompt_tokens_details": {"cached_tokens": 60}},
    }
    monkeypatch.setattr(provider, "_post", lambda *a, **k: _FakeHTTP(body))

    completion = provider.complete("gpt-5.6-sol", Request(messages=()))

    assert completion.text == "hi"
    assert completion.usage.input_tokens == 40
    assert completion.usage.cache_read_input_tokens == 60
    assert completion.stop_reason == "stop"


def test_openai_content_filter_is_normalised_to_a_refusal(monkeypatch) -> None:
    """So the dispatcher's refusal handling covers both providers with one path."""
    provider = OpenAIProvider(api_key="k")
    body = {
        "choices": [{"message": {"content": None, "refusal": "no"},
                     "finish_reason": "content_filter"}],
        "usage": {},
    }
    monkeypatch.setattr(provider, "_post", lambda *a, **k: _FakeHTTP(body))

    completion = provider.complete("gpt-5", Request(messages=()))

    assert completion.refused
    assert completion.refusal_category == "content_filter"


def test_openai_stream_skips_keepalives_and_stops_on_done(monkeypatch) -> None:
    provider = OpenAIProvider(api_key="k")
    lines = [
        b": keepalive\n",
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n',
        b"data: {malformed\n",
        b'data: {"choices":[{"delta":{}}]}\n',
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n',
        b"data: [DONE]\n",
        b'data: {"choices":[{"delta":{"content":"never"}}]}\n',
    ]
    monkeypatch.setattr(provider, "_post", lambda *a, **k: _FakeHTTP(lines=lines))

    assert "".join(provider.stream("gpt-5", Request(messages=()))) == "ab"


class _FakeHTTP:
    """Stands in for the object urlopen returns."""

    def __init__(self, body: dict | None = None, lines: list[bytes] | None = None) -> None:
        self._body = body
        self._lines = lines or []

    def read(self):
        return json.dumps(self._body).encode()

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ------------------------------------------------------------ prompt caching


def test_a_large_system_prompt_gets_a_cache_breakpoint() -> None:
    """Agent clients resend a big stable system prompt every turn; uncached that
    dominates the bill more than model choice ever could."""
    from dms.pricing import PriceBook

    provider = AnthropicProvider(cache_system=True, book=PriceBook.load())
    kwargs = provider._kwargs(
        HIGH,
        Request(messages=({"role": "user", "content": "hi"},), system="x" * 40_000),
    )

    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_a_prefix_below_the_model_floor_gets_no_breakpoint() -> None:
    """Haiku 4.5 needs 4096 tokens. Marking a smaller prefix would be a silent
    no-op that still reads as 'caching enabled' in the code."""
    from dms.pricing import PriceBook

    provider = AnthropicProvider(cache_system=True, book=PriceBook.load())
    kwargs = provider._kwargs(
        LOW, Request(messages=({"role": "user", "content": "hi"},), system="short")
    )

    assert "cache_control" not in kwargs["system"][0]


def test_the_same_prefix_caches_on_opus_and_not_on_haiku() -> None:
    """The floor is not monotonic with price: 512 on Opus 5, 4096 on Haiku 4.5."""
    from dms.pricing import PriceBook

    provider = AnthropicProvider(cache_system=True, book=PriceBook.load())
    system = "x" * 8_000  # ~2k tokens: above Opus's floor, below Haiku's
    request = Request(messages=({"role": "user", "content": "hi"},), system=system)

    assert "cache_control" in provider._kwargs(HIGH, request)["system"][0]
    assert "cache_control" not in provider._kwargs(LOW, request)["system"][0]


def test_caching_is_off_unless_asked_for() -> None:
    kwargs = AnthropicProvider(cache_system=False)._kwargs(
        HIGH, Request(messages=({"role": "user", "content": "hi"},), system="x" * 40_000)
    )

    assert "cache_control" not in kwargs["system"][0]


def test_developer_role_folds_into_the_system_prompt() -> None:
    """Codex sends role 'developer'; Anthropic allows only user/assistant."""
    request = wire.parse_responses(
        {
            "instructions": "base",
            "input": [
                {"type": "message", "role": "developer", "content": "operator note"},
                {"type": "message", "role": "user", "content": "hello"},
            ],
        }
    )

    assert "operator note" in request.system
    assert [m["role"] for m in request.messages] == ["user"]


def test_openai_tool_definitions_translate_to_anthropic_shape() -> None:
    from dms.dispatch.providers import _anthropic_tools

    chat_style = {"type": "function", "function": {
        "name": "shell", "description": "run", "parameters": {"type": "object"}}}
    responses_style = {"type": "function", "name": "apply_patch",
                       "description": "patch", "parameters": {"type": "object"}}

    out = _anthropic_tools((chat_style, responses_style))

    assert [t["name"] for t in out] == ["shell", "apply_patch"]
    assert all("input_schema" in t for t in out)


def test_an_anthropic_tool_passes_through_untouched() -> None:
    from dms.dispatch.providers import _anthropic_tools

    native = {"name": "x", "description": "d", "input_schema": {"type": "object"}}

    assert _anthropic_tools((native,)) == [native]


# --------------------------------------------------- openrouter / verification


def test_openrouter_variant_points_at_openrouter_with_its_own_key_var() -> None:
    """Same adapter, different config -- so exercising it against OpenRouter is a
    genuine test of the OpenAI code path, not a parallel implementation."""
    from dms.dispatch.providers import openrouter_provider

    provider = openrouter_provider()

    assert provider.base_url == "https://openrouter.ai/api/v1"
    assert provider.key_env == "OPENROUTER_API_KEY"
    assert provider.handles("openai/gpt-5.6-sol")
    assert "HTTP-Referer" in provider.headers


def test_a_missing_key_names_the_variable_it_wants(monkeypatch) -> None:
    from dms.dispatch.providers import openrouter_provider

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        _ = openrouter_provider().api_key


@pytest.mark.parametrize(
    "namespaced,bare",
    [
        ("openai/gpt-5.6-sol", "gpt-5.6-sol"),
        ("codex-cli/gpt-5.6-sol", "gpt-5.6-sol"),
        ("openai/gpt-5.6", "gpt-5.6-sol"),      # via alias
    ],
)
def test_namespaced_ids_price_against_the_bare_model(namespaced, bare) -> None:
    """The transport differs; the per-token rates do not."""
    from dms.pricing import PriceBook

    assert PriceBook.load().resolve(namespaced) == bare
