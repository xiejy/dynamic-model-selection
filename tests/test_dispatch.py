"""The dispatcher and the proxy, exercised end to end with fake providers."""
import json
import threading
import urllib.request
from decimal import Decimal

import pytest

from dms.dispatch.affinity import SessionAffinity
from dms.dispatch.config import DispatchConfig
from dms.dispatch.core import Dispatcher
from dms.dispatch.providers import (
    Completion,
    ProviderError,
    ProviderRegistry,
    Request,
    _openai_usage,
    _to_openai_messages,
)
from dms.dispatch.server import build_server
from dms.usage import UsageRecord

LOW = "claude-haiku-4-5"
HIGH = "claude-opus-5"


class FakeProvider:
    """Scripted responses per model, and a record of every call made."""

    name = "fake"

    def __init__(self, script: dict[str, list[Completion]] | None = None) -> None:
        self.script = script or {}
        self.calls: list[tuple[str, Request]] = []

    def handles(self, model: str) -> bool:
        return True

    def complete(self, model: str, request: Request) -> Completion:
        self.calls.append((model, request))
        queue = self.script.get(model)
        if queue:
            return queue.pop(0)
        return Completion(
            text=f"{model} says ok",
            model=model,
            usage=UsageRecord(input_tokens=100, output_tokens=20),
        )

    def stream(self, model: str, request: Request, usage_sink=None):
        self.calls.append((model, request))
        yield from ("hel", "lo")
        # Real providers report usage on the terminal chunk; mirror that so the
        # billing path is actually exercised.
        if usage_sink is not None:
            usage_sink.append(UsageRecord(input_tokens=40, output_tokens=2))


def _ok(model: str, text: str = "fine", out: int = 20) -> Completion:
    return Completion(
        text=text, model=model, usage=UsageRecord(input_tokens=100, output_tokens=out)
    )


def _refusal(model: str) -> Completion:
    return Completion(
        text="", model=model, usage=UsageRecord(input_tokens=100, output_tokens=0),
        stop_reason="refusal", refusal_category="cyber",
    )


def _dispatcher(provider: FakeProvider, **cfg) -> Dispatcher:
    return Dispatcher(
        DispatchConfig(low_model=LOW, high_model=HIGH, **cfg),
        providers=ProviderRegistry((provider,)),
    )


def _req(text: str = "hello", **kw) -> Request:
    return Request(messages=({"role": "user", "content": text},), **kw)


# ------------------------------------------------------------------ cascade


def test_cascade_stops_at_the_low_model_when_the_verifier_accepts() -> None:
    provider = FakeProvider({LOW: [_ok(LOW, "42"), _ok(LOW, "yes")]})

    result = _dispatcher(provider).dispatch(_req())

    assert result.model == LOW
    assert result.text == "42"
    assert [leg.role for leg in result.legs] == ["answer", "verify"]
    assert not result.escalated


def test_cascade_escalates_when_the_verifier_rejects() -> None:
    provider = FakeProvider(
        {LOW: [_ok(LOW, "probably 41"), _ok(LOW, "no")], HIGH: [_ok(HIGH, "42")]}
    )

    result = _dispatcher(provider).dispatch(_req())

    assert result.model == HIGH
    assert result.text == "42"
    assert [leg.role for leg in result.legs] == ["answer", "verify", "escalation"]
    assert result.escalated


def test_cascade_charges_for_the_answer_it_threw_away() -> None:
    """The discarded cheap attempt and the verification are real spend."""
    provider = FakeProvider(
        {LOW: [_ok(LOW, "wrong"), _ok(LOW, "no")], HIGH: [_ok(HIGH, "right")]}
    )

    result = _dispatcher(provider).dispatch(_req())

    assert result.overhead_usd > 0
    assert result.overhead_usd < result.cost_usd


def test_an_unreadable_verdict_escalates_rather_than_accepting() -> None:
    """Fail toward quality: never ship an answer nobody vouched for."""
    provider = FakeProvider(
        {LOW: [_ok(LOW, "maybe"), _ok(LOW, "¯\\_(ツ)_/¯")], HIGH: [_ok(HIGH, "sure")]}
    )

    assert _dispatcher(provider).dispatch(_req()).escalated


def test_cascade_skips_verification_when_the_low_model_refuses() -> None:
    """No point paying to verify an empty response."""
    provider = FakeProvider({LOW: [_refusal(LOW)], HIGH: [_ok(HIGH, "answer")]})

    result = _dispatcher(provider).dispatch(_req())

    assert [leg.role for leg in result.legs] == ["answer", "escalation"]
    assert result.model == HIGH


def test_when_the_high_model_refuses_the_paid_for_cheap_answer_is_kept() -> None:
    """Measured on real traffic: Opus 5 refused a benign POSIX shell question
    that Haiku answered fine. Discarding the cheap answer there would return
    nothing to the caller for money already spent."""
    provider = FakeProvider(
        {LOW: [_ok(LOW, "127"), _ok(LOW, "no")], HIGH: [_refusal(HIGH)]}
    )

    result = _dispatcher(provider).dispatch(_req())

    assert result.text == "127"
    assert result.model == LOW
    assert "refused" in result.why


# ---------------------------------------------------------------- heuristic


def test_heuristic_sends_a_lookup_to_the_low_model() -> None:
    provider = FakeProvider()

    result = _dispatcher(provider, strategy="heuristic").dispatch(
        _req("Extract the port number from this string.")
    )

    assert result.model == LOW
    assert len(result.legs) == 1  # no verification call


def test_heuristic_sends_reasoning_to_the_high_model() -> None:
    provider = FakeProvider()

    result = _dispatcher(provider, strategy="heuristic").dispatch(
        _req("Explain the root cause of this deadlock between two mutexes.")
    )

    assert result.model == HIGH


def test_heuristic_spends_no_tokens_deciding() -> None:
    provider = FakeProvider()

    _dispatcher(provider, strategy="heuristic").dispatch(_req("anything"))

    assert len(provider.calls) == 1  # exactly the answering call


def test_routers_score_only_user_turns() -> None:
    """Keying on the assistant's own prior output would make the decision drift
    across a conversation."""
    request = Request(
        messages=(
            {"role": "user", "content": "extract the port"},
            {"role": "assistant", "content": "deadlock mutex complexity"},
        )
    )

    assert "deadlock" not in request.text_prompt


# ----------------------------------------------------------------- refusals


def test_a_refusal_on_a_fixed_strategy_retries_the_other_tier() -> None:
    provider = FakeProvider({LOW: [_refusal(LOW)], HIGH: [_ok(HIGH, "recovered")]})

    result = _dispatcher(provider, strategy="always_low").dispatch(_req())

    assert result.text == "recovered"
    assert result.model == HIGH
    assert [leg.role for leg in result.legs] == ["answer", "retry"]


def test_refusal_retry_can_be_turned_off() -> None:
    provider = FakeProvider({LOW: [_refusal(LOW)]})

    result = _dispatcher(
        provider, strategy="always_low", retry_other_tier_on_refusal=False
    ).dispatch(_req())

    assert result.stop_reason == "refusal"
    assert len(result.legs) == 1


# ---------------------------------------------------------------- affinity


def test_session_affinity_pins_the_first_choice() -> None:
    """A model switch invalidates the prompt cache with no escape hatch, so a
    reroute can cost more than the tier gap it saves."""
    provider = FakeProvider()
    dispatcher = _dispatcher(provider, strategy="heuristic")

    first = dispatcher.dispatch(_req("Extract the port number."), session_id="s1")
    second = dispatcher.dispatch(
        _req("Explain this deadlock root cause in detail."), session_id="s1"
    )

    assert first.model == second.model == LOW
    assert "affinity" in second.why


def test_different_sessions_are_pinned_independently() -> None:
    provider = FakeProvider()
    dispatcher = _dispatcher(provider, strategy="heuristic")

    a = dispatcher.dispatch(_req("Extract the port number."), session_id="a")
    b = dispatcher.dispatch(
        _req("Explain the root cause of this deadlock."), session_id="b"
    )

    assert a.model == LOW
    assert b.model == HIGH


def test_affinity_expires() -> None:
    # set() consumes one tick (expires_at = 0 + 10); get() consumes the next.
    ticks = iter([0.0, 9_999.0])
    affinity = SessionAffinity(ttl_seconds=10, clock=lambda: next(ticks))
    affinity.set("s", HIGH, "pinned")

    assert affinity.get("s") is None
    assert len(affinity) == 0  # the dead pin is dropped on read


def test_affinity_can_be_disabled() -> None:
    provider = FakeProvider()
    dispatcher = _dispatcher(provider, strategy="heuristic", session_affinity=False)

    dispatcher.dispatch(_req("Extract the port number."), session_id="s")
    second = dispatcher.dispatch(
        _req("Explain the root cause of this deadlock."), session_id="s"
    )

    assert second.model == HIGH  # re-decided, not pinned


def test_purge_drops_expired_pins() -> None:
    ticks = iter([0.0, 5_000.0, 5_000.0])
    affinity = SessionAffinity(ttl_seconds=1, clock=lambda: next(ticks))
    affinity.set("s", LOW, "x")

    assert affinity.purge_expired() == 1
    assert len(affinity) == 0


# ------------------------------------------------------------------ streaming


def test_streaming_never_uses_the_cascade() -> None:
    """The verifier needs the complete cheap answer, so a streamed cascade would
    buffer the whole response and destroy time-to-first-token."""
    with pytest.raises(ValueError, match="cannot serve a streaming request"):
        DispatchConfig(streaming_strategy="cascade")


def test_stream_returns_the_chosen_model_and_tokens() -> None:
    provider = FakeProvider()
    dispatcher = _dispatcher(provider, strategy="cascade")

    model, tokens, sink = dispatcher.stream(_req("Extract the port.", stream=True))

    assert model == LOW
    assert "".join(tokens) == "hello"
    assert sink, "the provider must report usage once the stream drains"


# ----------------------------------------------------------------- accounting


def test_every_leg_is_costed_and_totalled() -> None:
    provider = FakeProvider(
        {LOW: [_ok(LOW, "x"), _ok(LOW, "no")], HIGH: [_ok(HIGH, "y")]}
    )
    dispatcher = _dispatcher(provider)

    result = dispatcher.dispatch(_req())

    assert len(result.legs) == 3
    assert all(leg.cost_usd > 0 for leg in result.legs)
    assert result.cost_usd == sum((leg.cost_usd for leg in result.legs), Decimal(0))
    assert dispatcher.total_cost_usd == result.cost_usd


def test_log_record_names_the_model_and_the_reason() -> None:
    provider = FakeProvider()

    record = _dispatcher(provider, strategy="heuristic").dispatch(_req()).to_log()

    assert record["model"] in {LOW, HIGH}
    assert record["why"]
    assert record["total_cost_usd"]
    assert len(record["legs"]) == 1


# ------------------------------------------------------------------ providers


def test_openai_usage_does_not_double_count_cached_tokens() -> None:
    """OpenAI's prompt_tokens INCLUDES cached tokens; Anthropic's input_tokens
    excludes them. Failing to subtract bills the cached portion twice."""
    usage = _openai_usage(
        {"prompt_tokens": 1000, "completion_tokens": 50,
         "prompt_tokens_details": {"cached_tokens": 800}}
    )

    assert usage.input_tokens == 200
    assert usage.cache_read_input_tokens == 800
    assert usage.prompt_tokens == 1000


def test_block_content_flattens_for_openai() -> None:
    out = _to_openai_messages(
        ({"role": "user", "content": [{"type": "text", "text": "hi"}]},)
    )

    assert out == [{"role": "user", "content": "hi"}]


def test_registry_rejects_an_unknown_model() -> None:
    with pytest.raises(ProviderError, match="no provider handles"):
        ProviderRegistry().for_model("llama-3")


def test_registry_routes_by_model_prefix() -> None:
    registry = ProviderRegistry()

    assert registry.for_model("claude-opus-5").name == "anthropic"
    assert registry.for_model("gpt-5.6-sol").name == "openai"


# --------------------------------------------------------------------- proxy


@pytest.fixture
def proxy():
    provider = FakeProvider()
    dispatcher = Dispatcher(
        DispatchConfig(low_model=LOW, high_model=HIGH, strategy="heuristic"),
        providers=ProviderRegistry((provider,)),
    )
    server = build_server("127.0.0.1", 0, dispatcher=dispatcher)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}", dispatcher, provider
    server.shutdown()
    server.server_close()


def _post(url: str, body: dict, headers: dict | None = None) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def test_proxy_speaks_the_anthropic_messages_api(proxy) -> None:
    base, _, _ = proxy

    body = _post(
        f"{base}/v1/messages",
        {"model": "claude-opus-5", "max_tokens": 64,
         "messages": [{"role": "user", "content": "Extract the port number."}]},
    )

    assert body["type"] == "message"
    assert body["content"][0]["text"]
    assert body["usage"]["input_tokens"] > 0
    assert body["dms_dispatch"]["model"] == LOW  # dispatched, not obeyed


def test_proxy_speaks_the_openai_chat_api(proxy) -> None:
    """This is the endpoint Codex CLI and the OpenAI SDKs point at."""
    base, _, _ = proxy

    body = _post(
        f"{base}/v1/chat/completions",
        {"model": "gpt-5.6-sol",
         "messages": [
             {"role": "system", "content": "be terse"},
             {"role": "user", "content": "Extract the port number."},
         ]},
    )

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] > 0


def test_openai_system_message_is_lifted_out(proxy) -> None:
    base, _, provider = proxy

    _post(
        f"{base}/v1/chat/completions",
        {"model": "x", "messages": [
            {"role": "system", "content": "SYSTEM_MARKER"},
            {"role": "user", "content": "hi"},
        ]},
    )

    _, request = provider.calls[-1]
    assert request.system == "SYSTEM_MARKER"
    assert all(m["role"] != "system" for m in request.messages)


def test_proxy_reports_the_dispatch_decision(proxy) -> None:
    """Without this a caller cannot tell which tier answered or what it cost."""
    base, _, _ = proxy

    meta = _post(
        f"{base}/v1/messages",
        {"model": "auto", "max_tokens": 32,
         "messages": [{"role": "user", "content": "hi"}]},
    )["dms_dispatch"]

    assert meta["strategy"] == "heuristic"
    assert meta["why"]
    assert Decimal(meta["cost_usd"]) > 0
    assert meta["legs"]


def test_proxy_honours_the_session_header(proxy) -> None:
    base, dispatcher, _ = proxy

    _post(f"{base}/v1/messages",
          {"model": "x", "max_tokens": 16,
           "messages": [{"role": "user", "content": "Extract the port number."}]},
          {"X-Session-Id": "sess-1"})
    second = _post(f"{base}/v1/messages",
                   {"model": "x", "max_tokens": 16,
                    "messages": [{"role": "user", "content": "Explain this deadlock."}]},
                   {"X-Session-Id": "sess-1"})

    assert "affinity" in second["dms_dispatch"]["why"]
    assert len(dispatcher.affinity) == 1


def test_healthz_and_stats(proxy) -> None:
    base, _, _ = proxy
    _post(f"{base}/v1/messages",
          {"model": "x", "max_tokens": 16,
           "messages": [{"role": "user", "content": "hi"}]})

    with urllib.request.urlopen(f"{base}/healthz", timeout=5) as r:
        assert json.load(r)["status"] == "ok"
    with urllib.request.urlopen(f"{base}/stats", timeout=5) as r:
        stats = json.load(r)

    assert stats["requests_served"] == 1
    assert Decimal(stats["total_cost_usd"]) > 0


def test_bad_json_is_a_400_not_a_traceback(proxy) -> None:
    base, _, _ = proxy
    req = urllib.request.Request(
        f"{base}/v1/messages", data=b"{not json",
        headers={"Content-Type": "application/json"},
    )

    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)

    assert exc.value.code == 400


def test_empty_messages_is_rejected(proxy) -> None:
    base, _, _ = proxy
    req = urllib.request.Request(
        f"{base}/v1/messages", data=json.dumps({"messages": []}).encode(),
        headers={"Content-Type": "application/json"},
    )

    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)

    assert exc.value.code == 400


def test_unknown_route_is_a_404(proxy) -> None:
    base, _, _ = proxy

    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{base}/v1/nope", timeout=5)

    assert exc.value.code == 404


# ------------------------------------------------------- proxy streaming/errors


def _raw_post(url: str, body: dict, headers: dict | None = None):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    return urllib.request.urlopen(req, timeout=10)


def test_proxy_streams_openai_sse(proxy) -> None:
    base, _, _ = proxy

    with _raw_post(
        f"{base}/v1/chat/completions",
        {"model": "x", "stream": True,
         "messages": [{"role": "user", "content": "hi"}]},
    ) as response:
        payload = response.read().decode()
        model_header = response.headers.get("X-DMS-Model")

    assert model_header == LOW
    assert payload.count("data: ") >= 3
    assert payload.rstrip().endswith("data: [DONE]")


def test_proxy_streams_anthropic_sse(proxy) -> None:
    base, _, _ = proxy

    with _raw_post(
        f"{base}/v1/messages",
        {"model": "x", "max_tokens": 16, "stream": True,
         "messages": [{"role": "user", "content": "hi"}]},
    ) as response:
        payload = response.read().decode()

    for event in ("message_start", "content_block_delta", "message_stop"):
        assert f"event: {event}" in payload


def test_streaming_uses_the_pre_request_strategy_not_the_cascade(proxy) -> None:
    """A streamed cascade would have to buffer the whole answer before the
    verifier could judge it, so it must not be used here."""
    base, _, provider = proxy

    with _raw_post(
        f"{base}/v1/chat/completions",
        {"model": "x", "stream": True,
         "messages": [{"role": "user", "content": "Extract the port."}]},
    ) as response:
        response.read()

    # Exactly one upstream call: no verification round trip.
    assert len(provider.calls) == 1


def test_a_provider_failure_becomes_a_502_not_a_500() -> None:
    """A caller must be able to tell an upstream outage from a proxy bug."""
    class BrokenProvider(FakeProvider):
        def complete(self, model, request):
            raise ProviderError("upstream is down")

    dispatcher = Dispatcher(
        DispatchConfig(low_model=LOW, high_model=HIGH, strategy="always_low"),
        providers=ProviderRegistry((BrokenProvider(),)),
    )
    server = build_server("127.0.0.1", 0, dispatcher=dispatcher)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(f"{base}/v1/messages",
                  {"model": "x", "max_tokens": 8,
                   "messages": [{"role": "user", "content": "hi"}]})
        assert exc.value.code == 502
        assert "upstream is down" in exc.value.read().decode()
    finally:
        server.shutdown()
        server.server_close()


def test_missing_body_is_rejected(proxy) -> None:
    base, _, _ = proxy
    req = urllib.request.Request(
        f"{base}/v1/messages", data=b"", method="POST",
        headers={"Content-Type": "application/json", "Content-Length": "0"},
    )

    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)

    assert exc.value.code == 400


def test_unknown_post_route_names_the_valid_ones(proxy) -> None:
    base, _, _ = proxy

    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{base}/v1/completions", {"messages": [{"role": "user", "content": "x"}]})

    assert exc.value.code == 404
    assert "/v1/chat/completions" in exc.value.read().decode()


def test_streamed_requests_are_counted_and_billed(proxy) -> None:
    """Usage arrives only on the terminal chunk. Before this was wired up the
    proxy reported 0 requests and $0 spend for an entirely streaming client --
    which, for an agent like Codex, is all of its traffic."""
    base, dispatcher, _ = proxy

    with _raw_post(
        f"{base}/v1/chat/completions",
        {"model": "x", "stream": True,
         "messages": [{"role": "user", "content": "hi"}]},
    ) as response:
        response.read()

    assert dispatcher.requests_served == 1
    assert dispatcher.total_cost_usd > 0
