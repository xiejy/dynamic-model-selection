"""The pass-through proxy end to end, against a fake upstream router."""
import http.server
import json
import threading
import urllib.error
import urllib.request

import pytest

from dms.passthrough.ledger import Ledger
from dms.passthrough.select import PassthroughSelector
from dms.passthrough.server import build_passthrough_server

HIGH, LOW = "gpt-5.6-sol@personal", "gpt-5.6-luna@personal"
EASY = "Extract the port number from this string."
HARD = "Explain the root cause of this deadlock between two mutexes, step by step."

RESPONSES_SSE = (
    b'event: response.created\ndata: {"type":"response.created"}\n\n'
    b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"ok"}\n\n'
    b'event: response.completed\ndata: {"type":"response.completed","response":{"usage":'
    b'{"input_tokens":1000,"input_tokens_details":{"cached_tokens":600},"output_tokens":20}}}\n\n'
)


class Upstream:
    """A fake router: records every request, replies with a scripted response."""

    def __init__(self) -> None:
        self.received: list[dict] = []
        self.reply = (200, "text/event-stream", RESPONSES_SSE)
        self.extra_headers: dict[str, str] = {}
        upstream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _serve(self):
                n = int(self.headers.get("Content-Length") or 0)
                upstream.received.append({
                    "method": self.command, "path": self.path,
                    "headers": dict(self.headers), "body": self.rfile.read(n),
                })
                status, ctype, body = upstream.reply
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for name, value in upstream.extra_headers.items():
                    self.send_header(name, value)
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the proxy hung up, as it should when its client does

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = _serve

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.origin = f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def rig(tmp_path):
    upstream = Upstream()
    ledger = Ledger(tmp_path / "usage.jsonl")
    selector = PassthroughSelector(family="openai", low=LOW, high=HIGH)
    proxy = build_passthrough_server(
        "127.0.0.1", 0, family="openai", upstream=upstream.origin,
        selector=selector, ledger=ledger,
    )
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{proxy.server_address[1]}"
    yield base, upstream, ledger
    proxy.shutdown()
    upstream.server.shutdown()


def _codex_body(*items, model=HIGH) -> dict:
    return {"model": model, "stream": True, "input": list(items),
            "tools": [{"type": "custom", "name": "exec", "description": "run a command"}],
            "tool_choice": "auto", "prompt_cache_key": "sess-1"}


def _user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _post(base, body, headers=None, path="/v1/responses"):
    raw = json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=raw, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer chatgpt-token-XYZ",
                 "session-id": "sess-1", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.read(), raw


# ------------------------------------------------------------ what is forwarded


def test_a_low_decision_rewrites_only_the_model(rig) -> None:
    base, upstream, _ = rig
    _post(base, _codex_body(_user(EASY)))

    sent = json.loads(upstream.received[0]["body"])
    original = _codex_body(_user(EASY))
    assert sent["model"] == LOW
    assert {k: v for k, v in sent.items() if k != "model"} == \
           {k: v for k, v in original.items() if k != "model"}


def test_tools_reach_the_upstream_untouched(rig) -> None:
    """The whole point of pass-through: tool calling works because the proxy
    never translates it."""
    base, upstream, _ = rig
    _post(base, _codex_body(_user(EASY)))

    sent = json.loads(upstream.received[0]["body"])
    assert sent["tools"] == _codex_body()["tools"]
    assert sent["tool_choice"] == "auto"


def test_the_login_header_is_forwarded(rig) -> None:
    """Auth stays with the user's own router; the proxy just relays it."""
    base, upstream, _ = rig
    _post(base, _codex_body(_user(EASY)))

    headers = {k.lower(): v for k, v in upstream.received[0]["headers"].items()}
    assert headers["authorization"] == "Bearer chatgpt-token-XYZ"
    assert headers["session-id"] == "sess-1"


def test_a_request_that_stays_high_is_forwarded_byte_for_byte(rig) -> None:
    base, upstream, _ = rig
    _, _, raw = _post(base, _codex_body(_user(HARD)))

    assert upstream.received[0]["body"] == raw


def test_other_paths_are_forwarded_untouched(rig) -> None:
    """Codex probes /v1/models on start; that must reach the router."""
    base, upstream, _ = rig
    upstream.reply = (200, "application/json", b'{"object":"list","data":[]}')

    with urllib.request.urlopen(base + "/v1/models?client_version=0.153", timeout=5) as r:
        assert json.load(r)["object"] == "list"
    assert upstream.received[0]["path"] == "/v1/models?client_version=0.153"


# ------------------------------------------------------------ what comes back


def test_the_stream_reaches_the_client_unchanged(rig) -> None:
    base, _, _ = rig

    _, body, _ = _post(base, _codex_body(_user(EASY)))

    assert body == RESPONSES_SSE


def test_an_upstream_error_is_passed_through_with_its_status(rig) -> None:
    base, upstream, ledger = rig
    upstream.reply = (400, "application/json", b'{"error":{"message":"bad effort"}}')

    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(base, _codex_body(_user(EASY)))

    assert exc.value.code == 400
    assert b"bad effort" in exc.value.read()
    assert ledger.read()[0].status == 400


def test_an_unreachable_upstream_is_a_502(tmp_path) -> None:
    proxy = build_passthrough_server(
        "127.0.0.1", 0, family="openai", upstream="http://127.0.0.1:1",
        selector=PassthroughSelector(family="openai", low=LOW, high=HIGH),
        ledger=Ledger(tmp_path / "u.jsonl"),
    )
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{proxy.server_address[1]}"

    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(base, _codex_body(_user(EASY)))

    assert exc.value.code == 502
    proxy.shutdown()


# ------------------------------------------------------------------ the ledger


def test_usage_is_recorded_with_its_tier(rig) -> None:
    base, _, ledger = rig
    _post(base, _codex_body(_user(EASY)))
    _post(base, _codex_body(_user(HARD)))

    events = ledger.read()
    assert [e.tier for e in events] == ["low", "high"]
    assert events[0].routed_model == LOW
    assert events[0].usage.input_tokens == 400
    assert events[0].usage.cache_read_input_tokens == 600
    assert events[0].session == "sess-1"


def test_tool_steps_ride_on_the_session_header(rig) -> None:
    base, _, ledger = rig
    _post(base, _codex_body(_user(EASY)))
    _post(base, _codex_body(_user(EASY), {"type": "function_call", "name": "exec"},
                            {"type": "function_call_output", "output": HARD}))

    step = ledger.read()[1]
    assert (step.turn, step.tier) == ("continuation", "low")


def test_the_ledger_holds_no_credentials_or_prompt_text(rig) -> None:
    base, _, ledger = rig
    _post(base, _codex_body(_user(EASY)))

    raw = ledger.path.read_text()
    assert "chatgpt-token-XYZ" not in raw
    assert "port number" not in raw


def test_stats_summarise_this_run(rig) -> None:
    base, _, _ = rig
    _post(base, _codex_body(_user(EASY)))
    _post(base, _codex_body(_user(HARD)))

    with urllib.request.urlopen(base + "/_dms/stats", timeout=5) as r:
        stats = json.load(r)

    assert (stats["requests"], stats["low"], stats["high"]) == (2, 1, 1)
    assert float(stats["all_high_usd"]) > float(stats["actual_usd"])


# ------------------------------------------------------------- review findings


def _raw(base: str, method: str, path: str, body: bytes | None = None,
         headers: dict | None = None, chunked: bool = False) -> tuple[int, dict, bytes]:
    """http.client, so the test client itself never follows a redirect."""
    import http.client
    from urllib.parse import urlsplit

    parts = urlsplit(base)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    conn.request(method, path, body=body, headers=headers or {}, encode_chunked=chunked)
    resp = conn.getresponse()
    out = resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    conn.close()
    return out


def test_a_system_or_env_proxy_is_never_used(rig, monkeypatch) -> None:
    """The login header and the prompt must not leave localhost through a proxy
    the environment happens to configure."""
    base, upstream, _ = rig
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    urllib.request.install_opener(None)  # a default opener built now would read them
    monkeypatch.setattr(urllib.request, "_opener", None)

    status, _, _ = _raw(base, "POST", "/v1/responses", json.dumps(_codex_body(_user(EASY))).encode(),
                        {"Content-Type": "application/json"})

    assert status == 200
    assert len(upstream.received) == 1


@pytest.mark.parametrize("method,code", [("POST", 302), ("GET", 307)])
def test_an_upstream_redirect_is_relayed_not_followed(rig, method, code) -> None:
    """Following it would re-send the login header to wherever it points."""
    base, upstream, _ = rig
    elsewhere = Upstream()
    upstream.reply = (code, "text/plain", b"")
    upstream.extra_headers = {"Location": elsewhere.origin + "/steal"}
    body = json.dumps(_codex_body(_user(EASY))).encode() if method == "POST" else None

    status, headers, _ = _raw(base, method, "/v1/responses", body,
                              {"Authorization": "Bearer chatgpt-token-XYZ"})

    assert status == code
    assert headers["location"].endswith("/steal")
    assert elsewhere.received == []
    elsewhere.server.shutdown()


def test_a_failing_ledger_never_truncates_the_response(tmp_path) -> None:
    """Accounting is secondary: a full disk or unwritable log must not cost the
    client its response -- above all an upstream error it needs to read."""
    upstream = Upstream()
    upstream.reply = (400, "application/json", b'{"error":{"message":"bad effort"}}')
    proxy = build_passthrough_server(
        "127.0.0.1", 0, family="openai", upstream=upstream.origin,
        selector=PassthroughSelector(family="openai", low=LOW, high=HIGH),
        ledger=Ledger(tmp_path),  # a directory: every write fails
    )
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{proxy.server_address[1]}"

    status, _, body = _raw(base, "POST", "/v1/responses", json.dumps(_codex_body(_user(EASY))).encode())

    assert status == 400
    assert body == b'{"error":{"message":"bad effort"}}'
    proxy.shutdown()
    upstream.server.shutdown()


def test_credentials_in_an_upstream_error_are_not_logged(rig, caplog) -> None:
    """claude-router's 401 carries its own exception text; never let a token
    reach the operator log."""
    base, upstream, _ = rig
    upstream.reply = (401, "application/json", (
        b'{"error":{"message":"bad credential sk-fake-0123456789abcdefXYZ '
        b'or Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJlLXZhbHVl"}}'))

    with caplog.at_level("WARNING", logger="dms.passthrough"):
        status, _, _ = _raw(base, "POST", "/v1/responses", json.dumps(_codex_body(_user(EASY))).encode())

    assert status == 401
    assert "upstream 401" in caplog.text
    assert "sk-fake-0123" not in caplog.text
    assert "eyJhbGciOiJIUzI1NiJ9" not in caplog.text


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "::", "example.com"])
def test_only_loopback_addresses_can_be_bound(tmp_path, host) -> None:
    """Behind it sits the user's logged-in router: on a network address, anyone
    who can reach the port spends the user's subscription."""
    with pytest.raises(ValueError, match="loopback"):
        build_passthrough_server(
            host, 0, family="openai", upstream="http://127.0.0.1:1",
            selector=PassthroughSelector(family="openai", low=LOW, high=HIGH),
            ledger=Ledger(tmp_path / "u.jsonl"),
        )


def test_a_negative_content_length_is_rejected(rig) -> None:
    import socket

    base, upstream, _ = rig
    port = int(base.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(b"POST /v1/responses HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n"
                     b"GET /smuggled HTTP/1.1\r\nHost: x\r\n\r\n")
        reply = b""
        while chunk := sock.recv(4096):  # the proxy must close after refusing
            reply += chunk

    assert reply.startswith(b"HTTP/1.1 400")
    assert reply.count(b"HTTP/1.1") == 1
    assert upstream.received == []


def test_a_chunked_request_body_is_forwarded_whole(rig) -> None:
    base, upstream, _ = rig
    body = json.dumps(_codex_body(_user(EASY))).encode()

    status, _, _ = _raw(base, "POST", "/v1/responses", iter([body[:40], body[40:]]),
                        {"Content-Type": "application/json"}, chunked=True)

    assert status == 200
    assert json.loads(upstream.received[0]["body"])["model"] == LOW


def test_a_body_without_a_model_is_forwarded_unchanged(rig) -> None:
    base, upstream, _ = rig
    raw = json.dumps({"input": [_user(EASY)]}).encode()

    _raw(base, "POST", "/v1/responses", raw, {"Content-Type": "application/json"})

    assert upstream.received[0]["body"] == raw


def test_a_lone_surrogate_in_a_rewritten_body_does_not_crash(rig) -> None:
    """JSON allows an unpaired \\ud800 escape; it must survive the rewrite."""
    base, upstream, _ = rig
    raw = json.dumps(_codex_body(_user(EASY + " \ud800"))).encode()

    status, _, _ = _raw(base, "POST", "/v1/responses", raw, {"Content-Type": "application/json"})

    assert status == 200
    sent = json.loads(upstream.received[0]["body"])
    assert sent["model"] == LOW and sent["input"] == json.loads(raw)["input"]


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_other_methods_are_forwarded(rig, method) -> None:
    base, upstream, _ = rig
    upstream.reply = (200, "application/json", b"{}")

    status, _, _ = _raw(base, method, "/v1/files/f1", b"{}", {"Content-Type": "application/json"})

    assert status == 200
    assert upstream.received[0]["method"] == method


def test_head_is_forwarded_with_the_upstream_length(rig) -> None:
    base, upstream, _ = rig
    upstream.reply = (200, "application/json", b'{"object":"list"}')

    status, headers, body = _raw(base, "HEAD", "/v1/models")

    assert status == 200 and body == b""
    assert headers["content-length"] == str(len(b'{"object":"list"}'))
    assert upstream.received[0]["method"] == "HEAD"


@pytest.mark.parametrize("path", ["/v1/responses/compact", "/v1/messages/count_tokens"])
def test_only_the_generation_endpoint_itself_is_routed(tmp_path, path) -> None:
    """A compaction or token count is not a user's message to score."""
    upstream = Upstream()
    upstream.reply = (200, "application/json", b"{}")
    family = "openai" if "responses" in path else "anthropic"
    high, low = (HIGH, LOW) if family == "openai" else ("claude-opus-5", "claude-sonnet-5")
    proxy = build_passthrough_server(
        "127.0.0.1", 0, family=family, upstream=upstream.origin,
        selector=PassthroughSelector(family=family, low=low, high=high),
        ledger=Ledger(tmp_path / "u.jsonl"),
    )
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{proxy.server_address[1]}"
    raw = json.dumps({"model": high, "input": [_user(EASY)],
                      "messages": [{"role": "user", "content": EASY}]}).encode()

    _raw(base, "POST", path, raw, {"Content-Type": "application/json"})

    assert upstream.received[0]["body"] == raw
    proxy.shutdown()
    upstream.server.shutdown()


def test_a_base_url_ending_in_v1_is_still_routed(tmp_path) -> None:
    """Claude Code with ANTHROPIC_BASE_URL=.../v1 requests /v1/v1/messages --
    seen 503 times in the claude-router log."""
    upstream = Upstream()
    upstream.reply = (200, "application/json", b"{}")
    proxy = build_passthrough_server(
        "127.0.0.1", 0, family="anthropic", upstream=upstream.origin,
        selector=PassthroughSelector(family="anthropic", low="claude-sonnet-5",
                                     high="claude-opus-5"),
        ledger=Ledger(tmp_path / "u.jsonl"),
    )
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{proxy.server_address[1]}"

    _raw(base, "POST", "/v1/v1/messages",
         json.dumps({"model": "claude-opus-5", "messages": [{"role": "user", "content": EASY}]}).encode())

    assert json.loads(upstream.received[0]["body"])["model"] == "claude-sonnet-5"
    proxy.shutdown()
    upstream.server.shutdown()


# ------------------------------------------------------------------ session key


def test_codex_sub_agents_are_told_apart_by_thread() -> None:
    """A Codex sub-agent shares its parent's session-id; only thread-id differs."""
    from dms.passthrough.server import session_key

    parent = session_key({"session-id": "S", "thread-id": "S"})
    child = session_key({"session-id": "S", "thread-id": "T2"})

    assert parent != child


def test_claude_code_sub_agents_are_told_apart_by_agent_id() -> None:
    """Claude Code sends one session id on every request; sub-agents add an agent id."""
    from dms.passthrough.server import session_key

    main = session_key({"x-claude-code-session-id": "S"})
    agent = session_key({"x-claude-code-session-id": "S", "x-claude-code-agent-id": "a1"})

    assert main == "S" and agent != main


def test_no_session_header_gives_no_session() -> None:
    from dms.passthrough.server import session_key

    assert session_key({}) is None


def test_a_client_hanging_up_mid_stream_is_recorded_quietly(rig, capsys) -> None:
    """Ctrl-C in Codex, or a killed `codex exec`, closes the socket while the
    stream is still flowing. That is routine, not a crash -- and the request
    still cost tokens, so it is still recorded."""
    import socket
    import time

    base, upstream, ledger = rig
    big = b'event: response.output_text.delta\ndata: {"type":"x","delta":"' + b"z" * 8000 + b'"}\n\n'
    upstream.reply = (200, "text/event-stream", big * 2000)
    port = int(base.rsplit(":", 1)[1])
    raw = json.dumps(_codex_body(_user(EASY))).encode()

    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(b"POST /v1/responses HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     + f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
        sock.recv(1024)  # the stream has started
    deadline = time.monotonic() + 10
    while not ledger.read() and time.monotonic() < deadline:
        time.sleep(0.05)

    events = ledger.read()
    assert len(events) == 1 and events[0].usage is None
    assert "Traceback" not in capsys.readouterr().err


def test_screenshots_do_not_push_a_message_off_the_low_model(tmp_path) -> None:
    """The live bug, end to end: megabytes of screenshots are not ~770k tokens."""
    upstream = Upstream()
    proxy = build_passthrough_server(
        "127.0.0.1", 0, family="openai", upstream=upstream.origin,
        selector=PassthroughSelector(family="openai", low=LOW, high=HIGH,
                                     low_context_tokens=400_000),
        ledger=Ledger(tmp_path / "u.jsonl"),
    )
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{proxy.server_address[1]}"
    shot = "data:image/png;base64,iVBORw0KGgo" + "A" * 120 * 1024
    images = {"type": "message", "role": "user",
              "content": [{"type": "input_image", "image_url": shot} for _ in range(19)]}

    _post(base, _codex_body(images, _user(EASY)))

    assert json.loads(upstream.received[0]["body"])["model"] == LOW
    proxy.shutdown()
    upstream.server.shutdown()
