"""Pass-through proxy: choose the model, forward everything else untouched.

    client (codex / claude) ─▶ this proxy ─▶ the router it already uses ─▶ provider

Only one field of one kind of request is ever changed: the `model` of a
generation request addressed to the main model. Tools, tool results, the login
header, streaming and every other path pass through byte for byte, so tool
calling works and no API key is needed -- the user's own router authenticates.

The response is streamed back unchanged while a sniffer reads its token usage,
and each request is appended to the usage ledger.

Security: the login header passes through this process in memory, between two
localhost sockets. It is never logged; the ledger stores no headers and no
prompt text. The proxy binds only to loopback, ignores any HTTP proxy the
environment or the OS configures, and relays redirects instead of following
them -- each of which would otherwise let the header leave localhost.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from dms.passthrough.ledger import Ledger, UsageEvent, _as_dict, summarise
from dms.passthrough.select import PassDecision, PassthroughSelector, Tier, prompt_bytes
from dms.passthrough.usage import UsageSniffer
from dms.pricing import PriceBook

UPSTREAM_TIMEOUT_SECONDS = 600
MAX_BODY_BYTES = 64 * 1024 * 1024
STATS_PATH = "/_dms/stats"
CHUNK = 16 * 1024
ERROR_LOG_BYTES = 400

# Most specific first. Codex sends session-id and thread-id, and a sub-agent
# shares its parent's session-id; Claude Code sends x-claude-code-session-id on
# every request and marks sub-agents with x-claude-code-agent-id. x-session-id
# is for callers that set one themselves.
SESSION_HEADERS = ("thread-id", "session-id", "x-claude-code-session-id", "x-session-id")
AGENT_HEADER = "x-claude-code-agent-id"

# The generation endpoint itself, however the client's base URL is spelled
# (/v1/messages, /v1/v1/messages). Not count_tokens, not /responses/compact.
GENERATION_ENDPOINT = {"openai": "/responses", "anthropic": "/messages"}

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailer", "transfer-encoding", "upgrade", "host", "content-length",
    "accept-encoding",
}

# Token-shaped strings, removed from anything logged: claude-router's own 401
# text is built from exception messages.
_SECRET = re.compile(r"(?i)(bearer\s+)\S+|sk-[\w-]{8,}|[\w+/=.-]{32,}")

log = logging.getLogger("dms.passthrough")


class _RelayRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # following would re-send the login header to the new location


# Never the default opener: it honours http_proxy and the macOS system proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RelayRedirects)


def session_key(headers: Mapping[str, str]) -> str | None:
    """The most specific conversation id the client sent, or None."""
    lowered = {k.lower(): v for k, v in headers.items()}
    session = next((lowered[h] for h in SESSION_HEADERS if lowered.get(h)), None)
    agent = lowered.get(AGENT_HEADER)
    if agent:
        return f"{session}/{agent}" if session else agent
    return session


def _redact(text: str) -> str:
    return _SECRET.sub(lambda m: (m.group(1) or "") + "[redacted]", text)


def _is_loopback(host: str) -> bool:
    """IPv4 loopback only: the server socket is AF_INET."""
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.version == 4 and address.is_loopback


class PassthroughHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "dms-passthrough/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return  # no access log: it would record paths with query strings

    # ------------------------------------------------------------------ routes

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == STATS_PATH:
            self._send_json(200, self._stats())
            return
        self._forward(body=None)

    def do_POST(self) -> None:
        body = self._read_body()
        if body is not None:
            self._forward(body=body)

    do_PUT = do_PATCH = do_DELETE = do_POST

    def do_HEAD(self) -> None:
        self._forward(body=None)

    def _read_body(self) -> bytes | None:
        """The request body, or None after replying with an error."""
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            return self._read_chunked()
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            return self._reject(400, "invalid Content-Length")
        if length > MAX_BODY_BYTES:
            return self._reject(413, "request body too large")
        return self.rfile.read(length) if length else b""

    def _reject(self, status: int, message: str) -> None:
        """Refuse a body without reading it -- and close, or its unread bytes
        would be parsed as the next request on this connection."""
        self.close_connection = True
        self._send_json(status, {"error": {"message": message}}, close=True)

    def _read_chunked(self) -> bytes | None:
        body = bytearray()
        while True:
            try:
                size = int(self.rfile.readline(1024).split(b";", 1)[0].strip(), 16)
            except ValueError:
                size = -1
            if size < 0:
                return self._reject(400, "malformed chunked body")
            if size == 0:
                while self.rfile.readline(1024).strip():  # trailers, until the blank line
                    pass
                return bytes(body)
            if len(body) + size > MAX_BODY_BYTES:
                return self._reject(413, "request body too large")
            body.extend(self.rfile.read(size))
            self.rfile.readline(1024)  # the CRLF closing the chunk

    # --------------------------------------------------------------- forwarding

    def _forward(self, body: bytes | None) -> None:
        cfg = self.server.cfg  # type: ignore[attr-defined]
        started = time.perf_counter()
        decision, out_body = self._maybe_route(body)

        request = urllib.request.Request(
            cfg["upstream"] + self.path,
            data=out_body,
            method=self.command,
            headers=self._upstream_headers(),
        )
        try:
            response = _OPENER.open(request, timeout=UPSTREAM_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as err:
            response = err  # an upstream 3xx/4xx/5xx is a response to relay, not a failure
        except (urllib.error.URLError, OSError) as err:
            self._send_json(502, {"error": {"message": f"upstream unreachable: {err}"}})
            self._record(decision, 502, None, started)
            return

        sniffer = UsageSniffer(cfg["family"]) if decision else None
        status = response.status if hasattr(response, "status") else response.code
        recorded = False

        def finish() -> None:
            # Called before the last bytes reach the client: once a client has a
            # complete response it may read the ledger, which must already agree.
            nonlocal recorded
            if not recorded:
                recorded = True
                self._record(decision, status, sniffer.result() if sniffer else None, started)

        try:
            self._relay(response, status, sniffer, finish)
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-response (Ctrl-C, a killed `codex exec`).
            # Routine; the tokens were still spent, so it is still recorded.
            self.close_connection = True
            log.info("client disconnected before the response finished")
        finally:
            response.close()
            finish()  # a relay cut short is still recorded

    def _maybe_route(self, body: bytes | None) -> tuple[PassDecision | None, bytes | None]:
        """Decide and rewrite a generation request; leave everything else alone."""
        cfg = self.server.cfg  # type: ignore[attr-defined]
        path = self.path.split("?", 1)[0].rstrip("/")
        is_generation = self.command == "POST" and path.endswith(GENERATION_ENDPOINT[cfg["family"]])
        if not is_generation or not body:
            return None, body
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, body
        if not isinstance(payload, dict) or "model" not in payload:
            return None, body

        decision = cfg["selector"].decide(payload, session_key(self.headers),
                                          body_bytes=prompt_bytes(body))
        if decision.tier is Tier.UNTOUCHED or decision.model == payload["model"]:
            return decision, body  # unchanged: forward the original bytes exactly
        # ASCII escapes keep an unpaired surrogate the client sent encodable.
        return decision, json.dumps({**payload, "model": decision.model}).encode("ascii")

    def _relay(self, response: Any, status: int, sniffer: UsageSniffer | None,
               finish) -> None:
        headers = response.headers
        streaming = "text/event-stream" in (headers.get("Content-Type") or "")

        self.send_response(status)
        for name, value in headers.items():
            if name.lower() not in HOP_BY_HOP:
                self.send_header(name, value)

        if self.command == "HEAD":
            if length := headers.get("Content-Length"):
                self.send_header("Content-Length", length)
            self.end_headers()
            finish()
            return

        if not streaming:
            data = response.read()
            if sniffer:
                sniffer.feed(data)
            if status >= 400:
                # Operator diagnostics only -- never the ledger. Provider error
                # bodies name the rejected parameter, e.g. a feature the smaller
                # model lacks, which is what an operator needs to see.
                log.warning("upstream %s: %s", status,
                            _redact(data[:ERROR_LOG_BYTES].decode("utf-8", "replace")))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            finish()
            self.wfile.write(data)
            return

        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        while chunk := response.read1(CHUNK) if hasattr(response, "read1") else response.read(CHUNK):
            if sniffer:
                sniffer.feed(chunk)
            self.wfile.write(chunk)
            self.wfile.flush()
        finish()  # the connection closes only after this handler returns

    def _upstream_headers(self) -> dict[str, str]:
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["Accept-Encoding"] = "identity"  # a compressed stream cannot be sniffed
        return headers

    # ------------------------------------------------------------------ ledger

    def _record(self, decision: PassDecision | None, status: int, usage, started: float) -> None:
        if decision is None:
            return
        cfg = self.server.cfg  # type: ignore[attr-defined]
        event = UsageEvent(
            ts=datetime.now(UTC).isoformat(timespec="seconds"),
            family=cfg["family"],
            session=session_key(self.headers),
            requested_model=cfg["selector"].high if decision.tier is not Tier.UNTOUCHED
            else decision.model,
            routed_model=decision.model,
            tier=str(decision.tier),
            turn=str(decision.turn),
            why=decision.why,
            status=status,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            usage=usage,
        )
        try:
            cfg["ledger"].record(event)
        except OSError:
            # Accounting must never cost the client its response.
            log.exception("usage ledger write to %s failed; response unaffected",
                          cfg["ledger"].path)
        with cfg["lock"]:
            cfg["events"].append(event)
        log.info("%s %s -> %s (%s) %s", event.turn, event.tier, event.routed_model,
                 event.status, event.why)

    def _stats(self) -> dict[str, Any]:
        cfg = self.server.cfg  # type: ignore[attr-defined]
        with cfg["lock"]:
            events = list(cfg["events"])
        summary = summarise(events, cfg["book"])
        return {**_as_dict(summary), "since": cfg["started"], "ledger": str(cfg["ledger"].path)}

    def _send_json(self, status: int, body: dict[str, Any], *, close: bool = False) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if close:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)


def build_passthrough_server(
    host: str,
    port: int,
    *,
    family: str,
    upstream: str,
    selector: PassthroughSelector,
    ledger: Ledger,
    book: PriceBook | None = None,
) -> ThreadingHTTPServer:
    if family not in GENERATION_ENDPOINT:
        raise ValueError(f"family must be one of {sorted(GENERATION_ENDPOINT)}")
    if not _is_loopback(host):
        raise ValueError(
            f"host must be a loopback address (127.0.0.1 or localhost), got {host!r}: "
            "the router behind this proxy spends your login for anyone who can reach it"
        )
    parts = urlsplit(upstream)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"upstream must be an origin like http://127.0.0.1:18790, got {upstream!r}")

    from threading import Lock

    server = ThreadingHTTPServer((host, port), PassthroughHandler)
    server.daemon_threads = True
    server.cfg = {  # type: ignore[attr-defined]
        "family": family,
        # Paths are forwarded as the client sent them, so only the origin is kept.
        "upstream": f"{parts.scheme}://{parts.netloc}",
        "selector": selector,
        "ledger": ledger,
        "book": book or PriceBook.load(),
        "events": [],
        "lock": Lock(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return server
