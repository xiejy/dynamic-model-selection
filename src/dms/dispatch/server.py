"""The proxy: an HTTP service that dispatches between a low and a high model.

Point any Anthropic or OpenAI client at it by changing `base_url` alone.

    POST /v1/messages          Anthropic Messages   (Claude SDKs, Claude Code)
    POST /v1/chat/completions  OpenAI Chat          (Codex CLI, OpenAI SDKs)
    GET  /healthz              liveness
    GET  /stats                requests served, spend, live session pins

Built on stdlib `ThreadingHTTPServer`: no new dependencies, a thread per
request, and streaming works because the provider calls are synchronous. That is
sufficient for an internal service and a demo. For high concurrency, swap this
module for an ASGI app -- everything below `Dispatcher` is transport-agnostic
and would carry over unchanged.
"""
from __future__ import annotations

import json
import logging
import os
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dms.dispatch.config import DispatchConfig
from dms.dispatch.core import DispatchResult, Dispatcher
from dms.dispatch.providers import ProviderError
from dms.dispatch import wire

MAX_BODY_BYTES = 32 * 1024 * 1024  # matches the Messages API request ceiling
SESSION_HEADERS = ("x-session-id", "x-dms-session", "anthropic-session-id")

log = logging.getLogger("dms.proxy")


class DispatchHTTPRequestHandler(BaseHTTPRequestHandler):
    """One request. `dispatcher` is attached to the server, not the handler."""

    protocol_version = "HTTP/1.1"
    server_version = "dms-dispatch/0.1"

    # ----------------------------------------------------------------- routing

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.startswith("/healthz"):
            self._json(200, {"status": "ok", "strategy": self._config.strategy})
        elif self.path.startswith("/stats"):
            self._json(200, self._stats())
        elif self.path.split("?", 1)[0].rstrip("/") == "/v1/models":
            # Codex CLI probes this on startup and logs an error if it 404s.
            self._json(200, self._models())
        else:
            self._json(404, wire.error_body(f"no route for GET {self.path}", kind="not_found_error"))

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/v1/messages":
            self._handle(parse=wire.parse_anthropic, render=wire.render_anthropic, dialect="anthropic")
        elif path == "/v1/chat/completions":
            self._handle(parse=wire.parse_openai, render=wire.render_openai, dialect="openai")
        elif path == "/v1/responses":
            # Codex CLI 0.146+ dropped `wire_api = "chat"` and speaks only this.
            self._handle(
                parse=wire.parse_responses, render=wire.render_responses, dialect="responses"
            )
        else:
            self._json(
                404,
                wire.error_body(
                    f"no route for POST {path}; try /v1/messages, "
                    "/v1/chat/completions or /v1/responses",
                    kind="not_found_error",
                ),
            )

    # ---------------------------------------------------------------- handling

    def _handle(self, *, parse, render, dialect: str) -> None:
        payload = self._body()
        if payload is None:
            return

        try:
            request = parse(payload)
        except (KeyError, TypeError, ValueError) as exc:
            self._json(400, wire.error_body(f"could not parse request: {exc}"))
            return

        if not request.messages:
            self._json(400, wire.error_body("messages must not be empty"))
            return

        session_id = self._session_id()

        try:
            if request.stream:
                self._stream(request, session_id, dialect)
            else:
                result = self._dispatcher.dispatch(request, session_id=session_id)
                self._log(result)
                self._json(200, render(result))
        except ProviderError as exc:
            # Surface the upstream status so a caller can distinguish a bad
            # request from a rate limit from an outage.
            status = exc.status if exc.status and 400 <= exc.status < 600 else 502
            self._json(status, wire.error_body(str(exc), kind="api_error"))
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to a client
            log.exception("dispatch failed")
            self._json(500, wire.error_body(f"dispatch failed: {exc}", kind="api_error"))

    def _stream(self, request, session_id: str | None, dialect: str) -> None:
        request_id = wire.new_request_id()
        model, deltas, usage_sink = self._dispatcher.stream(
            request, session_id=session_id
        )

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-DMS-Model", model)
        self.send_header("X-DMS-Strategy", self._config.streaming_strategy)
        self.send_header("Connection", "close")
        self.end_headers()

        def write(chunk: str) -> None:
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

        if dialect == "anthropic":
            for event in wire.anthropic_stream_events(model, deltas, request_id):
                write(event)
        elif dialect == "responses":
            for event in wire.responses_stream_events(model, deltas, request_id, []):
                write(event)
        else:
            for delta in deltas:
                write(wire.openai_stream_chunk(model, delta, request_id))
            write(wire.openai_stream_done(model, request_id))

        # Usage arrives only on the terminal chunk, so billing happens here --
        # after the stream has drained, not before it starts.
        leg = self._dispatcher.bill_stream(
            model, usage_sink, why=f"streamed via {self._config.streaming_strategy}"
        )
        log.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "session_id": session_id,
                    "strategy": self._config.streaming_strategy,
                    "model": model,
                    "streamed": True,
                    "escalated": False,
                    "why": f"streamed via {self._config.streaming_strategy}",
                    "total_cost_usd": str(leg.cost_usd) if leg else "0",
                    "usage": leg.usage.to_dict() if leg else None,
                }
            )
        )

    # ----------------------------------------------------------------- helpers

    @property
    def _dispatcher(self) -> Dispatcher:
        return self.server.dispatcher  # type: ignore[attr-defined]

    @property
    def _config(self) -> DispatchConfig:
        return self._dispatcher.config

    def _session_id(self) -> str | None:
        for header in SESSION_HEADERS:
            if value := self.headers.get(header):
                return value
        return None

    def _body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json(400, wire.error_body("invalid Content-Length"))
            return None
        if length <= 0:
            self._json(400, wire.error_body("request body is required"))
            return None
        if length > MAX_BODY_BYTES:
            self._json(413, wire.error_body("request body too large"))
            return None
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            self._json(400, wire.error_body(f"body is not valid JSON: {exc}"))
            return None

    def _models(self) -> dict[str, Any]:
        """The two tiers this proxy dispatches between, plus the alias.

        A caller naming any of these gets dispatched, not obeyed -- the model
        field is a hint. Listing them keeps model-probing clients happy.
        """
        cfg = self._config
        ids = [cfg.low_model, cfg.high_model, "auto"]
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "created": 0, "owned_by": "dms"}
                for model in dict.fromkeys(ids)
            ],
        }

    def _stats(self) -> dict[str, Any]:
        d = self._dispatcher
        return {
            "requests_served": d.requests_served,
            "total_cost_usd": str(d.total_cost_usd),
            "live_session_pins": len(d.affinity),
            "config": {
                "strategy": d.config.strategy,
                "streaming_strategy": d.config.streaming_strategy,
                "low_model": d.config.low_model,
                "high_model": d.config.high_model,
                "session_affinity": d.config.session_affinity,
            },
        }

    def _json(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _log(self, result: DispatchResult) -> None:
        log.info(json.dumps(result.to_log()))

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the default stderr access log; we emit structured lines."""
        return


def build_server(
    host: str = "127.0.0.1",
    port: int = 8787,
    config: DispatchConfig | None = None,
    dispatcher: Dispatcher | None = None,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), DispatchHTTPRequestHandler)
    server.daemon_threads = True
    server.dispatcher = dispatcher or Dispatcher(config or DispatchConfig.from_env())  # type: ignore[attr-defined]
    return server


def serve(host: str = "127.0.0.1", port: int = 8787, config: DispatchConfig | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("DMS_LOG_LEVEL", "INFO"),
        format="%(message)s",
    )
    server = build_server(host, port, config)
    cfg = server.dispatcher.config  # type: ignore[attr-defined]

    print(f"dms proxy listening on http://{host}:{port}")
    print(f"  strategy      {cfg.strategy}  (streaming: {cfg.streaming_strategy})")
    print(f"  low  model    {cfg.low_model}")
    print(f"  high model    {cfg.high_model}")
    print(f"  affinity      {'on' if cfg.session_affinity else 'off'}"
          f" ({cfg.affinity_ttl_seconds}s)")
    print("  endpoints     POST /v1/messages  POST /v1/chat/completions"
          "  POST /v1/responses")
    print("                GET /healthz  GET /stats")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.shutdown()
        server.server_close()
    return 0
