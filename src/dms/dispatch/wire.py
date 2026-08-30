"""Wire formats: parse two ingress dialects, emit the matching response.

The proxy accepts both so callers change only `base_url`:

  POST /v1/messages          Anthropic Messages    (Claude SDKs, Claude Code)
  POST /v1/chat/completions  OpenAI Chat           (Codex CLI, OpenAI SDKs)

The `model` a caller names is treated as a *hint*, not an instruction -- that is
the whole point of a dispatcher. A caller naming an explicit tier alias
("low"/"high") or a model the proxy knows still gets dispatched, and the response
reports which model actually answered so nothing has to guess.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from dms.dispatch.core import DispatchResult
from dms.dispatch.providers import Request

# Callers can force a tier by naming it as the model.
TIER_ALIASES = {"low", "high", "auto", "cascade", "heuristic"}

# Roles that mean "operator instruction" rather than a conversation turn.
# Anthropic accepts only user/assistant in `messages`, so these fold into system.
_SYSTEM_ROLES = {"system", "developer"}


def parse_anthropic(payload: dict[str, Any]) -> Request:
    """POST /v1/messages -> internal Request."""
    system = payload.get("system")
    if isinstance(system, list):
        system = "\n".join(
            block.get("text", "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )

    return Request(
        messages=tuple(payload.get("messages") or ()),
        system=system or None,
        tools=tuple(payload["tools"]) if payload.get("tools") else None,
        max_tokens=int(payload.get("max_tokens") or 4096),
        temperature=payload.get("temperature"),
        stop_sequences=tuple(payload.get("stop_sequences") or ()),
        stream=bool(payload.get("stream")),
    )


def parse_openai(payload: dict[str, Any]) -> Request:
    """POST /v1/chat/completions -> internal Request.

    OpenAI carries the system prompt as the first message rather than a separate
    field, so it is lifted out here to give the dispatcher one shape to reason
    about.
    """
    messages = list(payload.get("messages") or ())
    system_parts = [
        m.get("content", "") for m in messages if m.get("role") in _SYSTEM_ROLES
    ]
    rest = tuple(m for m in messages if m.get("role") not in _SYSTEM_ROLES)

    # OpenAI renamed max_tokens -> max_completion_tokens; accept either.
    max_tokens = (
        payload.get("max_completion_tokens") or payload.get("max_tokens") or 4096
    )
    stop = payload.get("stop")
    if isinstance(stop, str):
        stop = [stop]

    return Request(
        messages=rest,
        system="\n".join(p for p in system_parts if p) or None,
        tools=tuple(payload["tools"]) if payload.get("tools") else None,
        max_tokens=int(max_tokens),
        temperature=payload.get("temperature"),
        stop_sequences=tuple(stop or ()),
        stream=bool(payload.get("stream")),
    )


def render_anthropic(result: DispatchResult) -> dict[str, Any]:
    """Internal result -> a Messages API response body."""
    usage = result.usage
    return {
        "id": f"msg_{result.request_id}",
        "type": "message",
        "role": "assistant",
        "model": result.model,
        "content": [{"type": "text", "text": result.text}],
        "stop_reason": _anthropic_stop(result.stop_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
        },
        # Non-standard, and deliberately so: without it a caller cannot tell
        # which tier answered or what the request cost.
        "dms_dispatch": _dispatch_meta(result),
    }


def render_openai(result: DispatchResult) -> dict[str, Any]:
    """Internal result -> a Chat Completions response body."""
    usage = result.usage
    return {
        "id": f"chatcmpl-{result.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": _openai_finish(result.stop_reason),
            }
        ],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "prompt_tokens_details": {"cached_tokens": usage.cache_read_input_tokens},
        },
        "dms_dispatch": _dispatch_meta(result),
    }


def openai_stream_chunk(model: str, delta: str, request_id: str) -> str:
    body = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
    }
    return f"data: {json.dumps(body)}\n\n"


def openai_stream_done(model: str, request_id: str) -> str:
    body = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n"


def anthropic_stream_events(model: str, deltas, request_id: str):
    """Yield the Messages API SSE sequence for a streamed response."""
    message_id = f"msg_{request_id}"
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    for delta in deltas:
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": delta},
            },
        )
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 0},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


def error_body(message: str, *, kind: str = "invalid_request_error") -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


# ------------------------------------------------------------------------ helpers


def _dispatch_meta(result: DispatchResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "strategy": result.strategy,
        "model": result.model,
        "why": result.why,
        "escalated": result.escalated,
        "cost_usd": str(result.cost_usd),
        "overhead_usd": str(result.overhead_usd),
        "legs": [
            {"model": leg.model, "role": leg.role, "cost_usd": str(leg.cost_usd)}
            for leg in result.legs
        ],
    }


def _anthropic_stop(reason: str) -> str:
    return reason if reason in {
        "end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal", "pause_turn"
    } else "end_turn"


def _openai_finish(reason: str) -> str:
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "refusal": "content_filter",
    }.get(reason, "stop")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------- responses API

# Codex CLI 0.146+ removed `wire_api = "chat"` and speaks only the Responses
# API, so a proxy that offers Chat Completions alone cannot serve it:
#   Error loading config.toml: `wire_api = "chat"` is no longer supported.
# https://github.com/openai/codex/discussions/7782


def parse_responses(payload: dict[str, Any]) -> Request:
    """POST /v1/responses -> internal Request.

    `input` is either a bare string or a list of items; an item's content is
    either a string or a list of typed parts (`input_text`, `output_text`).
    `instructions` is the system prompt.
    """
    messages: list[dict[str, Any]] = []
    system_parts: list[str] = []
    if instructions := payload.get("instructions"):
        system_parts.append(instructions)

    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    else:
        for item in raw_input or ():
            if not isinstance(item, dict):
                continue
            # Non-message items (function_call, reasoning, ...) carry no prompt
            # text for a router to score, so they are skipped rather than
            # stringified into noise.
            if item.get("type") not in (None, "message"):
                continue
            role = item.get("role", "user")
            text = _responses_text(item.get("content"))
            # Codex sends role "developer" -- the Responses API's name for
            # operator instructions. Anthropic allows only user/assistant, so
            # anything else folds into the system prompt rather than 400ing.
            if role in _SYSTEM_ROLES:
                system_parts.append(text)
            else:
                messages.append({"role": role, "content": text})

    max_tokens = payload.get("max_output_tokens") or payload.get("max_tokens") or 4096
    return Request(
        messages=tuple(m for m in messages if m["content"]),
        system="\n\n".join(p for p in system_parts if p) or None,
        tools=tuple(payload["tools"]) if payload.get("tools") else None,
        max_tokens=int(max_tokens),
        temperature=payload.get("temperature"),
        stream=bool(payload.get("stream")),
    )


def _responses_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
            and part.get("type") in ("input_text", "output_text", "text")
        )
    return ""


def render_responses(result: DispatchResult) -> dict[str, Any]:
    """Internal result -> a Responses API body."""
    usage = result.usage
    return {
        "id": f"resp_{result.request_id}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": result.model,
        "output": [
            {
                "type": "message",
                "id": f"msg_{result.request_id}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": result.text, "annotations": []}
                ],
            }
        ],
        "output_text": result.text,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "input_tokens_details": {"cached_tokens": usage.cache_read_input_tokens},
            "output_tokens": usage.output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.total_tokens,
        },
        "dms_dispatch": _dispatch_meta(result),
    }


def responses_stream_events(model: str, deltas, request_id: str, result_text_box: list):
    """Yield the Responses API SSE sequence.

    `result_text_box` collects the streamed text so the caller can log what was
    actually sent without buffering it twice.
    """
    response_id = f"resp_{request_id}"
    item_id = f"msg_{request_id}"

    def shell(status: str, text: str = "") -> dict[str, Any]:
        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": status,
            "model": model,
            "output": (
                [
                    {
                        "type": "message",
                        "id": item_id,
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": text, "annotations": []}
                        ],
                    }
                ]
                if status == "completed"
                else []
            ),
        }

    yield _sse("response.created", {"type": "response.created", "response": shell("in_progress")})
    yield _sse(
        "response.in_progress",
        {"type": "response.in_progress", "response": shell("in_progress")},
    )
    yield _sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message", "id": item_id, "status": "in_progress",
                "role": "assistant", "content": [],
            },
        },
    )
    yield _sse(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": item_id, "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )

    for delta in deltas:
        result_text_box.append(delta)
        yield _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": item_id, "output_index": 0, "content_index": 0,
                "delta": delta,
            },
        )

    text = "".join(result_text_box)
    yield _sse(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": item_id, "output_index": 0, "content_index": 0, "text": text,
        },
    )
    yield _sse(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": item_id, "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": text, "annotations": []},
        },
    )
    yield _sse(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message", "id": item_id, "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        },
    )
    yield _sse(
        "response.completed",
        {"type": "response.completed", "response": shell("completed", text)},
    )
