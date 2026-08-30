"""Provider adapters: one internal request shape, several wire formats.

Anthropic goes through the official SDK. OpenAI-compatible endpoints go through
stdlib `urllib` rather than the `openai` package -- it keeps the dependency list
at one, matches the zero-dependency client pattern already in ~/openrouter, and
means the same adapter serves OpenAI, OpenRouter, Azure, or a local server by
changing `base_url` alone.

The internal `Request`/`Completion` pair is deliberately provider-neutral so the
dispatcher never has to know which vendor answered.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from dms.usage import UsageRecord

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 600


class ProviderError(RuntimeError):
    """A provider call failed. Carries the upstream status when there was one."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class Request:
    """A completion request, independent of any vendor's wire format."""

    messages: tuple[dict[str, Any], ...]
    system: str | None = None
    tools: tuple[dict[str, Any], ...] | None = None
    max_tokens: int = 4096
    temperature: float | None = None
    stop_sequences: tuple[str, ...] = ()
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def text_prompt(self) -> str:
        """Flattened user text, for routers that score the prompt.

        Only user turns: a router must not key on the assistant's own prior
        output, which would make the decision drift over a conversation.
        """
        parts: list[str] = []
        for message in self.messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        return "\n".join(part for part in parts if part)

    def with_(self, **changes: Any) -> Request:
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class Completion:
    """A completed response, independent of any vendor's wire format."""

    text: str
    model: str
    usage: UsageRecord
    stop_reason: str = "end_turn"
    refusal_category: str = ""
    latency_ms: float = 0.0
    raw: Any = None

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def empty(self) -> bool:
        """Refusals and some filtered responses return HTTP 200 with no text."""
        return not self.text.strip()


class Provider(Protocol):
    name: str

    def handles(self, model: str) -> bool: ...

    def complete(self, model: str, request: Request) -> Completion: ...

    def stream(
        self, model: str, request: Request, usage_sink: list[UsageRecord] | None = None
    ) -> Iterator[str]: ...


# --------------------------------------------------------------------- anthropic


class AnthropicProvider:
    """Claude models via the official SDK."""

    name = "anthropic"

    def __init__(
        self,
        client: Any | None = None,
        *,
        cache_system: bool = False,
        cache_ttl: str = "5m",
        book: Any | None = None,
    ) -> None:
        self._client = client
        self.cache_system = cache_system
        self.cache_ttl = cache_ttl
        self._book = book

    def handles(self, model: str) -> bool:
        return model.startswith("claude-")

    @property
    def client(self) -> Any:
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ProviderError(
                    "ANTHROPIC_API_KEY is not set; cannot call Claude models"
                )
            from anthropic import Anthropic

            self._client = Anthropic()
        return self._client

    def _kwargs(self, model: str, request: Request) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            # Last line of defence: an unexpected role is a 400 from the API and
            # a mid-stream disconnect for the caller.
            "messages": _anthropic_roles(request.messages),
        }
        if request.system:
            block: dict[str, Any] = {"type": "text", "text": request.system}
            if self.cache_system and self._caches(model, request.system):
                block["cache_control"] = (
                    {"type": "ephemeral"}
                    if self.cache_ttl == "5m"
                    else {"type": "ephemeral", "ttl": self.cache_ttl}
                )
            kwargs["system"] = [block]
        if request.tools:
            kwargs["tools"] = _anthropic_tools(request.tools)
        if request.stop_sequences:
            kwargs["stop_sequences"] = list(request.stop_sequences)
        # temperature is rejected on Opus 4.7+/Sonnet 5/Fable 5; pass it only
        # where the caller asked and the model still accepts it.
        if request.temperature is not None and not _rejects_sampling(model):
            kwargs["temperature"] = request.temperature
        kwargs.update(request.extra.get("anthropic", {}))
        return kwargs

    def complete(self, model: str, request: Request) -> Completion:
        started = time.perf_counter()
        try:
            response = self.client.messages.create(**self._kwargs(model, request))
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
            raise ProviderError(f"anthropic call failed: {exc}") from exc
        latency = (time.perf_counter() - started) * 1000

        # stop_details is populated ONLY on a refusal; guard before reading it.
        details = getattr(response, "stop_details", None)
        return Completion(
            text="".join(
                block.text
                for block in getattr(response, "content", [])
                if getattr(block, "type", None) == "text"
            ).strip(),
            model=getattr(response, "model", model),
            usage=UsageRecord.from_response(response),
            stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            refusal_category=(getattr(details, "category", "") or "") if details else "",
            latency_ms=latency,
            raw=response,
        )

    def stream(
        self, model: str, request: Request, usage_sink: list[UsageRecord] | None = None
    ) -> Iterator[str]:
        """Yield text deltas, then append the final usage to `usage_sink`.

        A streamed response reports usage only at the end. Without this the
        proxy silently bills nothing for streaming traffic -- which, for an
        agent client like Codex, is all of it.
        """
        kwargs = self._kwargs(model, request)
        with self.client.messages.stream(**kwargs) as stream:
            yield from stream.text_stream
            if usage_sink is not None:
                usage_sink.append(UsageRecord.from_response(stream.get_final_message()))


    def _caches(self, model: str, system: str) -> bool:
        """Whether this prefix clears the model's minimum cacheable size.

        Below the floor the API does not error -- it silently writes nothing and
        every request keeps paying full price. Haiku 4.5's floor is 4096 tokens,
        8x Opus 5's 512, so the cheap model is the harder one to cache.
        """
        if self._book is None:
            return True
        # ~3.6 chars/token is only for the floor check; real counts come from
        # response.usage. Bias low so a borderline prefix is not wrongly marked.
        estimated = len(system) / 4.0
        try:
            return estimated >= self._book.min_cache_prefix_tokens(model)
        except KeyError:
            return True


def _anthropic_roles(messages: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Coerce every turn to a role the Messages API accepts."""
    return [
        {**m, "role": "user" if m.get("role") != "assistant" else "assistant"}
        for m in messages
    ]


def _anthropic_tools(tools: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Translate OpenAI tool definitions into Anthropic's shape.

    Chat sends `{"type":"function","function":{name,description,parameters}}`;
    the Responses API flattens it to `{"type":"function",name,description,
    parameters}`. Anthropic wants `{name,description,input_schema}`. A tool
    already in Anthropic shape passes through untouched.
    """
    out: list[dict[str, Any]] = []
    for tool in tools:
        if "input_schema" in tool:
            out.append(tool)
            continue
        spec = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = spec.get("name")
        if not name:
            continue  # nothing usable; dropping beats a 400
        out.append(
            {
                "name": name,
                "description": spec.get("description", ""),
                "input_schema": spec.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return out


def _rejects_sampling(model: str) -> bool:
    """Models that 400 on temperature/top_p/top_k."""
    return any(
        marker in model
        for marker in ("opus-5", "opus-4-7", "opus-4-8", "sonnet-5", "fable-5", "mythos-5")
    )


# ------------------------------------------------------------------------ openai


class OpenAIProvider:
    """GPT / Codex models over the OpenAI-compatible Chat Completions API.

    stdlib only. Point `base_url` elsewhere to reach OpenRouter, Azure, or a
    local server with the same code.
    """

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = OPENAI_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        prefixes: tuple[str, ...] = ("gpt-", "o1", "o3", "o4", "codex", "openai/"),
        key_env: str = "OPENAI_API_KEY",
        headers: dict[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.prefixes = prefixes
        # OpenRouter and Azure read the key from a different variable and want
        # their own attribution headers; both are config, not code.
        self.key_env = key_env
        self.headers = headers or {}

    def handles(self, model: str) -> bool:
        return model.startswith(self.prefixes)

    @property
    def api_key(self) -> str:
        key = self._api_key or os.environ.get(self.key_env)
        if not key:
            raise ProviderError(
                f"{self.key_env} is not set; cannot call models at {self.base_url}"
            )
        return key

    def _payload(self, model: str, request: Request) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(_to_openai_messages(request.messages))

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop_sequences:
            payload["stop"] = list(request.stop_sequences)
        if request.tools:
            payload["tools"] = list(request.tools)
        payload.update(request.extra.get("openai", {}))
        return payload

    def _post(self, payload: dict[str, Any], *, stream: bool = False) -> Any:
        body = json.dumps({**payload, "stream": stream}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.headers,
            },
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise ProviderError(
                f"openai HTTP {exc.code}: {detail}", status=exc.code
            ) from None
        except urllib.error.URLError as exc:
            raise ProviderError(f"openai connection failed: {exc.reason}") from None

    def complete(self, model: str, request: Request) -> Completion:
        started = time.perf_counter()
        with self._post(self._payload(model, request)) as response:
            data = json.load(response)
        latency = (time.perf_counter() - started) * 1000

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason") or "stop"

        # An OpenAI content-filter stop is the same situation as an Anthropic
        # refusal: HTTP 200, no usable text. Normalise it so the dispatcher's
        # refusal handling covers both providers with one code path.
        refusal_text = message.get("refusal")
        stop_reason = "refusal" if (finish == "content_filter" or refusal_text) else finish

        return Completion(
            text=(message.get("content") or "").strip(),
            model=data.get("model", model),
            usage=_openai_usage(data.get("usage") or {}),
            stop_reason=stop_reason,
            refusal_category="content_filter" if stop_reason == "refusal" else "",
            latency_ms=latency,
            raw=data,
        )

    def stream(
        self, model: str, request: Request, usage_sink: list[UsageRecord] | None = None
    ) -> Iterator[str]:
        payload = self._payload(model, request)
        # Ask for usage on the terminal chunk; without it a streamed request
        # cannot be costed. Servers that ignore the option simply omit it.
        payload.setdefault("stream_options", {"include_usage": True})
        with self._post(payload, stream=True) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    return
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if usage_sink is not None and parsed.get("usage"):
                    usage_sink.append(_openai_usage(parsed["usage"]))
                try:
                    delta = parsed["choices"][0]["delta"].get("content")
                except (KeyError, IndexError):
                    continue
                if delta:
                    yield delta


def _openai_usage(usage: dict[str, Any]) -> UsageRecord:
    """Map OpenAI's usage shape onto the internal one.

    `prompt_tokens` INCLUDES cached tokens, unlike Anthropic's `input_tokens`
    which is the uncached remainder. Subtract, or cached tokens get billed twice.
    """
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    return UsageRecord(
        input_tokens=max(0, prompt - cached),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cache_read_input_tokens=cached,
    )


def _to_openai_messages(messages: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Flatten Anthropic-style block content into OpenAI's plain strings."""
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        out.append({"role": message.get("role", "user"), "content": content or ""})
    return out


def openrouter_provider(**kwargs: Any) -> OpenAIProvider:
    """The same adapter pointed at OpenRouter.

    OpenRouter speaks the OpenAI wire format, so this exercises exactly the code
    path that serves OpenAI itself -- which makes it a genuine verification of
    that adapter, not a separate one. Model ids are `vendor/model`.
    """
    return OpenAIProvider(
        base_url=OPENROUTER_BASE_URL,
        key_env="OPENROUTER_API_KEY",
        prefixes=("openai/", "anthropic/", "google/", "meta-llama/", "deepseek/", "gpt-"),
        headers={
            "HTTP-Referer": "https://github.com/dynamic-model-selection",
            "X-Title": "dms dispatch",
        },
        **kwargs,
    )


class ProviderRegistry:
    """Picks the adapter that handles a model id."""

    def __init__(
        self,
        providers: tuple[Provider, ...] | None = None,
        *,
        cache_system: bool = False,
        cache_ttl: str = "5m",
        book: Any | None = None,
    ) -> None:
        if providers is None:
            from dms.dispatch.codex_cli import CodexCLIProvider

            providers = (
                AnthropicProvider(
                    cache_system=cache_system, cache_ttl=cache_ttl, book=book
                ),
                # Must precede OpenAIProvider: a `codex-cli/gpt-...` id would
                # otherwise be claimed by the API adapter and sent to a key that
                # may not exist.
                CodexCLIProvider(),
                OpenAIProvider(),  # OpenAI caches automatically; nothing to declare
            )
        self.providers = providers

    def for_model(self, model: str) -> Provider:
        for provider in self.providers:
            if provider.handles(model):
                return provider
        raise ProviderError(
            f"no provider handles model {model!r}; "
            f"known prefixes: claude-, gpt-, o1/o3/o4, codex"
        )
