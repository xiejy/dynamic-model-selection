"""Codex CLI as a dispatch backend: GPT models without an API key.

`codex exec --json` authenticates through the user's ChatGPT login and emits
JSONL events that carry both halves a provider needs -- the final text on an
`item.completed` event of type `agent_message`, and token counts on
`turn.completed`. That makes GPT reachable as a low or high tier with no
`OPENAI_API_KEY` and no separate API billing.

**It is an agent, not a completion endpoint, and that matters:**

* It carries its own harness prompt -- ~35k tokens in the observed run, mostly
  cached -- so a short question is never a short request.
* It will run tools. The observed run shelled out to `sed` to read a skills file
  before answering. This provider forces read-only sandboxing and never
  forwards caller-supplied tools, but it cannot make Codex stop being an agent.
* Latency is seconds, dominated by process start and the agent loop, not by the
  model.
* Spend lands on the ChatGPT subscription, not a metered API key, so the dollar
  figures this provider reports are *what the equivalent API call would cost*
  from the price book -- useful for comparison, not a bill.

Use it to compare tiers across providers or to reach GPT without a key. For
latency-sensitive production traffic, the API path (`OpenAIProvider`) is the
right one.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Iterator
from typing import Any

from dms.dispatch.providers import Completion, ProviderError, Request
from dms.usage import UsageRecord

DEFAULT_TIMEOUT_SECONDS = 300
MODEL_PREFIX = "codex-cli/"


class CodexCLIProvider:
    """Runs `codex exec` as a subprocess and adapts it to the provider contract.

    Model ids are namespaced `codex-cli/<model>` so they can never be confused
    with API model ids -- the two are not interchangeable, and a silent mix-up
    would bill the wrong way and report the wrong cost.
    """

    name = "codex-cli"

    def __init__(
        self,
        *,
        binary: str = "codex",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        sandbox: str = "read-only",
        cwd: str | None = None,
        price_model: str | None = None,
    ) -> None:
        self.binary = binary
        self.timeout = timeout
        self.sandbox = sandbox
        self.cwd = cwd
        # Which price-book entry to cost this against; defaults to the model name.
        self.price_model = price_model

    def handles(self, model: str) -> bool:
        return model.startswith(MODEL_PREFIX)

    @staticmethod
    def underlying(model: str) -> str:
        """codex-cli/gpt-5.6-sol -> gpt-5.6-sol"""
        return model[len(MODEL_PREFIX):] if model.startswith(MODEL_PREFIX) else model

    def _argv(self, model: str) -> list[str]:
        if not shutil.which(self.binary):
            raise ProviderError(f"{self.binary!r} is not on PATH")
        return [
            self.binary, "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox", self.sandbox,
            "-m", self.underlying(model),
        ]

    @staticmethod
    def _prompt(request: Request) -> str:
        """Flatten the request into the single prompt `codex exec` accepts.

        Caller-supplied tools are deliberately dropped: Codex brings its own, and
        forwarding a caller's tool definitions into an agent that will actually
        execute things is not a translation, it is a security decision nobody
        asked for.
        """
        parts: list[str] = []
        if request.system:
            parts.append(request.system)
        for message in request.messages:
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") in ("text", "input_text")
                )
            if content:
                role = message.get("role", "user")
                parts.append(content if role == "user" else f"[{role}] {content}")
        return "\n\n".join(parts)

    def complete(self, model: str, request: Request) -> Completion:
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                self._argv(model),
                input=self._prompt(request),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
            )
        except subprocess.TimeoutExpired:
            raise ProviderError(
                f"codex exec exceeded {self.timeout}s -- it is an agent loop, not a "
                "single completion; raise the timeout or use the API provider"
            ) from None
        except OSError as exc:
            raise ProviderError(f"could not run {self.binary!r}: {exc}") from None

        latency = (time.perf_counter() - started) * 1000
        text, usage = parse_events(proc.stdout)

        if not text and proc.returncode != 0:
            raise ProviderError(
                f"codex exec failed (exit {proc.returncode}): "
                f"{(proc.stderr or '').strip()[:300]}"
            )

        return Completion(
            text=text,
            model=model,
            usage=usage,
            stop_reason="end_turn" if text else "refusal",
            latency_ms=latency,
            raw=proc.stdout,
        )

    def stream(
        self, model: str, request: Request, usage_sink: list[UsageRecord] | None = None
    ) -> Iterator[str]:
        """Emit each agent message as it completes.

        Codex's JSONL carries completed items, not token deltas, so this is
        chunk-per-message rather than true token streaming. Honest coarse
        streaming beats pretending to stream tokens that were never streamed.
        """
        try:
            proc = subprocess.Popen(
                self._argv(model),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=self.cwd,
            )
        except OSError as exc:
            raise ProviderError(f"could not run {self.binary!r}: {exc}") from None

        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(self._prompt(request))
        proc.stdin.close()

        usage = UsageRecord()
        for line in proc.stdout:
            event = _load(line)
            if event is None:
                continue
            if _is_agent_message(event):
                yield event["item"].get("text", "")
            elif event.get("type") == "turn.completed":
                usage = _usage_of(event.get("usage") or {})
        proc.wait(timeout=self.timeout)

        if usage_sink is not None:
            usage_sink.append(usage)


# ------------------------------------------------------------------- parsing


def parse_events(stdout: str) -> tuple[str, UsageRecord]:
    """Pull the answer text and token usage out of a `codex exec --json` stream."""
    messages: list[str] = []
    usage = UsageRecord()

    for line in stdout.splitlines():
        event = _load(line)
        if event is None:
            continue
        if _is_agent_message(event):
            messages.append(event["item"].get("text", ""))
        elif event.get("type") == "turn.completed":
            usage = _usage_of(event.get("usage") or {})

    return "\n".join(part for part in messages if part).strip(), usage


def _usage_of(usage: dict[str, Any]) -> UsageRecord:
    """Map Codex's usage shape onto the internal one.

    `input_tokens` INCLUDES the cached portion, as it does everywhere in the
    OpenAI lineage -- subtract it or the cached tokens are billed twice.
    `reasoning_output_tokens` is a subset of `output_tokens`, not an addition.
    """
    total_input = int(usage.get("input_tokens") or 0)
    cached = int(usage.get("cached_input_tokens") or 0)
    return UsageRecord(
        input_tokens=max(0, total_input - cached),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_creation_input_tokens=int(usage.get("cache_write_input_tokens") or 0),
        cache_read_input_tokens=cached,
    )


def _is_agent_message(event: dict[str, Any]) -> bool:
    item = event.get("item")
    return (
        event.get("type") == "item.completed"
        and isinstance(item, dict)
        and item.get("type") == "agent_message"
    )


def _load(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
