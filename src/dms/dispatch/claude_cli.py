"""Claude Code (`claude -p`) as a dispatch backend: Claude without an API key.

`claude -p --output-format json` is Claude Code's documented headless mode. It
authenticates with the Claude Code login -- a Pro/Max/Team subscription -- and
prints one JSON result carrying the answer and Anthropic-shaped token usage, so
Claude becomes a dispatch tier with no ANTHROPIC_API_KEY.

Claude Code is an agent, so every call switches its harness off: no tools, no
MCP servers, no settings or plugins (and so no hooks), no skills, no saved
session, and the caller's system prompt in place of Claude Code's own. Measured
on claude 2.1.280, that kept a one-line question at 382 input tokens.

Which account pays is whichever login `CLAUDE_CONFIG_DIR` points at (unset:
`~/.claude`). The variables that would redirect billing to an API key or a
gateway are stripped from the child environment, so this backend can only ever
spend the subscription it was started under.

Limits, each inherent to driving a CLI rather than the API:

* `max_tokens`, `temperature` and `stop_sequences` cannot be passed and are
  ignored. Claude Code picks its own output ceiling.
* Caller tools are never honoured: `claude -p` executes its own tools rather
  than returning calls to the caller, so a tool relay cannot run over it.
* Every call starts a process: ~3 s measured, against ~1.3 s of model time.
* Streaming is one chunk -- the finished result -- not token deltas.

`--bare` would skip more of the harness, but it reads only ANTHROPIC_API_KEY and
never the OAuth login, which defeats the purpose.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from typing import Any

from dms.dispatch.providers import Completion, ProviderError, Request, transcript_text
from dms.usage import UsageRecord

MODEL_PREFIX = "claude-cli/"
DEFAULT_TIMEOUT_SECONDS = 300

# Replaces Claude Code's own agent prompt, which describes tools this backend
# switches off. Used only when the caller sends no system prompt.
DEFAULT_SYSTEM_PROMPT = "Answer the user's request directly."

# Each of these routes `claude` to an API key or a gateway instead of the
# subscription login.
STRIPPED_ENV = frozenset(
    {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"}
)

# Switches off tools, MCP servers, settings/plugins/hooks, skills and session
# persistence -- everything that makes Claude Code an agent rather than a model.
HARNESS_OFF = (
    "--tools", "",
    "--strict-mcp-config",
    "--setting-sources", "",
    "--disable-slash-commands",
    "--no-session-persistence",
)

Runner = Callable[..., subprocess.CompletedProcess]


class ClaudeCLIProvider:
    """Runs `claude -p` and adapts its JSON result to the provider contract.

    Model ids are namespaced `claude-cli/<model>` so they are never confused with
    API ids: one bills a subscription login, the other an API key.
    """

    name = "claude-cli"

    def __init__(
        self,
        *,
        binary: str = "claude",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        runner: Runner = subprocess.run,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self.binary = binary
        self.timeout = timeout
        self._run = runner
        self._which = which

    def handles(self, model: str) -> bool:
        return model.startswith(MODEL_PREFIX)

    @staticmethod
    def underlying(model: str) -> str:
        """claude-cli/claude-haiku-4-5 -> claude-haiku-4-5"""
        return model[len(MODEL_PREFIX):] if model.startswith(MODEL_PREFIX) else model

    def _argv(self, model: str, request: Request) -> list[str]:
        if not self._which(self.binary):
            raise ProviderError(f"{self.binary!r} is not on PATH")
        return [
            self.binary, "-p",
            "--output-format", "json",
            "--model", self.underlying(model),
            "--system-prompt", request.system or DEFAULT_SYSTEM_PROMPT,
            *HARNESS_OFF,
        ]

    @staticmethod
    def _env() -> dict[str, str]:
        return {k: v for k, v in os.environ.items() if k not in STRIPPED_ENV}

    def complete(self, model: str, request: Request) -> Completion:
        started = time.perf_counter()
        try:
            proc = self._run(
                self._argv(model, request),
                # stdin, not argv: argv is readable by every local user via `ps`.
                input=transcript_text(request.messages),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=self._env(),
            )
        except subprocess.TimeoutExpired:
            raise ProviderError(f"claude -p exceeded {self.timeout}s") from None
        except OSError as exc:
            raise ProviderError(f"could not run {self.binary!r}: {exc}") from None

        result = _parse(proc)
        return Completion(
            text=(result.get("result") or "").strip(),
            model=model,
            usage=UsageRecord.from_dict(result.get("usage") or {}),
            stop_reason=result.get("stop_reason") or "end_turn",
            latency_ms=(time.perf_counter() - started) * 1000,
            raw=result,
        )

    def stream(
        self, model: str, request: Request, usage_sink: list[UsageRecord] | None = None
    ) -> Iterator[str]:
        """One chunk: the finished result. Honest coarse streaming beats
        pretending to stream deltas that `--output-format json` never produced."""
        completion = self.complete(model, request)
        if usage_sink is not None:
            usage_sink.append(completion.usage)
        yield completion.text


def _parse(proc: subprocess.CompletedProcess) -> dict[str, Any]:
    """Read the JSON result, raising on anything that is not a real answer.

    An API error arrives with `subtype: "success"` and the error message in
    `result` -- only `is_error` marks it. Without this check the error text
    would be returned to the caller as though the model had said it.
    """
    try:
        result = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        detail = (proc.stderr or proc.stdout or "").strip()[:300]
        raise ProviderError(
            f"claude -p output was not JSON (exit {proc.returncode}): {detail}"
        ) from None

    if not isinstance(result, dict):
        raise ProviderError(f"claude -p returned {type(result).__name__}, not an object")
    if result.get("is_error"):
        raise ProviderError(
            f"claude -p failed: {str(result.get('result') or 'unknown error')[:300]}",
            status=result.get("api_error_status"),
        )
    return result
