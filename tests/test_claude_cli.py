"""Claude Code (`claude -p`) as a dispatch backend. No subprocess is spawned:
the runner is injected, and the outputs below were recorded from claude 2.1.280."""
import json
import subprocess

import pytest

from dms.dispatch.claude_cli import (
    DEFAULT_SYSTEM_PROMPT,
    STRIPPED_ENV,
    ClaudeCLIProvider,
)
from dms.dispatch.providers import (
    AnthropicProvider,
    ProviderError,
    ProviderRegistry,
    Request,
)
from dms.pricing import PriceBook

MODEL = "claude-cli/claude-haiku-4-5"

SUCCESS = {
    "type": "result", "subtype": "success", "is_error": False,
    "result": "CLI_OK", "stop_reason": "end_turn", "num_turns": 1,
    "total_cost_usd": 0.000617,
    "usage": {"input_tokens": 382, "cache_creation_input_tokens": 0,
              "cache_read_input_tokens": 0, "output_tokens": 47},
}

# Recorded verbatim in shape: an error still reports subtype "success".
API_ERROR = {
    "type": "result", "subtype": "success", "is_error": True,
    "api_error_status": 404, "stop_reason": "stop_sequence", "terminal_reason": "api_error",
    "result": "There's an issue with the selected model (claude-nonexistent-9). "
              "It may not exist or you may not have access to it.",
}


class Runner:
    """Stands in for subprocess.run and records exactly what it was given."""

    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.result = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self.calls: list[dict] = []

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": argv, **kwargs})
        return self.result


def _provider(runner: Runner) -> ClaudeCLIProvider:
    return ClaudeCLIProvider(runner=runner, binary="claude", which=lambda _: "/usr/bin/claude")


def _req(text: str = "hello", **kw) -> Request:
    return Request(messages=({"role": "user", "content": text},), **kw)


# ------------------------------------------------------------------ parsing


def test_the_answer_and_usage_come_from_the_json_result() -> None:
    call = _provider(Runner(json.dumps(SUCCESS))).complete(MODEL, _req())

    assert call.text == "CLI_OK"
    assert call.model == MODEL
    assert call.stop_reason == "end_turn"
    assert (call.usage.input_tokens, call.usage.output_tokens) == (382, 47)


def test_an_error_is_raised_never_returned_as_an_answer() -> None:
    """claude -p reports an API error with subtype "success" and the error text in
    `result`. Trusting subtype would hand the error message to the caller as the
    model's answer -- a confident, wrong response."""
    provider = _provider(Runner(json.dumps(API_ERROR), returncode=1))

    with pytest.raises(ProviderError, match="selected model") as exc:
        provider.complete(MODEL, _req())

    assert exc.value.status == 404


def test_output_that_is_not_json_is_a_clear_error() -> None:
    with pytest.raises(ProviderError, match="not JSON"):
        _provider(Runner("Error: something broke", returncode=1)).complete(MODEL, _req())


def test_a_missing_binary_is_a_clear_error() -> None:
    provider = ClaudeCLIProvider(runner=Runner("{}"), which=lambda _: None)

    with pytest.raises(ProviderError, match="not on PATH"):
        provider.complete(MODEL, _req())


def test_a_timeout_is_a_clear_error() -> None:
    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1)

    provider = ClaudeCLIProvider(runner=slow, which=lambda _: "/usr/bin/claude")

    with pytest.raises(ProviderError, match="exceeded"):
        provider.complete(MODEL, _req())


# ------------------------------------------------------------ the command line


def _argv(runner: Runner) -> list[str]:
    return runner.calls[0]["argv"]


def test_the_underlying_model_is_passed_to_claude() -> None:
    runner = Runner(json.dumps(SUCCESS))
    _provider(runner).complete(MODEL, _req())

    argv = _argv(runner)
    assert argv[argv.index("--model") + 1] == "claude-haiku-4-5"


def test_claude_codes_agent_harness_is_switched_off() -> None:
    """No tools, no MCP servers, no settings/plugins/hooks, no skills, no saved
    session: that is what kept the probe request at 382 input tokens."""
    runner = Runner(json.dumps(SUCCESS))
    _provider(runner).complete(MODEL, _req())

    argv = _argv(runner)
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in argv


def test_bare_mode_is_never_used() -> None:
    """--bare reads only ANTHROPIC_API_KEY and never the OAuth login, which is
    the whole reason this backend exists."""
    runner = Runner(json.dumps(SUCCESS))
    _provider(runner).complete(MODEL, _req())

    assert "--bare" not in _argv(runner)


def test_the_callers_system_prompt_replaces_claude_codes() -> None:
    runner = Runner(json.dumps(SUCCESS))
    _provider(runner).complete(MODEL, _req(system="be terse"))

    argv = _argv(runner)
    assert argv[argv.index("--system-prompt") + 1] == "be terse"


def test_without_a_system_prompt_a_minimal_one_still_replaces_claude_codes() -> None:
    """Omitting --system-prompt would bring back Claude Code's full agent prompt,
    describing tools the request does not have."""
    runner = Runner(json.dumps(SUCCESS))
    _provider(runner).complete(MODEL, _req())

    argv = _argv(runner)
    assert argv[argv.index("--system-prompt") + 1] == DEFAULT_SYSTEM_PROMPT


def test_the_prompt_goes_through_stdin_not_the_command_line() -> None:
    """Anything on argv is visible to every local user via `ps`."""
    runner = Runner(json.dumps(SUCCESS))
    _provider(runner).complete(MODEL, _req("SECRET_QUESTION_TEXT"))

    assert "SECRET_QUESTION_TEXT" not in json.dumps(_argv(runner))
    assert "SECRET_QUESTION_TEXT" in runner.calls[0]["input"]


def test_caller_tools_are_never_forwarded() -> None:
    """claude -p runs its own tools instead of returning calls to the caller, so
    a caller's tool definitions can never be honoured here."""
    runner = Runner(json.dumps(SUCCESS))
    tools = ({"name": "rm_rf", "description": "x", "input_schema": {}},)
    _provider(runner).complete(MODEL, _req(tools=tools))

    assert "rm_rf" not in json.dumps(runner.calls[0])


# ---------------------------------------------------------------- the account


def test_api_key_and_router_variables_are_stripped(monkeypatch) -> None:
    """Any of these makes claude bill an API key or a gateway instead of the
    subscription login -- the thing this backend exists to use."""
    for name in STRIPPED_ENV:
        monkeypatch.setenv(name, "must-not-leak")
    runner = Runner(json.dumps(SUCCESS))

    _provider(runner).complete(MODEL, _req())

    env = runner.calls[0]["env"]
    assert not STRIPPED_ENV & env.keys()
    assert {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"} <= STRIPPED_ENV


def test_the_config_dir_is_kept_so_the_account_can_be_chosen(monkeypatch) -> None:
    """CLAUDE_CONFIG_DIR is how a Claude login (auto / extra / work) is picked."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/Users/x/.claude-extra")
    runner = Runner(json.dumps(SUCCESS))

    _provider(runner).complete(MODEL, _req())

    assert runner.calls[0]["env"]["CLAUDE_CONFIG_DIR"] == "/Users/x/.claude-extra"


# ------------------------------------------------------------------ streaming


def test_stream_yields_the_answer_and_reports_usage() -> None:
    """One chunk, honestly: this backend returns a finished result."""
    sink: list = []

    chunks = list(_provider(Runner(json.dumps(SUCCESS))).stream(MODEL, _req(), sink))

    assert chunks == ["CLI_OK"]
    assert sink[0].output_tokens == 47


# ------------------------------------------------------------------- routing


def test_the_cli_namespace_is_not_claimed_by_the_api_provider() -> None:
    """`claude-cli/...` starts with `claude-`; without care the API adapter would
    claim it and fail for want of a key."""
    assert not AnthropicProvider().handles(MODEL)
    assert AnthropicProvider().handles("claude-haiku-4-5")


def test_the_registry_sends_cli_ids_to_the_cli_backend() -> None:
    registry = ProviderRegistry()

    assert registry.for_model(MODEL).name == "claude-cli"
    assert registry.for_model("claude-haiku-4-5").name == "anthropic"


def test_a_cli_model_is_priced_off_the_underlying_model() -> None:
    book = PriceBook.load()

    assert book.resolve(MODEL) == "claude-haiku-4-5"
