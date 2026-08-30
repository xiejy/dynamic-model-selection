"""Codex CLI as a dispatch backend, parsed from recorded JSONL (no subprocess)."""
import pytest

from dms.dispatch.codex_cli import CodexCLIProvider, parse_events
from dms.dispatch.providers import ProviderError, ProviderRegistry, Request
from dms.pricing import PriceBook

# A real `codex exec --json` stream, trimmed. Note the tool call before the
# answer: Codex is an agent, and a backend built on it inherits that.
EVENTS = """
{"type":"thread.started","thread_id":"01a0"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"command_execution","command":"sed -n 1,5p x"}}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"CODEX_BACKEND_OK"}}
{"type":"turn.completed","usage":{"input_tokens":16862,"cached_input_tokens":11008,"cache_write_input_tokens":0,"output_tokens":9,"reasoning_output_tokens":4}}
"""


def test_the_answer_is_taken_from_the_agent_message() -> None:
    text, _ = parse_events(EVENTS)

    assert text == "CODEX_BACKEND_OK"


def test_tool_call_events_are_not_mistaken_for_the_answer() -> None:
    """A command_execution item is Codex working, not Codex answering."""
    text, _ = parse_events(EVENTS)

    assert "sed" not in text


def test_cached_input_is_not_billed_twice() -> None:
    """`input_tokens` includes the cached portion, as everywhere in the OpenAI
    lineage. Failing to subtract would bill 11,008 tokens at full rate."""
    _, usage = parse_events(EVENTS)

    assert usage.input_tokens == 5854          # 16862 - 11008
    assert usage.cache_read_input_tokens == 11008
    assert usage.prompt_tokens == 16862        # the sum still reconstructs the total


def test_reasoning_tokens_are_not_added_to_output() -> None:
    """reasoning_output_tokens is a subset of output_tokens, not an addition."""
    _, usage = parse_events(EVENTS)

    assert usage.output_tokens == 9


def test_garbage_lines_are_ignored() -> None:
    text, usage = parse_events("not json\n\n" + EVENTS + "\n[]\n")

    assert text == "CODEX_BACKEND_OK"
    assert usage.output_tokens == 9


def test_an_empty_stream_yields_nothing_rather_than_crashing() -> None:
    text, usage = parse_events("")

    assert text == ""
    assert usage.total_tokens == 0


# ------------------------------------------------------------------- routing


def test_the_namespace_keeps_cli_and_api_ids_apart() -> None:
    """`codex-cli/gpt-5.6-sol` and `gpt-5.6-sol` are not interchangeable: one
    needs a ChatGPT login, the other an API key. Confusing them would call the
    wrong backend and report the wrong cost."""
    registry = ProviderRegistry()

    assert registry.for_model("codex-cli/gpt-5.6-sol").name == "codex-cli"
    assert registry.for_model("gpt-5.6-sol").name == "openai"


def test_underlying_model_is_recovered_for_the_subprocess() -> None:
    assert CodexCLIProvider.underlying("codex-cli/gpt-5.6-sol") == "gpt-5.6-sol"


def test_a_cli_model_is_priced_off_its_underlying_model() -> None:
    book = PriceBook.load()

    assert book.resolve("codex-cli/gpt-5.6-sol") == "gpt-5.6-sol"
    assert book.rate("codex-cli/gpt-5.6-sol").input_usd_per_million == (
        book.rate("gpt-5.6-sol").input_usd_per_million
    )


def test_a_missing_binary_is_a_clear_error() -> None:
    provider = CodexCLIProvider(binary="definitely-not-installed-xyz")

    with pytest.raises(ProviderError, match="not on PATH"):
        provider.complete("codex-cli/gpt-5", Request(messages=()))


# -------------------------------------------------------------------- prompt


def test_system_and_user_turns_are_flattened_into_one_prompt() -> None:
    prompt = CodexCLIProvider._prompt(
        Request(
            messages=({"role": "user", "content": "question"},),
            system="be terse",
        )
    )

    assert prompt == "be terse\n\nquestion"


def test_block_content_is_flattened() -> None:
    prompt = CodexCLIProvider._prompt(
        Request(messages=({"role": "user", "content": [
            {"type": "input_text", "text": "a"}, {"type": "text", "text": "b"}]},))
    )

    assert prompt == "ab"


def test_caller_tools_are_never_forwarded_into_the_agent() -> None:
    """Codex actually executes tools. Forwarding a caller's definitions into it
    is a security decision, not a format translation."""
    prompt = CodexCLIProvider._prompt(
        Request(
            messages=({"role": "user", "content": "hi"},),
            tools=({"name": "rm_rf", "description": "delete everything",
                    "input_schema": {}},),
        )
    )

    assert "rm_rf" not in prompt


def test_the_sandbox_is_read_only_by_default() -> None:
    argv = CodexCLIProvider()._argv("codex-cli/gpt-5")

    assert "--sandbox" in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[argv.index("-m") + 1] == "gpt-5"
