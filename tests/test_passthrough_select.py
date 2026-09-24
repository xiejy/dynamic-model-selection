"""Per-request model choice for pass-through mode."""
import json

import pytest

from dms.passthrough.select import (
    PassthroughSelector,
    Tier,
    TurnKind,
    classify_turn,
    latest_user_text,
)

HIGH, LOW = "gpt-5.6-sol@personal", "gpt-5.6-luna@personal"
C_HIGH, C_LOW = "claude-opus-5", "claude-sonnet-5"


def _codex(*items, model=HIGH) -> dict:
    return {"model": model, "input": list(items)}


def _user(text: str) -> dict:
    return {"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": text}]}


def _claude(*messages, model=C_HIGH) -> dict:
    return {"model": model, "messages": list(messages)}


EASY = "Extract the port number from this string."
HARD = "Explain the root cause of this deadlock between two mutexes, step by step."


# ------------------------------------------------------------- turn detection


def test_a_codex_user_message_is_a_new_turn() -> None:
    assert classify_turn("openai", _codex(_user("hi"))) is TurnKind.NEW


@pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
def test_a_codex_tool_result_is_a_continuation(kind) -> None:
    body = _codex(_user("run ls"), {"type": "function_call", "name": "exec"},
                  {"type": kind, "output": "a b"})

    assert classify_turn("openai", body) is TurnKind.CONTINUATION


def test_a_claude_text_message_is_a_new_turn() -> None:
    body = _claude({"role": "user", "content": [{"type": "text", "text": "hi"}]})

    assert classify_turn("anthropic", body) is TurnKind.NEW


def test_a_claude_tool_result_is_a_continuation() -> None:
    body = _claude(
        {"role": "user", "content": "run ls"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Bash"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t"}]},
    )

    assert classify_turn("anthropic", body) is TurnKind.CONTINUATION


def test_a_trailing_mid_conversation_system_message_is_looked_past() -> None:
    """Claude Code appends a role:"system" message after the user's turn --
    captured from a real request. The turn is decided by what precedes it."""
    body = _claude({"role": "user", "content": "hi"},
                   {"role": "system", "content": [{"type": "text", "text": "reminder"}]})

    assert classify_turn("anthropic", body) is TurnKind.NEW


def test_the_scored_text_is_the_latest_user_message_without_system_reminders() -> None:
    body = _claude({"role": "user", "content": [
        {"type": "text", "text": "<system-reminder>huge injected context</system-reminder>"},
        {"type": "text", "text": "Extract the port."}]})

    assert latest_user_text("anthropic", body) == "Extract the port."


# ------------------------------------------------------------------- decisions


def _selector(**kw) -> PassthroughSelector:
    return PassthroughSelector(family="openai", low=LOW, high=HIGH, **kw)


def test_an_easy_new_turn_goes_low_and_a_hard_one_high() -> None:
    selector = _selector()

    assert selector.decide(_codex(_user(EASY)), "s1").tier is Tier.LOW
    assert selector.decide(_codex(_user(HARD)), "s2").tier is Tier.HIGH


def test_the_low_decision_rewrites_to_the_low_model() -> None:
    decision = _selector().decide(_codex(_user(EASY)), "s1")

    assert decision.model == LOW


def test_a_tool_step_keeps_the_model_its_turn_chose() -> None:
    """Switching mid-task discards the prompt cache and changes behaviour halfway."""
    selector = _selector()
    selector.decide(_codex(_user(EASY)), "s1")

    step = selector.decide(
        _codex(_user(EASY), {"type": "function_call_output", "output": HARD * 20}), "s1"
    )

    assert step.tier is Tier.LOW
    assert step.turn is TurnKind.CONTINUATION


def test_each_new_message_is_decided_afresh() -> None:
    selector = _selector()
    assert selector.decide(_codex(_user(EASY)), "s1").tier is Tier.LOW

    assert selector.decide(_codex(_user(EASY), _user(HARD)), "s1").tier is Tier.HIGH


def test_a_tool_step_with_no_known_turn_fails_toward_quality() -> None:
    step = _selector().decide(
        _codex(_user("x"), {"type": "function_call_output", "output": "y"}), "never-seen"
    )

    assert step.tier is Tier.HIGH


def test_requests_for_other_models_pass_untouched() -> None:
    """Clients make background calls on other models (Claude Code's safety
    classifier, titles). The proxy only ever downgrades the main model."""
    decision = _selector().decide(_codex(_user(EASY), model="gpt-5.5@personal"), "s1")

    assert decision.tier is Tier.UNTOUCHED
    assert decision.model == "gpt-5.5@personal"


def test_a_prompt_too_big_for_the_low_model_stays_high() -> None:
    """Claude Code on Opus runs a 1M window; Haiku has 200k. Downgrading an
    oversized conversation would fail outright."""
    selector = PassthroughSelector(
        family="anthropic", low="claude-haiku-4-5", high=C_HIGH, low_context_tokens=200_000
    )
    huge = _claude({"role": "user", "content": "x " * 500_000 + EASY})

    decision = selector.decide(huge, "s1", body_bytes=1_200_000)

    assert decision.tier is Tier.HIGH
    assert "context" in decision.why


def test_no_session_still_decides_new_turns() -> None:
    assert _selector().decide(_codex(_user(EASY)), None).tier is Tier.LOW


# ------------------------------------------------------- compatibility warning


def test_downgrading_claude_code_to_haiku_warns() -> None:
    """Measured: Haiku 4.5 rejected Claude Code's Opus-shaped request three times
    (adaptive thinking, effort, role 'system') before Claude Code's own retries
    stripped enough for it to pass."""
    from dms.passthrough.select import downgrade_warning

    warning = downgrade_warning("anthropic", "claude-haiku-4-5")

    assert warning and "claude-sonnet-5" in warning


@pytest.mark.parametrize("family,low", [("anthropic", "claude-sonnet-5"),
                                        ("openai", "gpt-5.6-luna@personal")])
def test_measured_clean_downgrades_do_not_warn(family, low) -> None:
    from dms.passthrough.select import downgrade_warning

    assert downgrade_warning(family, low) is None


# ------------------------------------------- side requests and sub-agents (review)


def _claude_selector() -> PassthroughSelector:
    return PassthroughSelector(family="anthropic", low=C_LOW, high=C_HIGH)


def _text(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _tool_result() -> dict:
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t"}]}


def _tool_use() -> dict:
    return {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "WebSearch"}]}


def test_a_side_request_on_the_main_model_does_not_move_the_main_turn() -> None:
    """Claude Code's WebSearch tool sends a side request on the main model, in
    the same session, with a one-message conversation of its own. Scoring it
    must not overwrite the choice the user's own message made."""
    selector = _claude_selector()
    assert selector.decide(_claude(_text("user", HARD)), "s1").tier is Tier.HIGH

    side = selector.decide(
        _claude(_text("user", "Perform a web search for the query: glibc 2.39 release notes")),
        "s1",
    )
    step = selector.decide(_claude(_text("user", HARD), _tool_use(), _tool_result()), "s1")

    assert side.tier is Tier.LOW
    assert (step.turn, step.tier) == (TurnKind.CONTINUATION, Tier.HIGH)


def test_a_request_that_is_not_a_new_message_keeps_the_existing_choice() -> None:
    """A Codex sub-agent's first request ends with an agent message, not a user
    one. It must not re-decide -- and re-pin -- the conversation."""
    selector = _selector()
    first = _user(HARD)
    assert selector.decide(_codex(first), "s1").tier is Tier.HIGH

    other = selector.decide(
        _codex(first, _user(EASY), {"type": "message", "role": "assistant",
                                    "content": [{"type": "output_text", "text": "spawned"}]}),
        "s1")
    step = selector.decide(_codex(first, {"type": "function_call_output", "output": "x"}), "s1")

    assert other.turn is TurnKind.OTHER and other.tier is Tier.HIGH
    assert step.tier is Tier.HIGH


def test_a_conversation_is_recognised_without_a_session_header() -> None:
    """Pins are keyed by the conversation's opening message too, so a client
    that sends no session header still keeps a turn's choice for its tool steps."""
    selector = _selector()
    selector.decide(_codex(_user(EASY)), None)

    step = selector.decide(_codex(_user(EASY), {"type": "function_call_output", "output": "y"}),
                           None)

    assert step.tier is Tier.LOW


# ------------------------------------------------ Codex's injected context (review)


ENV = "<environment_context>\n  <cwd>/repo</cwd>\n  <shell>zsh</shell>\n</environment_context>"


def test_codex_context_appended_after_a_tool_result_is_still_that_turn() -> None:
    """Codex can append an <environment_context> user item right after a tool
    output, mid-turn. It is not a message the user typed."""
    selector = _selector()
    selector.decide(_codex(_user(HARD)), "s1")

    step = selector.decide(
        _codex(_user(HARD), {"type": "custom_tool_call_output", "output": "ok"}, _user(ENV)), "s1")

    assert (step.turn, step.tier) == (TurnKind.CONTINUATION, Tier.HIGH)


def test_the_scored_text_skips_items_made_only_of_injected_context() -> None:
    body = _codex(_user(HARD), _user(ENV))

    assert latest_user_text("openai", body) == HARD


def test_ide_context_around_the_typed_request_is_not_scored() -> None:
    """The Codex IDE extension wraps the typed request in 50-60 words of browser
    or file context, which alone crossed the long-prompt threshold."""
    wrapped = ('<in-app-browser-context source="ambient-ui-state">\n# In app browser:\n'
               + "tab title and url " * 15 + "\n</in-app-browser-context>\n\n"
               + "## My request:\n" + EASY)

    assert latest_user_text("openai", _codex(_user(wrapped))) == EASY
    assert _selector().decide(_codex(_user(wrapped)), "s1").tier is Tier.LOW


@pytest.mark.parametrize("injected", [
    "# AGENTS.md instructions for /repo\n\n<INSTRUCTIONS>\nbe terse\n</INSTRUCTIONS>",
    "<turn_aborted>\nThe user interrupted.\n</turn_aborted>",
    "<recommended_plugins>\n- a\n- b\n</recommended_plugins>",
])
def test_other_codex_wrappers_are_not_typed_text(injected) -> None:
    assert latest_user_text("openai", _codex(_user(EASY), _user(injected))) == EASY


# --------------------------------------- screenshots and the context guard (live bug)


def _screenshot(kb: int) -> str:
    """A fake PNG as Codex's browser tools send it: base64 behind a data URL."""
    return "data:image/png;base64,iVBORw0KGgo" + "A" * (kb * 1024)


def test_screenshots_are_not_measured_by_their_bytes() -> None:
    """Measured 2026-09-24: a browser task's request was 1 MB, 0.45 MB of it
    screenshots, for a 123k-token prompt. At 3 bytes a token that read as ~340k,
    85% of gpt-5.6-luna's 400k window, and the rest of the turn went to Sol."""
    from dms.passthrough.select import prompt_bytes

    images = [{"type": "input_image", "image_url": _screenshot(120)} for _ in range(19)]
    body = json.dumps(_codex(_user(EASY), {"type": "message", "role": "user", "content": images},
                             {"type": "function_call_output", "output": "ok " * 60_000})).encode()
    selector = PassthroughSelector(family="openai", low=LOW, high=HIGH, low_context_tokens=400_000)
    selector.decide(_codex(_user(EASY)), "s1")  # the typed message: low

    step = selector.decide(json.loads(body), "s1", body_bytes=prompt_bytes(body))

    assert len(body) > 2_300_000
    assert "context" not in step.why and step.tier is Tier.LOW


def test_each_image_still_counts_as_some_tokens() -> None:
    from dms.passthrough.select import CHARS_PER_TOKEN, IMAGE_TOKENS, prompt_bytes

    raw = json.dumps({"input": [{"image_url": _screenshot(200)}]}).encode()

    assert prompt_bytes(raw) >= IMAGE_TOKENS * CHARS_PER_TOKEN
    assert prompt_bytes(raw) < len(raw) / 10


def test_other_long_blobs_keep_their_full_weight() -> None:
    """Encrypted reasoning is replayed into the context; only images are discounted."""
    from dms.passthrough.select import prompt_bytes

    raw = json.dumps({"input": [{"encrypted_content": "gAAAA" + "b" * 200_000}]}).encode()

    assert prompt_bytes(raw) == len(raw)


def test_anthropic_image_blocks_are_images_too() -> None:
    from dms.passthrough.select import prompt_bytes

    block = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": "/9j/4AAQ" + "B" * 300_000}}
    raw = json.dumps({"messages": [{"role": "user", "content": [block]}]}).encode()

    assert prompt_bytes(raw) < len(raw) / 10
