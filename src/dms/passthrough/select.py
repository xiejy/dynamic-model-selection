"""Choose the model for one pass-through request.

The rules, in order:

1. **Only the main model is ever changed.** A request for any other model --
   Claude Code's safety classifier, title generation, a background call -- is
   forwarded untouched. The proxy can downgrade the main model; it never
   touches the rest.
2. **Never downgrade past the low model's context window.** Claude Code on
   Opus runs a 1M window; Haiku has 200k. An oversized conversation stays high.
3. **Decide on each new message the user types**, by scoring that message --
   without the context the client injects around it (Claude Code's reminders,
   Codex's environment and IDE blocks).
4. **Keep that choice for every tool step it triggers.** Switching mid-task
   discards the prompt cache and changes behaviour halfway through. A tool step
   whose turn was never seen fails toward quality: high.

A choice is remembered per conversation: the session header plus the text of
the conversation's opening message. Side requests on the main model (Claude
Code's web search) and sub-agents open with a message of their own, so they are
decided on their own and never overwrite the main conversation's choice.

The scorer is the repo's zero-token heuristic; the cascade cannot be used here,
because it works by checking a finished cheap answer and a tool step is not one.
"""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from typing import Any, Protocol

from dms.routers.heuristic import HeuristicRouter

CHARS_PER_TOKEN = 3.0     # conservative: overestimates tokens, so errs toward high
CONTEXT_HEADROOM = 0.85   # leave room for the reply and estimation error
IMAGE_TOKENS = 2_500      # per image: a generous ceiling for one screenshot
MAX_PINS = 10_000         # conversations remembered; the oldest is forgotten first

# Blocks a client wraps around or appends to the user's message. Tag names from
# Claude Code 2.1 requests and Codex 0.153 rollouts.
_INJECTED_TAGS = (
    "system-reminder", "environment_context", "in-app-browser-context",
    "recommended_plugins", "turn_aborted", "guardian_tool_descriptions",
    "user_instructions", "INSTRUCTIONS", "response-annotations",
)
_INJECTED_BLOCK = re.compile(
    rf"<({'|'.join(map(re.escape, _INJECTED_TAGS))})\b[^>]*>.*?</\1>", re.DOTALL
)
_INJECTED_LINE = re.compile(
    r"^(# AGENTS\.md instructions for .*|<image\b[^>]*>|</image>)$", re.MULTILINE
)
# The Codex IDE extension puts the typed request last, under this heading.
_TYPED_REQUEST = re.compile(r"^## My request(?: for Codex)?:[ \t]*$", re.MULTILINE)

# A JSON string of base64 image data -- behind a data: URL (OpenAI, Codex) or as
# an Anthropic image block's bare "data" -- known by the formats' own base64
# signatures: PNG, JPEG, GIF, WebP.
_IMAGE_BLOB = re.compile(
    rb'"(?:data:image/[a-z0-9.+-]+;base64,)?(?:iVBORw0KGgo|/9j/|R0lGOD|UklGR)[A-Za-z0-9+/]*+={0,2}"'
)

_TEXT_PARTS = {"text", "input_text"}
_TOOL_OUTPUT_ITEMS = {"function_call_output", "custom_tool_call_output"}


class Tier(StrEnum):
    LOW = "low"
    HIGH = "high"
    UNTOUCHED = "untouched"


class TurnKind(StrEnum):
    NEW = "new"                    # the user typed something
    CONTINUATION = "continuation"  # a tool result is being handed back
    OTHER = "other"


class Scorer(Protocol):
    def score(self, prompt: str) -> tuple[float, Any]: ...


@dataclass(frozen=True, slots=True)
class PassDecision:
    model: str
    tier: Tier
    turn: TurnKind
    why: str


# --------------------------------------------------------------- reading a body


def classify_turn(family: str, body: dict[str, Any]) -> TurnKind:
    if family == "openai" and isinstance(body.get("input"), str):
        return TurnKind.NEW
    last = next((i for i in reversed(_items(family, body)) if not _injected_only(i)), None)
    if last is None:
        return TurnKind.OTHER
    if family == "openai":
        if last.get("type") in _TOOL_OUTPUT_ITEMS:
            return TurnKind.CONTINUATION
        return TurnKind.NEW if last.get("role") == "user" else TurnKind.OTHER

    if last.get("role") != "user":
        return TurnKind.OTHER
    content = last.get("content")
    if isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    ):
        return TurnKind.CONTINUATION
    return TurnKind.NEW


def latest_user_text(family: str, body: dict[str, Any]) -> str:
    """The text of the user's latest typed message, without injected context."""
    if family == "openai" and isinstance(body.get("input"), str):
        return body["input"]
    for item in reversed(_items(family, body)):
        if item.get("role") == "user" and (text := _strip(_text_of(item.get("content")))):
            return text
    return ""


def conversation_key(family: str, body: dict[str, Any]) -> str:
    """A digest of the conversation's opening user message.

    It is stable across a conversation's turns -- the prompt cache depends on
    the prefix not changing -- and differs for a side request or sub-agent that
    opens with a message of its own. Kept in memory only.
    """
    if family == "openai" and isinstance(body.get("input"), str):
        opening = body["input"]
    else:
        first = next((i for i in _items(family, body) if i.get("role") == "user"), None)
        opening = _text_of(first.get("content")) if first else ""
    return hashlib.sha256(opening.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _items(family: str, body: dict[str, Any]) -> list[dict[str, Any]]:
    """The conversation, minus the role:"system" messages Claude Code appends
    after the user's turn."""
    raw = body.get("input") if family == "openai" else body.get("messages")
    if not isinstance(raw, list):
        return []
    return [i for i in raw if isinstance(i, dict) and i.get("role") != "system"]


def _injected_only(item: dict[str, Any]) -> bool:
    """A user message made only of context the client injected, nothing typed."""
    if item.get("role") != "user":
        return False
    content = item.get("content")
    if isinstance(content, list) and any(
        not isinstance(b, dict) or b.get("type") not in _TEXT_PARTS for b in content
    ):
        return False  # an image, a tool result: something real
    text = _text_of(content)
    return bool(text.strip()) and not _strip(text)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") in _TEXT_PARTS
        )
    return ""


def _strip(text: str) -> str:
    typed = None
    for typed in _TYPED_REQUEST.finditer(text):
        pass
    if typed:
        text = text[typed.end():]
    return _INJECTED_LINE.sub("", _INJECTED_BLOCK.sub("", text)).strip()


def prompt_bytes(raw: bytes) -> int:
    """The request's size as the context guard should read it.

    A screenshot is tens of kilobytes of base64 but a couple of thousand tokens.
    Measured by bytes, a Codex browser task whose real prompt was 123k tokens --
    a 1 MB request, 0.45 MB of it seven screenshots -- read as ~340k and was
    pushed off a 400k-window model (2026-09-24). Each image now counts as
    IMAGE_TOKENS; everything else -- encrypted reasoning included, which is
    replayed into the context -- keeps its full size, so the estimate still errs
    high (that request now reads as ~207k).
    """
    images = _IMAGE_BLOB.findall(raw)
    blob_bytes = sum(len(m) for m in images)
    return len(raw) - blob_bytes + int(len(images) * IMAGE_TOKENS * CHARS_PER_TOKEN)


# ------------------------------------------------------------------- deciding


class PassthroughSelector:
    """Stateful per session: remembers the tier each typed message chose."""

    def __init__(
        self,
        *,
        family: str,
        low: str,
        high: str,
        low_context_tokens: int | None = None,
        scorer: Scorer | None = None,
        threshold: float = 1.0,
    ) -> None:
        self.family = family
        self.low = low
        self.high = high
        self.low_context_tokens = low_context_tokens
        self._scorer = scorer or HeuristicRouter()
        self._threshold = threshold
        self._pins: OrderedDict[str, Tier] = OrderedDict()
        self._lock = Lock()

    def decide(
        self, body: dict[str, Any], session: str | None, *, body_bytes: int = 0
    ) -> PassDecision:
        requested = body.get("model", "")
        turn = classify_turn(self.family, body)

        if requested != self.high:
            return PassDecision(requested, Tier.UNTOUCHED, turn, "not the main model")

        key = f"{session or ''}|{conversation_key(self.family, body)}"
        if self._too_big_for_low(body_bytes):
            return self._remember(key, Tier.HIGH, turn,
                                  "conversation exceeds the low model's context window")

        if turn is not TurnKind.NEW:
            with self._lock:
                pinned = self._pins.get(key)
            if pinned is not None:
                return PassDecision(self._model(pinned), pinned, turn,
                                    f"{turn} keeps its turn's choice ({pinned})")
            if turn is TurnKind.CONTINUATION:
                return PassDecision(self.high, Tier.HIGH, turn,
                                    "tool step of an unseen turn -> high")

        score, signals = self._scorer.score(latest_user_text(self.family, body))
        tier = Tier.HIGH if score >= self._threshold else Tier.LOW
        fired = ",".join(s.name for s in signals if s.hit) if signals else ""
        return self._remember(key, tier, turn,
                              f"heuristic {score:.1f} -> {tier} [{fired or 'none'}]")

    def _too_big_for_low(self, body_bytes: int) -> bool:
        if not self.low_context_tokens or not body_bytes:
            return False
        return body_bytes / CHARS_PER_TOKEN > self.low_context_tokens * CONTEXT_HEADROOM

    def _remember(self, key: str, tier: Tier, turn: TurnKind, why: str) -> PassDecision:
        with self._lock:
            self._pins[key] = tier
            self._pins.move_to_end(key)
            while len(self._pins) > MAX_PINS:
                self._pins.popitem(last=False)
        return PassDecision(self._model(tier), tier, turn, why)

    def _model(self, tier: Tier) -> str:
        return self.low if tier is Tier.LOW else self.high


# Low models measured to reject what a client sends when shaped for its main
# model, and what they rejected. Claude Code on Opus 5 sends adaptive thinking,
# an effort level and mid-conversation system messages; Haiku 4.5 supports none.
_INCOMPATIBLE_LOW = {
    ("anthropic", "claude-haiku-4-5"): (
        "adaptive thinking, the effort parameter and role 'system' messages",
        "claude-sonnet-5",
    ),
}


def downgrade_warning(family: str, low: str) -> str | None:
    """A startup warning when the low model is known to reject the client's requests."""
    bare = low.split("@", 1)[0].replace("[1m]", "")
    found = _INCOMPATIBLE_LOW.get((family, bare))
    if not found:
        return None
    rejected, instead = found
    return (
        f"{low} rejects {rejected}, which Claude Code sends when it believes it is on "
        f"its main model. Claude Code retries after stripping each one, so every "
        f"routed-down message costs three failed round trips first. "
        f"{instead} accepts the same requests unchanged."
    )
