"""Deterministic grading.

Every grader here is a pure string function -- no model in the loop. That is a
deliberate constraint: an LLM-as-judge would add cost and variance to the very
measurement that is supposed to adjudicate cost and variance. The workload was
written so that terse, checkable answers are possible.

The honest limitation, and it belongs on a slide: this measures *correctness on
short-answer tasks*, not the open-ended response quality that routing papers
grade with MT-Bench-style judges. Routing looks better on open-ended chat and
worse on knowledge-dense work; a short-answer suite sits closer to the pessimistic
end. Do not generalise a number from here to "our chat product".
"""
from __future__ import annotations

import re
from collections.abc import Callable

Grader = Callable[[str, str], bool]

_FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*\n?|\n?```$")
_PUNCTUATION = re.compile(r"[\s`'\"*.,;:!?()\[\]]+")
_AFFIRMATIVE = {"yes", "y", "true", "yeah", "correct", "affirmative"}
_NEGATIVE = {"no", "n", "false", "nope", "incorrect", "negative"}


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation and markdown."""
    # Fences first: stripping stray backticks earlier would eat the fence and
    # leave the language tag ("```sql\n..." -> "sql ...") in the graded text.
    cleaned = _FENCE.sub("", text.strip()).strip()
    cleaned = cleaned.strip("`").strip()
    return _PUNCTUATION.sub(" ", cleaned.lower()).strip()


def exact_ci(answer: str, expected: str) -> bool:
    return normalise(answer) == normalise(expected)


def contains(answer: str, expected: str) -> bool:
    return normalise(expected) in normalise(answer)


def contains_any(answer: str, expected: str) -> bool:
    """`expected` is a pipe-delimited list of acceptable phrasings."""
    haystack = normalise(answer)
    return any(normalise(option) in haystack for option in expected.split("|"))


def yesno(answer: str, expected: str) -> bool:
    """Grade a boolean answer, tolerating phrasing but not ambiguity."""
    got, want = _polarity(answer), _polarity(expected)
    return want is not None and got == want


def sql_shape(answer: str, expected: str) -> bool:
    """All pipe-delimited fragments must appear. Structure, not exact SQL text."""
    haystack = normalise(answer)
    return all(normalise(part) in haystack for part in expected.split("|"))


def cron(answer: str, expected: str) -> bool:
    """Compare five cron fields, ignoring surrounding prose and whitespace."""
    fields = _cron_fields(answer)
    return fields is not None and fields == _cron_fields(expected)


GRADERS: dict[str, Grader] = {
    "exact_ci": exact_ci,
    "contains": contains,
    "contains_any": contains_any,
    "yesno": yesno,
    "sql_shape": sql_shape,
    "cron": cron,
}


def grade(answer: str, expected: str, grader: str) -> bool:
    """Score one answer. Unknown grader names are a configuration bug, not a miss."""
    try:
        fn = GRADERS[grader]
    except KeyError:
        raise KeyError(
            f"unknown grader {grader!r}; known: {', '.join(sorted(GRADERS))}"
        ) from None
    return fn(answer, expected)


# ------------------------------------------------------------------------ helpers


def _polarity(text: str) -> bool | None:
    """First yes/no-ish token in the text, or None if it says neither."""
    for word in normalise(text).split():
        if word in _AFFIRMATIVE:
            return True
        if word in _NEGATIVE:
            return False
    return None


def _cron_fields(text: str) -> tuple[str, ...] | None:
    tokens = text.strip().strip("`").split()
    for start in range(len(tokens) - 4):
        window = tuple(tokens[start : start + 5])
        if all(re.fullmatch(r"[\d*/,\-]+", token) for token in window):
            return window
    return None
