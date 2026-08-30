"""The rest of the cost toolkit, ranked against each other on one workload.

The point of ranking them together: "dynamic model selection" is one lever among
several, and on agentic workloads it is not the biggest one. A team that argues
for a month about a router while re-sending an uncached 3k-token system prompt
20 times per task is optimising the wrong term.

What is exact arithmetic here (documented multipliers, no behaviour assumed):
  * Batch API            -- exactly 0.50x on all token usage
  * Prompt caching       -- 0.10x reads, 1.25x/2.00x writes
  * Context editing      -- bounds the re-sent history, killing the quadratic

What is NOT quantified and must be measured on your own traffic:
  * output_config.effort -- the docs give no per-level token ratio. Anthropic's
    own guidance on Opus 5 is to start at xhigh for agentic work and *sweep
    downward*, because low and medium "punch well above their weight". That is
    an instruction to measure, not a number to quote.
  * prompt-side verbosity -- a short conciseness instruction measured ~20% off
    user-facing response length. `effort` is explicitly NOT the lever for output
    length.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from dms.levers.caching import agent_loop_cost, agent_loop_usage
from dms.pricing import PriceBook
from dms.usage import UsageRecord

BATCH_TRADEOFF = "async, up to 24h; no fallbacks; no max_tokens:0 pre-warm"


@dataclass(frozen=True, slots=True)
class Lever:
    """One cost lever applied to the reference workload."""

    name: str
    cost_usd: Decimal
    saving_vs_baseline: float
    exact: bool  # True = arithmetic from documented rates; False = needs measuring
    tradeoff: str


def context_edited_loop_usage(
    *,
    steps: int,
    system_tokens: int,
    per_step_tokens: int,
    output_tokens: int,
    keep_last_turns: int,
) -> list[UsageRecord]:
    """A loop that clears tool results older than `keep_last_turns`.

    This is what `context_management={"edits": [{"type": "clear_tool_uses_20250919"}]}`
    does: it *prunes* old tool results rather than summarising them. History
    stops growing, so billed input goes from quadratic to linear -- the single
    biggest structural win available on a long agent loop.

    (Compaction is the different, summarising feature -- `compact_20260112`,
    beta `compact-2026-01-12`. Conflating the two is a common mistake; the
    critical gotcha there is that you must append `response.content` back, not
    just the text, or the compaction state is silently lost.)
    """
    return [
        UsageRecord(
            input_tokens=(
                system_tokens
                + per_step_tokens * min(step, keep_last_turns)
                + per_step_tokens
            ),
            output_tokens=output_tokens,
        )
        for step in range(steps)
    ]


def rank_levers(
    book: PriceBook,
    *,
    model: str = "claude-opus-5",
    steps: int = 20,
    system_tokens: int = 3000,
    per_step_tokens: int = 1000,
    output_tokens: int = 300,
    keep_last_turns: int = 3,
) -> tuple[Lever, ...]:
    """Rank the levers by what they save on one reference agent loop.

    Reference workload: a 20-step agent loop with a 3k-token system prompt and
    1k tokens of new content per step, on Opus 5. Change the parameters and the
    ranking moves -- which is the point. There is no universal ordering, only
    an ordering for a workload shape.
    """
    common = dict(
        steps=steps,
        system_tokens=system_tokens,
        per_step_tokens=per_step_tokens,
        output_tokens=output_tokens,
    )

    baseline = agent_loop_cost(book, label="baseline", model=model, caching=False, **common)
    cached = agent_loop_cost(book, label="cached", model=model, caching=True, **common)

    edited_records = context_edited_loop_usage(keep_last_turns=keep_last_turns, **common)
    edited = sum((book.cost_usd(r, model) for r in edited_records), Decimal(0))

    batched_records = agent_loop_usage(caching=False, **common)
    batched = sum(
        (book.cost_usd(r, model, batch=True) for r in batched_records), Decimal(0)
    )

    # Caching + context editing compose: a bounded history, cached.
    both_records = context_edited_loop_usage(keep_last_turns=keep_last_turns, **common)
    both = Decimal(0)
    for index, record in enumerate(both_records):
        cacheable = record.input_tokens - per_step_tokens
        both += book.cost_usd(
            UsageRecord(
                input_tokens=per_step_tokens,
                output_tokens=record.output_tokens,
                cache_creation_input_tokens=cacheable if index == 0 else 0,
                cache_read_input_tokens=cacheable if index > 0 else 0,
            ),
            model,
        )

    cheap = agent_loop_cost(
        book, label="haiku", model="claude-haiku-4-5", caching=False, **common
    )

    def lever(name: str, cost: Decimal, exact: bool, tradeoff: str) -> Lever:
        saving = float((baseline.cost_usd - cost) / baseline.cost_usd)
        return Lever(name=name, cost_usd=cost, saving_vs_baseline=saving, exact=exact,
                     tradeoff=tradeoff)

    levers = (
        lever("do nothing (baseline)", baseline.cost_usd, True, "-"),
        lever("prompt caching", cached.cost_usd, True,
              "prefix must be byte-stable and above the model's floor"),
        lever("context editing (keep 3 turns)", edited, True,
              "the agent loses access to older tool results"),
        lever("Batch API", batched, True, BATCH_TRADEOFF),
        lever("route to Haiku (no caching)", cheap.cost_usd, True,
              "quality drop on hard tasks; cache floor is 4096 tokens"),
        lever("caching + context editing", both, True,
              "compose cleanly; do these before touching model choice"),
    )
    return tuple(sorted(levers, key=lambda item: item.cost_usd))


def effort_note() -> str:
    """Why there is no effort row in the ranking."""
    return (
        "output_config.effort is deliberately absent from the ranking: the docs\n"
        "publish no per-level token ratio, so any number here would be invented.\n"
        "Anthropic's own Opus 5 guidance is to start at xhigh for agentic work and\n"
        "sweep downward because low/medium 'punch well above their weight'. Run\n"
        "examples/04_effort_sweep.py against your own traffic to get a real figure.\n"
        "Related trap: disabling thinking is the MORE expensive lever -- prefer\n"
        "adaptive thinking at lower effort."
    )
