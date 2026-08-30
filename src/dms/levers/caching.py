"""Prompt caching, agent-loop cost growth, and the routing/caching tension.

Everything here is **exact arithmetic over documented pricing mechanics**, not a
simulation of model behaviour. Given a token profile, these costs are what the
API bills. That makes this module's conclusions stronger than the benchmark's:
they do not depend on how well any model performs.

The three mechanics that compose into the headline result:

1. Cache reads cost 0.10x the input rate; cache writes cost 1.25x (5m TTL) or
   2.00x (1h TTL). Break-even is 2 requests on 5m, 3 on 1h.
2. **A model switch invalidates the cache completely** -- tools, system and
   messages tiers all. Tool edits and system-prompt edits have escape hatches;
   a model switch does not, because caches are model-scoped.
3. The minimum cacheable prefix is model-dependent and **not monotonic with
   price**: Opus 5 caches from 512 tokens, Haiku 4.5 needs 4096. Below the floor
   nothing errors -- `cache_creation_input_tokens` is just 0, forever.

Put together: on a cache-warm workload, routing "down" to the cheap model can
cost more than not routing at all. This is the same force behind LiteLLM's
`session_affinity` setting, which pins a session's first-turn model so the
provider cache survives follow-up turns.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from dms.pricing import CacheTTL, PriceBook
from dms.usage import UsageRecord


@dataclass(frozen=True, slots=True)
class LoopCost:
    """Cost of running a multi-step agent loop under one configuration."""

    label: str
    model: str
    steps: int
    cost_usd: Decimal
    prompt_tokens_billed: int
    cache_read_tokens: int
    cache_write_tokens: int
    cached_effectively: bool
    caching_requested: bool = False

    @property
    def cache_status(self) -> str:
        """Distinguishes 'did not ask' from 'asked and was silently ignored'."""
        if self.cached_effectively:
            return "cached"
        return "SILENT NO-OP" if self.caching_requested else "not requested"

    @property
    def cache_hit_rate(self) -> float:
        if self.prompt_tokens_billed == 0:
            return 0.0
        return self.cache_read_tokens / self.prompt_tokens_billed


def agent_loop_usage(
    *,
    steps: int,
    system_tokens: int,
    per_step_tokens: int,
    output_tokens: int,
    caching: bool,
) -> list[UsageRecord]:
    """Token profile of a naive agent loop, step by step.

    Every step re-sends the whole history, so billed input grows quadratically
    while the conversation grows linearly: 20 steps at 1k tokens/step bills
    ~210k input tokens, not 20k. This is the single biggest cost driver in
    agentic work, and it is why caching and context editing dominate model
    choice there.
    """
    records: list[UsageRecord] = []
    for step in range(steps):
        history_tokens = per_step_tokens * step
        fresh_tokens = per_step_tokens

        if not caching:
            records.append(
                UsageRecord(
                    input_tokens=system_tokens + history_tokens + fresh_tokens,
                    output_tokens=output_tokens,
                )
            )
            continue

        # With a breakpoint after the stable prefix: pay the write once, then
        # read the prefix and history at 0.1x, and only the new turn at full price.
        cacheable = system_tokens + history_tokens
        records.append(
            UsageRecord(
                input_tokens=fresh_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cacheable if step == 0 else 0,
                cache_read_input_tokens=cacheable if step > 0 else 0,
            )
        )
    return records


def agent_loop_cost(
    book: PriceBook,
    *,
    label: str,
    model: str,
    steps: int,
    system_tokens: int,
    per_step_tokens: int = 1000,
    output_tokens: int = 300,
    caching: bool = False,
    cache_ttl: CacheTTL = CacheTTL.FIVE_MINUTES,
) -> LoopCost:
    """Total cost of the loop, honouring this model's cache floor.

    If `caching` is requested but the prefix is below the model's minimum, the
    cache silently does not engage -- and this function reproduces that silence
    rather than pretending it worked.
    """
    effective = caching and book.will_cache(model, system_tokens)
    records = agent_loop_usage(
        steps=steps,
        system_tokens=system_tokens,
        per_step_tokens=per_step_tokens,
        output_tokens=output_tokens,
        caching=effective,
    )
    total = sum(
        (book.cost_usd(record, model, cache_ttl=cache_ttl) for record in records),
        Decimal(0),
    )
    return LoopCost(
        label=label,
        model=model,
        steps=steps,
        cost_usd=total,
        prompt_tokens_billed=sum(record.prompt_tokens for record in records),
        cache_read_tokens=sum(record.cache_read_input_tokens for record in records),
        cache_write_tokens=sum(record.cache_creation_input_tokens for record in records),
        cached_effectively=effective,
        caching_requested=caching,
    )


def alternating_model_loop_cost(
    book: PriceBook,
    *,
    label: str,
    models: tuple[str, ...],
    steps: int,
    system_tokens: int,
    per_step_tokens: int = 1000,
    output_tokens: int = 300,
    cache_ttl: CacheTTL = CacheTTL.FIVE_MINUTES,
) -> LoopCost:
    """A loop that reroutes between models every step.

    Caches are model-scoped and a switch has no escape hatch, so every reroute
    forfeits the 0.10x read AND re-pays the 1.25x write. Rerouting is not free
    even when the model you route to is cheaper.
    """
    total = Decimal(0)
    billed = read = written = 0
    warmed: set[str] = set()

    for step in range(steps):
        model = models[step % len(models)]
        prefix_tokens = system_tokens + per_step_tokens * step
        caches_at_all = book.will_cache(model, system_tokens)
        warm = model in warmed and caches_at_all

        if warm:
            record = UsageRecord(
                input_tokens=per_step_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=prefix_tokens,
            )
        elif caches_at_all:
            # First time on this model, or the cache was invalidated by the
            # switch away and back: pay the write premium again.
            record = UsageRecord(
                input_tokens=per_step_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=prefix_tokens,
            )
        else:
            # Below this model's floor: no cache exists, full price forever.
            record = UsageRecord(
                input_tokens=per_step_tokens + prefix_tokens,
                output_tokens=output_tokens,
            )

        if caches_at_all:
            warmed.add(model)

        total += book.cost_usd(record, model, cache_ttl=cache_ttl)
        billed += record.prompt_tokens
        read += record.cache_read_input_tokens
        written += record.cache_creation_input_tokens

    return LoopCost(
        label=label,
        model=" + ".join(models),
        steps=steps,
        cost_usd=total,
        prompt_tokens_billed=billed,
        cache_read_tokens=read,
        cache_write_tokens=written,
        cached_effectively=read > 0,
        caching_requested=True,
    )


def tension_scenarios(
    book: PriceBook,
    *,
    steps: int = 20,
    system_tokens: int = 3000,
    per_step_tokens: int = 1000,
) -> tuple[LoopCost, ...]:
    """The comparison that carries the talk.

    Default prefix is 3000 tokens on purpose: above Opus 5's 512-token floor and
    below Haiku 4.5's 4096-token floor. It is the size at which the expensive
    model caches and the cheap one silently does not.
    """
    common = {
        "steps": steps,
        "system_tokens": system_tokens,
        "per_step_tokens": per_step_tokens,
    }
    return (
        agent_loop_cost(
            book, label="Opus, no caching", model="claude-opus-5", caching=False, **common
        ),
        agent_loop_cost(
            book, label="Opus, cached", model="claude-opus-5", caching=True, **common
        ),
        agent_loop_cost(
            book, label="Haiku, no caching", model="claude-haiku-4-5", caching=False, **common
        ),
        agent_loop_cost(
            book,
            label="Haiku, caching REQUESTED",
            model="claude-haiku-4-5",
            caching=True,
            **common,
        ),
        alternating_model_loop_cost(
            book,
            label="Rerouting Opus<->Haiku each step",
            models=("claude-opus-5", "claude-haiku-4-5"),
            **common,
        ),
    )


def crossover_step(
    book: PriceBook,
    *,
    cheap_model: str = "claude-haiku-4-5",
    expensive_model: str = "claude-opus-5",
    system_tokens: int = 3000,
    per_step_tokens: int = 1000,
    max_steps: int = 500,
) -> int | None:
    """First loop length at which the CACHED expensive model beats the UNCACHED
    cheap one.

    This exists because the crossover is the least intuitive result in the repo
    and people will not believe it without being able to recompute it. Cache
    reads are 0.10x input, so a cached Opus 5 prefix effectively costs
    $0.50/MTok against Haiku 4.5's uncached $1.00/MTok. Past the crossover the
    flagship model is literally the cheaper way to run the loop.

    Returns None if no crossover occurs within `max_steps`.
    """
    for steps in range(2, max_steps + 1):
        expensive = agent_loop_cost(
            book,
            label="expensive-cached",
            model=expensive_model,
            steps=steps,
            system_tokens=system_tokens,
            per_step_tokens=per_step_tokens,
            caching=True,
        )
        cheap = agent_loop_cost(
            book,
            label="cheap-uncached",
            model=cheap_model,
            steps=steps,
            system_tokens=system_tokens,
            per_step_tokens=per_step_tokens,
            caching=False,
        )
        if expensive.cost_usd < cheap.cost_usd:
            return steps
    return None


def quadratic_growth(
    book: PriceBook,
    *,
    model: str = "claude-opus-5",
    per_step_tokens: int = 1000,
    system_tokens: int = 3000,
    checkpoints: tuple[int, ...] = (1, 5, 10, 20, 40),
) -> tuple[tuple[int, int, Decimal], ...]:
    """(steps, billed input tokens, cost) so the quadratic curve is visible.

    Conversation length grows linearly; billed input grows with the square,
    because every step re-sends everything before it. Naive per-step estimates
    understate a 20-step loop by roughly an order of magnitude.
    """
    rows = []
    for steps in checkpoints:
        records = agent_loop_usage(
            steps=steps,
            system_tokens=system_tokens,
            per_step_tokens=per_step_tokens,
            output_tokens=300,
            caching=False,
        )
        cost = sum((book.cost_usd(r, model) for r in records), Decimal(0))
        rows.append((steps, sum(r.prompt_tokens for r in records), cost))
    return tuple(rows)
