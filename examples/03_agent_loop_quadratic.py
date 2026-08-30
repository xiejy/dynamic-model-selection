"""3. Why agent loops cost what they do: billed input grows with the SQUARE.

Every step re-sends the whole history. Conversation length grows linearly;
billed input tokens grow quadratically. A 20-step loop at 1k tokens/step bills
~210k input tokens, not 20k -- a 10.5x error if you estimate per-step.

This is the term that dominates agentic spend, and it is why caching and
context editing matter more than model choice there.
"""
from _shared import banner, usd

from dms.levers.caching import quadratic_growth
from dms.pricing import PriceBook

banner("quadratic cost growth in an agent loop", exact=True)

book = PriceBook.load()

SYSTEM_TOKENS, PER_STEP = 3000, 1000

# The naive estimate is what a reasonable person computes: "each step sends the
# 3k system prompt plus 1k of new content, so N steps is N x 4k." That is the
# comparison worth making -- it is already generous, and still off by 10x.
print(f"{'steps':>6} {'billed input':>14} {'naive estimate':>15} {'error':>8} {'cost':>10}")
for steps, billed, cost in quadratic_growth(book, checkpoints=(1, 5, 10, 20, 40, 80)):
    naive = steps * (SYSTEM_TOKENS + PER_STEP)
    print(f"{steps:>6} {billed:>14,} {naive:>15,} {billed / naive:>7.1f}x {usd(cost):>10}")

print("\n=> the fix is structural, not a cheaper model: cache the prefix, and")
print("   prune old tool results with context editing (clear_tool_uses_20250919).")
