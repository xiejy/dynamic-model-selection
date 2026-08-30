"""5. Does the router pay for itself? Do this arithmetic before building one.

An LLM classifier runs on 100% of traffic to downgrade some fraction of it. It
is worth it only when:

    router_cost_per_request  <  downgrade_share x (strong_cost - weak_cost)

The detail people get wrong: **the classifier has to read the whole prompt** to
classify it, so its input cost scales with the prompt, not with the one word it
replies. Modelling it as a fixed small cost flatters it.

The honest finding below is not "LLM routers never pay" -- against a Haiku
classifier and a 5x Opus/Haiku gap they pay easily. It is that the margin
collapses when the tiers you are choosing between are close together, and it
inverts entirely if you classify with a model as expensive as the one you are
trying to avoid.
"""
from decimal import Decimal

from _shared import banner, usd

from dms.pricing import PriceBook
from dms.usage import UsageRecord

banner("router break-even", exact=True)

book = PriceBook.load()
CLASSIFIER_OUTPUT_TOKENS = 4  # one tier word

PROFILES = (
    ("short", 200, 100),
    ("medium", 2_000, 500),
    ("long", 20_000, 2_000),
)


def break_even(router_model: str, strong: str, weak: str) -> None:
    print(f"\nclassify with {router_model}   route {strong} -> {weak}")
    print(f"{'prompt':>8} {'router $':>11} {'gap $':>11} {'break-even share':>18}")

    for label, input_tokens, output_tokens in PROFILES:
        # The classifier reads the same prompt; it just answers in one word.
        router = book.cost_usd(
            UsageRecord(input_tokens=input_tokens, output_tokens=CLASSIFIER_OUTPUT_TOKENS),
            router_model,
        )
        answer = UsageRecord(input_tokens=input_tokens, output_tokens=output_tokens)
        gap = book.cost_usd(answer, strong) - book.cost_usd(answer, weak)

        share = router / gap if gap > 0 else Decimal("Infinity")
        note = "  <- impossible" if share >= 1 else ""
        print(f"{label:>8} {usd(router):>11} {usd(gap):>11} {share:>17.1%}{note}")


break_even("claude-haiku-4-5", "claude-opus-5", "claude-haiku-4-5")
break_even("claude-haiku-4-5", "claude-opus-5", "claude-sonnet-5")
break_even("claude-sonnet-5", "claude-opus-5", "claude-sonnet-5")

print(
    """
=> Against the 5x Opus/Haiku gap, a Haiku classifier needs to downgrade only
   ~8-17% of traffic to clear its own cost. The economics are fine; the real risk
   there is quality, not cost.
   Two things make it worse, both visible above: longer prompts (the router reads
   every token, so its cost scales while the saving per request does not scale as
   fast), and adjacent tiers (Opus->Sonnet saves less per downgrade). Classify
   with Sonnet instead of Haiku and the bar reaches 44% of traffic.
   A zero-token heuristic has no break-even to clear at all, which is why the
   most-deployed production router (LiteLLM's default) is a bag of regexes.
"""
)
