"""1. What one call actually costs, and the field everyone misreads.

`usage.input_tokens` is the UNCACHED REMAINDER, not the prompt size. An agent
that ran for an hour can report input_tokens=4000 while having sent 400k. The
prompt total is the sum of three fields:

    prompt_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
"""
from _shared import banner, client, usd

from dms.pricing import PriceBook

banner("cost of a single call")

book = PriceBook.load()
api = client()

for model in ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"):
    call = api.complete(model=model, prompt="Name the capital of France.", max_tokens=32)
    u = call.usage
    print(
        f"{model:<20} in={u.input_tokens:<6} out={u.output_tokens:<5} "
        f"prompt_total={u.prompt_tokens:<6} cost={usd(book.cost_usd(u, model))}"
    )

rate = book.rate("claude-opus-5")
print(f"\nOpus 5 rate: ${rate.input_usd_per_million}/MTok in, "
      f"${rate.output_usd_per_million}/MTok out")
print(f"run total: {usd(api.total_spend_usd)} over {api.calls_made} calls")
