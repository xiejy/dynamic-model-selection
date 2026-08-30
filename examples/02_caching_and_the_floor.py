"""2. Prompt caching -- the biggest single lever, and its silent failure mode.

Cache reads cost 0.10x input. Cache writes cost 1.25x (5m TTL) or 2.00x (1h).
Break-even is two requests on 5m, three on 1h.

The trap: the minimum cacheable prefix is model-dependent and NOT monotonic with
price. Opus 5 caches from 512 tokens; Haiku 4.5 needs 4096. Below the floor
nothing errors -- cache_creation_input_tokens is simply 0, forever.
"""
from _shared import banner, usd

from dms.levers.caching import agent_loop_cost
from dms.pricing import PriceBook

banner("caching, and the floor that silently swallows it", exact=True)

book = PriceBook.load()

print("minimum cacheable prefix, by model:")
for model in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"):
    print(f"  {model:<20} {book.min_cache_prefix_tokens(model):>5} tokens")

print("\na 3000-token system prompt, 20-step loop:")
for model in ("claude-opus-5", "claude-haiku-4-5"):
    plain = agent_loop_cost(book, label="", model=model, steps=20,
                            system_tokens=3000, caching=False)
    cached = agent_loop_cost(book, label="", model=model, steps=20,
                             system_tokens=3000, caching=True)
    saved = 1 - float(cached.cost_usd / plain.cost_usd)
    print(f"  {model:<20} uncached={usd(plain.cost_usd)}  "
          f"cached={usd(cached.cost_usd)}  saved={saved:+6.1%}  "
          f"[{cached.cache_status}]")

print("\nsame prompt padded to 5000 tokens, so it clears Haiku's floor:")
big = agent_loop_cost(book, label="", model="claude-haiku-4-5", steps=20,
                      system_tokens=5000, caching=True)
print(f"  claude-haiku-4-5     cached={usd(big.cost_usd)}  [{big.cache_status}]")
print("\n=> asking for caching is not the same as getting it. Check "
      "cache_creation_input_tokens.")
