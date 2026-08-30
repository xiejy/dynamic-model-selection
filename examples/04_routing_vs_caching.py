"""4. The result this repo exists for: routing and caching are in tension.

Three documented mechanics compose into something counter-intuitive:

  1. a model switch invalidates the prompt cache completely, and unlike tool or
     system-prompt edits it has NO escape hatch -- caches are model-scoped;
  2. cache reads cost 0.10x input, so a cached Opus 5 prefix effectively costs
     $0.50/MTok against Haiku 4.5's uncached $1.00/MTok;
  3. Haiku's cache floor (4096) is 8x Opus 5's (512), so the cheap model is the
     HARDER one to cache.

Past a crossover length, the 5x-more-expensive model is the cheaper way to run
the loop. Fix caching before arguing about model choice.
"""
from _shared import banner, usd

from dms.levers.caching import agent_loop_cost, crossover_step, tension_scenarios
from dms.pricing import PriceBook

banner("routing vs caching", exact=True)

book = PriceBook.load()

print("20-step loop, 3k system prompt (above Opus's floor, below Haiku's):")
for scenario in tension_scenarios(book):
    print(f"  {scenario.label:<34} {usd(scenario.cost_usd):>12}  "
          f"hit={scenario.cache_hit_rate:5.1%}  [{scenario.cache_status}]")

crossover = crossover_step(book)
print(f"\ncrossover: step {crossover}")
print(f"{'steps':>6} {'Opus (cached)':>15} {'Haiku (uncached)':>18}  cheaper")
for steps in (10, 20, 30, crossover, crossover + 20, crossover + 60):
    opus = agent_loop_cost(book, label="", model="claude-opus-5", steps=steps,
                           system_tokens=3000, caching=True).cost_usd
    haiku = agent_loop_cost(book, label="", model="claude-haiku-4-5", steps=steps,
                            system_tokens=3000, caching=False).cost_usd
    print(f"{steps:>6} {usd(opus):>15} {usd(haiku):>18}  "
          f"{'OPUS' if opus < haiku else 'haiku'}")

print("\n=> this is why LiteLLM ships session_affinity: pinning the first-turn")
print("   model keeps the provider cache warm. A reroute can cost more than the")
print("   tier difference it saves.")
