# dynamic-model-selection — what actually saves tokens

A measurement harness for LLM cost levers, built for an internal engineering share.

It exists because the published numbers do not survive contact with a 2026 Claude
workload. FrugalGPT's famous **98%** is one narrow classification dataset (the paper's
own range is 59–98%). RouteLLM's **>85%** is MT-Bench with a 2024 GPT-4-vs-Mixtral pair
whose price gap was 25–50×; Haiku→Opus today is **5×**. Independent benchmarks land far
lower — RouterArena ~**35%** at <2% accuracy loss, LLMRouterBench **31.7%** at parity.

So: rather than quote a number, measure one.

> **Status: measured.** 555 live calls, **$0.2572** of real API spend, Haiku 4.5 vs Opus 5
> on 36 tasks. Responses are recorded in `fixtures/`. The lever arithmetic is exact
> (published rates); the accuracy figures carry the confidence intervals shown.

---

## The answer: is a high/low dispatcher worth it?

`uv run dms bench --two-tier` — 36 tasks, Haiku 4.5 (low) vs Opus 5 (high), every dispatcher
charged for its own tokens.

| strategy | % to high | accuracy | 95% CI | cost | vs always-high | overhead |
|---|---:|---:|---:|---:|---:|---:|
| always: Haiku | 0% | 83.3% | 71–96% | $0.0041 | −88.0% | 0% |
| **cascade** | 14% | **97.2%** | 92–100% | **$0.0140** | **−59.0%** | 31.9% |
| heuristic | 58% | 97.2% | 92–100% | $0.0224 | −34.3% | 0% |
| lexical | 58% | 86.1% | 75–97% | $0.0236 | −30.7% | 0% |
| LLM classifier | 67% | 94.4% | 87–100% | $0.0281 | −17.5% | 18.1% |
| always: Opus | 100% | 91.7% | 83–100% | $0.0340 | — | 0% |

**Three results, in order of importance.**

**1. Only 19% of the workload is routable.** The two models return the same verdict on
81% of tasks (28 both-right, 1 both-wrong). Where they agree, a dispatcher can change cost
but never quality. That share is a hard ceiling on dispatch value and it is a property of
the *workload*, not the router. Of the 7 disagreements, 5 favour Opus (h01, h05, h12, m05,
m09) and **2 favour Haiku** (e09, m03) — the expensive model is not uniformly better.

**2. No quality difference is statistically detectable — including the premise.** Paired
exact-binomial McNemar over per-task correctness:

| comparison | A wins | B wins | p | conclusion |
|---|---:|---:|---:|---|
| **always-Opus vs always-Haiku** | 5 | 2 | **0.453** | **high model not measurably better** |
| cascade vs always-Opus | 3 | 1 | 0.625 | no detectable difference |
| heuristic vs always-Opus | 2 | 0 | 0.500 | no detectable difference |
| cascade vs always-Haiku | 5 | 0 | 0.062 | no detectable difference |
| cascade vs heuristic | 1 | 1 | 1.000 | no detectable difference |

At n=36 the resolution limit is ~14pp. Detecting a 5pp difference at this 19% disagreement
rate needs **~584 tasks**; 2pp needs **~3,712**. So the defensible claim is *"59% cheaper
with no detected quality regression"* — not *"cheaper and better"*.

**3. Decide after the cheap answer, not before it.** The pre-request routers converge on
"when unsure, send it high" — two of them pushed 58% of traffic to Opus and kept a third of
the saving. The cascade routed only **14%** high and saved **59%**, while paying for the
discarded cheap answer *and* a verification call on every request (31.9% of its budget
bought no answer). Routing needs a good prediction of quality *before* spending; cascading
needs a judgement *after* one cheap answer exists — the second problem is much easier.
The LLM classifier was the worst deal: 18.1% of its budget spent deciding, 67% of traffic
sent high anyway, 17.5% saved. The lexical router scored *worse than a coin* at its own rate.

**What bit us in 555 calls** — all three grade as "the model got it wrong" unless
instrumented for:
- Opus 5 returned `stop_reason: "refusal"` (category `cyber`) on *"what exit code does a
  POSIX shell return when a command is not found?"* — a benign question Haiku answered fine.
  Tier dispatch changes your refusal surface, not just your bill.
- Opus 5 thinks by default and `max_tokens` caps thinking **plus** text; a 512 cap truncated
  the expensive model into looking incompetent.
- Our first heuristic keyed on *"answer with the number only"* — present at every difficulty
  — and routed 100% of traffic to Haiku while reading fine in review.

---

## The three findings

### 1. Caching beats model choice, and eventually inverts it

Cache reads cost **0.10×** input. So a cached Opus 5 prefix effectively costs
**$0.50/MTok** against Haiku 4.5's uncached **$1.00/MTok** — and a model switch
invalidates the cache completely, with no escape hatch, because caches are model-scoped.

```
 steps   Opus (cached)   Haiku (uncached)  cheaper
    20       $0.392250          $0.300000  haiku
    30       $0.654750          $0.600000  haiku
    37       $0.868250          $0.869500  OPUS      <- crossover
    97       $3.703250          $5.189500  OPUS
```

**From step 37, the 5×-more-expensive model is the cheaper way to run the loop.**
Fix caching before arguing about model choice.

### 2. The cheap model is the *harder* one to cache

The minimum cacheable prefix is model-dependent and **not monotonic with price**:

| model | min cacheable prefix |
|---|---:|
| Claude Opus 5 | 512 |
| Claude Sonnet 5 | 1 024 |
| **Claude Haiku 4.5** | **4 096** |

A 3 000-token system prompt caches on Opus 5 and **silently does not** on Haiku 4.5 —
no error, `cache_creation_input_tokens` is just `0`:

```
claude-opus-5     uncached=$1.500000  cached=$0.392250  saved=+73.8%  [cached]
claude-haiku-4-5  uncached=$0.300000  cached=$0.300000  saved= +0.0%  [SILENT NO-OP]
```

### 3. Agent-loop cost is quadratic, not linear

Every step re-sends the whole history, so billed input grows with the *square* of steps:

| steps | billed input | naive estimate | error |
|---|---:|---:|---:|
| 5 | 30 000 | 20 000 | 1.5× |
| 20 | 270 000 | 80 000 | **3.4×** |
| 80 | 3 480 000 | 320 000 | **10.9×** |

("naive" = what a reasonable person computes: 3k system + 1k new per step, times
steps. Already generous, and still off by 10× at 80 steps.)

This is the term that dominates agentic spend. It is why caching and context editing
matter more there than which model you picked.

---

## Levers, ranked

Reference workload: a 20-step agent loop, 3k-token system prompt, 1k new tokens/step,
on Opus 5. **Exact arithmetic** — change the shape and the ranking moves, which is the
point. There is no universal ordering, only an ordering for a workload shape.

| lever | cost | saving | trade-off |
|---|---:|---:|---|
| route to Haiku (no caching) | $0.3000 | **+80.0%** | quality drop on hard tasks; 4096-token cache floor |
| caching + context editing | $0.3242 | +78.4% | agent loses older tool results |
| prompt caching | $0.3922 | +73.9% | prefix must be byte-stable and above the floor |
| Batch API | $0.7500 | +50.0% | async, up to 24 h; no fallbacks |
| context editing (keep 3 turns) | $0.8200 | +45.3% | agent loses older tool results |
| do nothing | $1.5000 | — | — |

`output_config.effort` is **deliberately absent**: the docs publish no per-level token
ratio, so any number here would be invented. Anthropic's own Opus 5 guidance is to start
at `xhigh` for agentic work and *sweep downward*, because `low`/`medium` "punch well
above their weight". That is an instruction to measure, not a figure to quote. Related
trap: **disabling thinking is the more expensive lever** — prefer adaptive thinking at
lower effort.

---

## Routing, measured honestly

Three accounting rules the harness enforces, each of which some published savings figure
quietly breaks:

1. **Router tokens are charged to the strategy that used them.** A classifier runs on
   100% of traffic to downgrade a fraction of it.
2. **The baseline is random routing at the same strong-model call fraction** — not
   always-strong, which any router beats trivially. This is RouteLLM's own baseline.
3. **A cascade pays for the cheap attempt it throws away**, and for the verification call
   on every task.

Break-even for an LLM classifier (exact, from `examples/05_router_economics.py`) — note
the router must read the *whole* prompt to classify it, so its cost scales with prompt
size:

| classify with | route | short | medium | long |
|---|---|---:|---:|---:|
| Haiku 4.5 | Opus → Haiku | 7.9% | 11.2% | 16.7% |
| Haiku 4.5 | Opus → Sonnet | 10.5% | 15.0% | 22.2% |
| Sonnet 5 | Opus → Sonnet | 21.0% | 29.9% | **44.5%** |

(the share of traffic you must downgrade for the router to clear its own cost)

Against a 5× tier gap the economics are fine and the real risk is **quality, not cost**.
Between adjacent tiers, or with an expensive classifier, the margin collapses. A
zero-token heuristic has no break-even to clear at all — which is why the most-deployed
production router (LiteLLM's Auto Router default) is a bag of regexes.

### What the literature says, honestly

| source | claim | evidence |
|---|---|---|
| FrugalGPT (TMLR 2024) | 59–98% cost cut | peer-reviewed; 98% is one narrow dataset |
| RouteLLM (ICLR 2025) | >85% MT-Bench, 45% MMLU | peer-reviewed; 2024 model pair |
| Hybrid LLM (ICLR 2024) | 40% fewer big-model calls at no quality drop | peer-reviewed |
| AutoMix (NeurIPS 2024) | >50% at parity | peer-reviewed |
| **RouterArena (2025)** | **~35% at <2% accuracy loss** | independent — quote this one |
| **LLMRouterBench (2026)** | **31.7% at parity** | independent, 33 models, 21 datasets |
| Azure Model Router | 4.5–14.2% measured | third-party run vs "up to 60%" marketing |
| LiteLLM | "51%", "69% with caching" | vendor claim, no methodology |
| Martian | "up to 98%" | marketing, uncitable |

And the findings that should temper any router project:

- **Routing collapse** (arXiv 2602.03478): routers send ~100% of queries to the top model
  under loose budgets where the oracle needs it <20% of the time. **94.9% of queries have
  inter-model margins ≤0.05** — the decision is fragile by construction.
- LLMRouterBench measured **OpenRouter at −24.7% versus just picking the best single
  model**. A commercial router that loses to no router at all.
- Method innovation has **plateaued** — leading methods are "broadly comparable", and a
  22.7M-parameter embedder performs about as well as large ones.
- **Silent quality regression.** A degraded answer is well-formed. Nothing alerts. A 3×
  cheaper model that fails 20% of the time costs more than the expensive one that doesn't,
  once retries and escalations are counted.

---

## Quickstart

Needs [uv](https://docs.astral.sh/uv/). No API key required for anything below.

```bash
uv sync && uv pip install -e .

uv run dms levers          # exact lever arithmetic — the strongest numbers here
uv run dms bench --two-tier  # the high/low dispatch verdict
uv run dms bench           # full three-tier strategy benchmark
uv run dms tasks           # show the workload mix
uv run pytest              # 133 tests, hermetic, offline
```

Individual demos:

```bash
uv run python examples/02_caching_and_the_floor.py    # the silent no-op
uv run python examples/03_agent_loop_quadratic.py     # quadratic growth
uv run python examples/04_routing_vs_caching.py       # the crossover
uv run python examples/05_router_economics.py         # router break-even
uv run python examples/06_bench.py                    # full bench
```

## Running it for real

`simulate` mode exists so the repo is runnable and testable with no credential. Its
model-behaviour numbers are **estimates** from `config/simulation.json`, and every
simulated output is labelled as such. To replace them with measurements:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run dms bench --mode record --max-spend 5.00   # live, writes fixtures/
uv run dms bench --mode replay                    # deterministic offline replay
```

`record` writes every response to `fixtures/`, so the talk demo replays identically with
no network — if the conference wifi dies, the demo still works. The `--max-spend` guard
applies in **every** mode, so you can dry-run a spend plan offline before pointing it at
a real key.

The lever arithmetic (`dms levers`, examples 02–05) needs no key at all and is exact
either way.

## Layout

```
config/model-pricing.json   dated price snapshots (Decimal strings, not floats)
config/simulation.json      the simulate-mode ASSUMPTIONS, in one editable file
tasks/workload.jsonl        36 dev tasks, 12 each easy/medium/hard, checkable answers
tasks/exemplars.jsonl       router fitting set — deliberately NOT the workload
src/dms/pricing.py          exact cost arithmetic incl. cache write/read multipliers
src/dms/routers/            always | random | heuristic | lexical | llm-classifier
src/dms/strategies.py       routed | cascade | oracle
src/dms/levers/             caching, context editing, batch, the crossover
src/dms/metrics.py          PGR, random-baseline comparison, Pareto frontier
```

Two deliberate choices worth noting:

- **Money is `Decimal`, never `float`** — matching the approach in
  `~/workspace/.workspace/ai-metrics/scripts/pricing.py`. Prices are decimal *strings* in
  JSON and validated as such at load.
- **Prices are dated snapshots**, because the Sonnet 5 introductory rate ($2/$10) lapses
  **2026-08-31** and reverts to $3/$15. A flat table silently under-reports every cost
  computed from September onward.

## Caveats

- The workload grades **short-answer correctness**, not open-ended response quality.
  Routing looks better on open-ended chat and worse on knowledge-dense work, so a
  short-answer suite sits nearer the pessimistic end. Do not generalise these numbers to
  a chat product.
- 36 tasks is enough to show mechanism, not enough for a confidence interval.
- Simulated accuracy figures depend entirely on `config/simulation.json`. Change a row,
  rerun, and watch which conclusions move — a conclusion that flips when Haiku's hard-task
  accuracy shifts by 0.05 was never a conclusion.

---

## The proxy

`dms proxy` puts the measured dispatcher in front of real traffic. Callers change
`base_url` and nothing else.

```bash
export ANTHROPIC_API_KEY=...
uv run dms proxy                       # 127.0.0.1:8787, cascade, affinity on
uv run dms proxy --strategy heuristic  # zero-token routing instead
uv run dms proxy --low claude-haiku-4-5 --high gpt-5.6-sol   # mixed providers
```

| endpoint | dialect | who points at it |
|---|---|---|
| `POST /v1/messages` | Anthropic Messages | Claude SDKs, Claude Code |
| `POST /v1/chat/completions` | OpenAI Chat | **Codex CLI**, OpenAI SDKs |
| `GET /healthz` · `GET /stats` | — | ops |

```bash
# Anthropic client
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude ...
# Codex / OpenAI client
OPENAI_BASE_URL=http://127.0.0.1:8787/v1 codex ...
```

**Verified live.** A caller asking for `claude-opus-5` got answered by Haiku for
**$0.000135** after the verifier accepted it. A consistency-guarantee question went
`haiku → verifier said no → opus`, returning the right answer for $0.002428 with
$0.000198 of overhead. Streaming, session affinity and `/stats` all confirmed against
real models.

Every response carries a `dms_dispatch` block naming the model that actually answered,
why, the per-leg spend, and the overhead — without it you cannot detect the silent
regression this repo spends its whole README warning about.

### Design decisions, each traceable to a measurement

- **Cascade is the default** — 59% cheaper than always-high, escalating 14% against a
  true need of 14%.
- **Streaming falls back to the heuristic.** The verifier needs the complete cheap
  answer, so a streamed cascade would buffer everything and destroy time-to-first-token.
  A real trade, not a gap.
- **Session affinity is on** (1h TTL). A model switch invalidates the prompt cache with
  no escape hatch, so a reroute forfeits the 0.10× read *and* re-pays the 1.25× write.
- **A refusal retries the other tier.** Measured: Opus 5 refused a benign POSIX shell
  question Haiku answered fine. When the escalation target refuses, the cheap answer
  already paid for is returned rather than discarded.
- **An unreadable verdict escalates.** Fail toward quality; never ship an answer nobody
  vouched for.
- **Routers score only user turns**, so the decision does not drift as the assistant's
  own output accumulates.

### Driving it from Codex CLI

Codex 0.146 **removed `wire_api = "chat"`** and speaks only the Responses API, so the
proxy implements `POST /v1/responses` as well. Two more things it needs: `GET /v1/models`
(it probes on startup) and tolerance for `role: "developer"`, which Anthropic rejects and
which the proxy folds into the system prompt.

```bash
export DMS_API_KEY=not-checked-by-the-proxy   # the proxy does not authenticate
echo "your prompt" | codex exec \
  -c model_providers.dms.name=dms \
  -c model_providers.dms.base_url=http://127.0.0.1:8787/v1 \
  -c model_providers.dms.wire_api=responses \
  -c model_providers.dms.env_key=DMS_API_KEY \
  -c model_provider=dms -c model=gpt-5.6-sol
```

**What running Codex through it revealed** — the repo's own argument, on real traffic:

| Codex turn | prompt | uncached | cache write | cache read | cost |
|---|---:|---:|---:|---:|---:|
| 1st | 13,443 | 1,932 | 11,511 | 0 | $0.0817 |
| **2nd** | 13,443 | 1,932 | 0 | **11,511** | **$0.0155** |

Codex resends a ~13k-token harness prompt on **every turn**. Uncached that was $0.069 a
turn and dwarfed anything model choice could recover — the heuristic routed it to Opus
regardless, because a 13k prompt trips every length signal there is. Turning on prompt
caching cut the steady-state turn to **$0.0155, about 78% off, with no model change.**

Note the first turn costs *more* than uncached ($0.0817 vs $0.069): the 1.25× write
premium. Break-even is the second request, exactly as the 5-minute-TTL arithmetic says.
If your agent client sends one turn and leaves, caching loses.

### Codex as a *backend*, with no API key

`codex exec --json` authenticates through your ChatGPT login and emits JSONL carrying both
things a provider needs: the answer on an `item.completed` / `agent_message` event, and
token counts on `turn.completed`. So GPT is reachable as a tier with **no
`OPENAI_API_KEY`**:

```bash
uv run dms proxy --low claude-haiku-4-5 --high codex-cli/gpt-5.6-sol
```

Model ids are namespaced `codex-cli/<model>` so they can never be confused with API ids —
one needs a ChatGPT login, the other a metered key, and a silent mix-up would call the
wrong backend and report the wrong cost. Costs are computed from the underlying model's
published rates: **what the equivalent API call would cost**, useful for comparison, not a
bill.

**Verified live, cross-provider** — Haiku low tier, GPT-5.6-sol high tier:

| | answered by | result | prompt tokens | cost | latency |
|---|---|---|---:|---:|---:|
| forced high | `codex-cli/gpt-5.6-sol` | **5** ✅ | 16,792 (11,008 cached) | $0.0368 | 13.6s |
| cascade, same question | `claude-haiku-4-5` | 4 ❌ | 1,200 | $0.0002 | 1.5s |

The cross-provider escalation gets right what the same-provider cascade got wrong — and
costs 180× more and takes 9× longer to do it.

**It is an agent, not a completion endpoint**, and that shapes everything:
- it carries a ~17k-token harness, so a short question is never a short request;
- it *runs tools* — an observed run shelled out to `sed` before answering. This provider
  forces `--sandbox read-only` and never forwards caller-supplied tool definitions, but it
  cannot make Codex stop being an agent;
- latency is ~10–14s, dominated by process start and the agent loop, not the model;
- streaming is chunk-per-message, because Codex's JSONL carries completed items rather
  than token deltas.

Use it to reach GPT without a key or to compare tiers across providers. For
latency-sensitive traffic the API path (`OpenAIProvider`) is the right one — though that
path is still **untested against real GPT traffic**, since no `OPENAI_API_KEY` is set here.

### Multi-provider

GPT-5.x/Codex rates were mirrored into `config/model-pricing.json` from
`~/workspace/.workspace/ai-metrics` snapshot `2026-07-31-a35a041dfafc`, so the two
tables agree by construction. The price book is provider-aware: OpenAI's explicit
`cached_input_usd_per_million` is used where present rather than Anthropic's 0.10×
multiplier, and OpenAI cache writes carry no premium. `prompt_tokens` includes cached
tokens on OpenAI and excludes them on Anthropic — the adapter subtracts, or the cached
portion gets billed twice.

The OpenAI adapter is stdlib `urllib`, so pointing `base_url` at OpenRouter, Azure or a
local server needs no code change. Live GPT calls need `OPENAI_API_KEY`; it is not set
here, so that path is covered by tests but has not been exercised against real GPT
traffic.

### Limits

`ThreadingHTTPServer`: a thread per request, fine for an internal service, not for high
concurrency — swap this one module for an ASGI app, everything below `Dispatcher` is
transport-agnostic. Session pins are in-process, so a multi-replica deployment needs a
shared store or sticky routing. Tool-use blocks pass through but tool *results* are not
re-dispatched.

## Slides

The share deck lives at `slides.html` (also published as an Artifact). Every figure in it
traces to `dms levers`, `dms bench`, or the citation table above, and every data block
carries an evidence badge: **exact** (arithmetic over published rates), **simulated**
(pending the live pass), or **literature** (someone else's number).
