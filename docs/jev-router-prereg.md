# Jev as the model selector — pre-registration

Written **2026-09-23, before any Jev response exists.** Nothing below may be changed
after the first live number is seen; a changed bar is a different experiment.

## Question

Can TypeSafe's Jev (`jev-1.13.0`, a typed judgment model, not an LLM) decide
low-vs-high model dispatch better than the routers already measured here?

## Method

- **Items:** the 36 benchmark prompts in `tasks/workload.jsonl` — already public in this
  repo, so sending them to api.typesafe.ai discloses nothing new.
- **State sent:** the task prompt text only. No answers, no labels, no difficulty tags.
- **One request per task**, one Score (the decision) plus two Nouls (diagnostics only).
- **Decision rule, fixed now:** route HIGH iff the Score rounds to level 2 or 3 of 0–3.
  No threshold is fitted: 36 items with 6 positives cannot support one (the skill's floor
  is ~100 labels).
- **Outcomes are replayed, not re-run.** A LOW verdict takes Haiku's recorded correctness
  from the three existing runs; HIGH takes Opus's. Per-task Claude cost comes from the
  recorded fixtures. Jev is the only thing called.
- **Labels:** `out/routing_labels.json` — a task NEEDS HIGH when Haiku failed it in at
  least 2 of 3 runs. 6 of 36 (`h01 h05 h11 h12 m03 m09`); `m03` is failed by both models.

## Incumbents, on the same replay

| router | cost to decide | % sent high | pooled accuracy |
|---|---|---:|---:|
| heuristic (the one to beat: free) | 0 tokens | 58% | 96.3% |
| cascade | 1 Haiku verify / request | 14–17% | 94.4% |
| LLM classifier | 1 Haiku call / request | 61–67% | 92.6% |

## Pass bar

Jev **passes** only if all three hold on the replay:

1. **Quality:** accuracy ≥ heuristic − 2.8 pp (one task in 36).
2. **Selectivity:** sends ≤ 30% of tasks high — materially fewer than the heuristic's 58%,
   or it has bought nothing the free router did not already provide.
3. **Cost:** total cost (Claude + Jev) below the heuristic's.

Reported alongside, never as the verdict: AUC of the Score against NEEDS HIGH (≥ 0.80
would make it a useful *ranker*), recall on the 5 routable positives, the confidence split,
and cost/latency per decision against the LLM classifier.

## Known weaknesses of this test

- **6 positives.** An AUC or recall from them carries a wide interval; one task moves
  recall by ~17 points. A pass is a reason to test on real traffic, not a conclusion.
- **The labels are this workload's.** Short-answer dev questions, not agentic traffic.
- **Jev may not help where it is weakest.** Its documentation warns against arithmetic and
  counting; several NEEDS HIGH tasks are exactly that. It is asked to *recognise* that a
  task needs careful stepping, never to do the stepping — but that distinction is what
  this experiment tests.
