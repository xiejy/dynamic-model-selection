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

---

# Results — added 2026-09-23, after the run. Everything above is unchanged.

**Verdict: Jev does not pass.** 36 live requests to `jev-1.13.0`, 19,643 input tokens,
**$0.000825** total, latency median 525 ms (max 688 ms; the vendor states 70–500 ms).

| router | accuracy | % sent high | cost | |
|---|---:|---:|---:|---|
| heuristic | 95.4% | 58% | $0.0217 | the bar |
| **Jev** | **90.7%** | **17%** | **$0.0122** | FAIL quality · PASS selectivity · PASS cost |
| label-perfect router | 94.4% | 17% | $0.0133 | the ceiling |

Diagnostics: AUC **0.703** (below the 0.80 "useful ranker" line); recall **3/5** on the
routable tasks; mean confidence 0.70 where right, 0.51 where wrong.

## What it gets right and wrong

- **Multi-step computation: perfect.** `h01` (trace a loop), `h12` (count LRU misses) and
  `m09` (evaluate a comprehension) all scored ~2.0 at confidence 0.96–1.00, with the
  `multi_step` Noul at 0.92–0.98.
- **Subtle-terminology traps: blind.** Jev **never used level 3** — the highest score
  across all 36 tasks was exactly 2.00. The `trap` Noul rated both misses (0.34, 0.51)
  *below* two false alarms (0.64, 0.67).

## The misses sit on this benchmark's weakest labels

Both routable tasks Jev missed are cases where **Haiku's answer was defensible and the
grader rejected it**:

- `h05` — Haiku answered *"Read-after-write consistency"*, a standard synonym for
  read-your-writes. The grader's accepted list omits it. This is a **grader bug**.
- `h11` — Haiku answered **Ω(n log n)**, the correct notation for a lower bound; the
  prompt asked for "big-O notation", which is itself imprecise here. A **task-design bug**.

These defects inflate Haiku's failure rate in **every** result in this repo, not only
Jev's. They are recorded here and **not corrected in this evaluation**: changing graders
after seeing which router they penalise is what pre-registration exists to prevent. Fix
them, then pre-register a fresh run.

## Exploratory, not a verdict

The one follow-up the Jev documentation itself suggests — route high when confidence is
low — was tried post-hoc:

| rule | accuracy | % high | cost |
|---|---:|---:|---:|
| pre-registered: level ≥ 2 | 90.7% | 17% | $0.0122 |
| level ≥ 2 **or** confidence < 0.60 | 93.5% | 50% | $0.0196 |

It recovers accuracy only by spending the selectivity, converging on the free heuristic
while still trailing it. Here the confidence lever turns Jev into a slightly worse,
slightly cheaper heuristic.

## Conclusion

Not as the sole model selector, on this workload. Jev's distinctive signal is real —
it recognises *"this needs careful stepping"* almost perfectly — but a router also needs
*"a plausible answer here is often wrong"*, and Jev does not see that. Worth revisiting
only after the graders are fixed, on traffic where multi-step work dominates, and with a
fresh pre-registration.
