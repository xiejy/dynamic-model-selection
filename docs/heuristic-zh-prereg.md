# Chinese support for the heuristic router — pre-registration

Written **2026-09-24, before any Chinese translation or lexicon has been seen.** Nothing
below may be changed after the first parity number is computed; a changed bar is a
different experiment.

## Question

The heuristic router (`src/dms/routers/heuristic.py`) decides low vs high for every typed
message in pass-through mode. Its vocabulary is English and it measures length by splitting
on spaces, so a Chinese prompt fires almost nothing and has almost no "words": it goes low
however hard it is. **After adding Chinese vocabulary and a character-based length measure,
does a Chinese prompt score like its English original?**

The target is *parity with the English router*, not correctness. Where the English router
misjudges a task, the Chinese one should misjudge it the same way; improving the router is a
separate question.

## Method

- **Items:** the 36 prompts in `tasks/workload.jsonl`, in **two independent translations**:
  formal written Chinese, and casual chat as a Chinese engineer types to Codex. Each was
  made by an agent that was forbidden to read the router or its tests.
- **Vocabulary:** built by **two agents** (precision-first and recall-first) from the
  English marker lists alone, forbidden to read the workload or any translation. Target:
  the same concepts as the English lists, no additions.
- **Merge:** the union of both lexicons, minus patterns removed for a stated
  false-positive reason. The merge is made on the lexicons and their stated risks only; it
  is frozen (committed) before the parity script runs, and any pattern changed after the
  first parity number is reported as a post-hoc change, with its effect.
- **Length:** a Chinese character counts as `1 / R` of a word, where `R` (Han characters
  per English word) comes from published sources gathered before measurement. Text with no
  Han characters is counted exactly as before.
- **Decision cut:** score ≥ 1.0 → high, the cut pass-through mode uses.

## Pass bar

The change **passes** only if all three hold:

1. **English unchanged (hard gate):** all 60 English prompts (`tasks/workload.jsonl` and
   `tasks/exemplars.jsonl`) produce exactly the same score and the same fired signals as
   before the change. Every benchmark number in this repo depends on them.
2. **Parity:** in *each* translation set, at least **33 of 36** Chinese prompts land on the
   same side of the 1.0 cut as their English original.
3. **No drift:** in each set, the number of Chinese prompts sent high is within **±3** of the
   English count (21 of 36).

Reported alongside, never as the verdict: the same parity for the current English-only
router (the baseline this fixes), per-signal agreement, the measured characters-per-word
ratio of the translations against `R`, and the score distribution on the owner's own
Chinese Codex prompts before and after (aggregate counts only — the prompts stay local).

## Known weaknesses of this test

- **Translations are LLM-made.** Real users' phrasing is wider; the casual set is the
  closer proxy, and the owner's own prompts are the reality check.
- **36 items.** One task is 2.8 points of parity.
- **Parity inherits the English router's blind spots.** It is keyword matching; a hard
  question phrased without any marker goes low in either language.
- **The merge is mine.** I will have seen the translations when the workflow returns; the
  freeze-before-measure rule and the post-hoc reporting rule are the guard.

---

# Results — added 2026-09-24, after the run. Everything above is unchanged.

Procedure as run: pre-registration committed `7e20745` (10:27); vocabulary and length rule
frozen in `a26576c` (10:59), with all 60 English prompts pinned by a test; then
`scripts/eval_heuristic_zh.py` run once. The translations were visible to me when the
inputs workflow returned, before the merge — the case the last weakness above anticipated.
The merge rules I applied were the ones stated in `heuristic_zh.py`'s docstring, decided
from the two lexicons and their stated risks.

## Verdict: PASS — at the margin

| | formal | casual | bar |
|---|---:|---:|---|
| parity (same side of 1.0 as English) | **33/36** | **34/36** | ≥ 33 |
| sent high (English: 21) | 24 (**+3**) | 23 (+2) | within ±3 |
| English-only router, parity | 25/36 | 28/36 | — |
| English-only router, sent high | 10 | 15 | — |
| English gate (60 frozen prompts) | unchanged | | hard gate |

The formal set passes with no slack on either criterion. Every disagreement errs toward
the expensive model; none sends a prompt low that English sends high:

- **h07** (both sets) — 最坏情况 fires *worst-case*; the English prompt writes "worst case"
  without the hyphen, which the English list does not match. h07 is a *hard* task, so the
  Chinese routing is the better one; the gap is the English list's narrowness.
- **e10** (both sets) — formal: 语义化版本 is the ordinary Chinese for "semantic version"
  and is rendered as *semver*; the English prompt spells out "semantic version", which does
  not fire. *Corrected after review:* the casual e10 is not the vocabulary at all — that
  translator wrote the English word "semver" inline, which the English list itself matches
  (the English-only router fires on it too). A translation added a router keyword.
- **m05** (formal only) — the formal translation is wordier than the English and crosses
  the 35-word length line.

Per-signal agreement: 31–36 of 36 per signal in both sets (`long_prompt` weakest in the
formal set, 31/36).

## Reported alongside

- **Characters per word.** Over whole prompts the translations run 1.30 (formal) and 0.97
  (casual) Han characters per English word — understated, because log lines, JSON and code
  stay in English, and the casual set keeps English terms inline. The router's 1.5 comes
  from the sources, not from these.
- **The owner's own Chinese Codex prompts** (507 distinct typed messages with Chinese, from
  local rollouts; content never leaves the machine): sent high **26% → 39%**; 66 moved
  low → high, none high → low. Length alone moved 15; the other 51 gained a vocabulary
  signal. A private read of the 16 short ones that moved found mostly genuine *why /
  explain / derive* questions, and at least one straddle bug: 搜索引擎 ("search engine")
  contains 索引 ("index").

---

# Post-hoc — everything below came after the pre-registered measurement

The verdict above stands as measured. What follows changed the router afterwards; each
change is reported with its effect, and none is part of the verdict.

## Two disclosures

- **About a third of the original test sentences were phrased after the translations.**
  14 of the 42 original "fires" test strings share a run of 6+ characters with a translation
  (e.g. 从 256 个元素开始, 哪种编程语言使用 .rs 文件扩展名). The *patterns* came from the two
  blind vocabulary builders, and the merged lexicon passed those tests on its first run
  without edits — but the test list is not independent of the measurement items.
- **The merge inputs are now in the repo**: `docs/heuristic-zh/lexicon-precision.json`,
  `lexicon-recall.json`, and the length sources in `chars-per-word-sources.json`.

## Why the 36-item test could not see the problems

Of 36 items, the English-only router already agreed on 25 (formal) and 28 (casual):
shared English payloads carry them. The vocabulary only mattered on 11 and 8 items — it
fixed 11/11 and 7/8, and broke 3 and 1. Nothing in the set is routine casual chat, and
nothing probes word boundaries.

## Round 1 — adversarial review (8 agents: 4 finders, 4 verifiers)

84 findings, 75 verified as real (some overlap: two reviewers could find the same
straddle; an earlier version of this note said "54, about 45", which was a miscount).
The largest class: **word-boundary straddles**.
Chinese is written without spaces, so a two-character marker matches across two words —
一下+标题 contains 下标 ("index"), 开发+生产 contains 发生 ("occur"), 一下+界面 contains 下界
("lower bound"). Also missing colloquial forms (怎么这么慢 as "why"), gaps that could not
cross `v1.2` or `main.py`, half-width punctuation counted as words, a negative signal
broader than its English marker, and traps that could never reach the guard they claimed
to test. All fixed test-first (commit `eff62e9`).

Effect, **in-sample** (these are the pairs the fixes were made against), on the review's
250 zh/en pairs: routine prompts 52% → 95% parity, hard prompts 9% → 71%.

## Round 2 — a fresh round, blind to round 1 (8 agents)

- **Natural sample.** Two agents who never saw the router wrote 208 zh/en pairs of
  everyday traffic (`tasks/zh-natural-pairs.jsonl`). This is the out-of-sample check.
- **Fresh hunters and a code re-review**, forbidden to read round 1's findings: 87
  findings, 82 verified as real.

The code re-review found a **high-severity regression introduced in round 1**: a "what +
noun" guard meant for 设置为什么值 also killed genuine whys whose subject follows 为什么
(为什么类型检查不通过, 为什么中文会乱码). Also: pangu spacing (a space around every Latin
token) used up the gap budget; the half-width punctuation rule was quadratic on long
runs; the NFKC fold turned the exam blank （） into code. The hunters found more straddles
and missing forms — 从上周开始 is "since", not "starting from"; an imperative 保证 is "make
sure", not "guarantee". All verified findings fixed test-first, with two **reverts informed
by the natural sample**: 偶发/偶现 and 啥原因/什么原因, added in round 1, are how Chinese says
"intermittent" and "what causes" — which the English lists do not key.

## The numbers that matter: the natural sample

| router | parity | hard sent high | routine sent high |
|---|---:|---:|---:|
| English, on the English side | — | 60 / 81 | 29 / 127 |
| English-only, on the Chinese side | 70.7% | 13 / 81 | 15 / 127 |
| frozen (`a26576c`) | 88.9% | 59 / 81 | 27 / 127 |
| round 1 (`eff62e9`) | 88.9% | 59 / 81 | 29 / 127 |
| round 2 (final) | 90.9% | 57 / 81 | 27 / 127 |

On the owner's 507 Chinese Codex messages the final router sends 38% high (26% before;
64 moved up, none down) — the frozen router's 39% is the figure under *Reported alongside*.

**Chinese support is the whole win** on real traffic: without it, 68 of 81 hard Chinese
prompts went to the cheap model; with it, the split matches English almost exactly. **The
post-hoc rounds did not move natural-traffic parity**: round 1 is flat, and round 2's +2
points are the two reverts the sample itself informed. The review defects were real but
rare in natural phrasing.

Of the 19 remaining natural disagreements: 9 are length (the two languages' messages differ
in verbosity, 6 one way and 3 the other); 5 are English-list accidents on the English side
("improve" contains "prove" twice, "optimistic" contains "optimi", `$2.80` looks like code,
"start with fix:" is a string prefix); 5 are translation choices (简单讲讲 written as
"short version", 为啥 dropped from the English, `->` for 调到).

## Limits

This is keyword matching over unsegmented text. Two rounds of adversarial review kept
finding straddles; each fix is a guard that could, in turn, block a legitimate phrasing
(round 1's own regression is the example). Parity on natural traffic has plateaued around
89–91%, bounded by translation choice and the English lists' own accidents rather than by
missing vocabulary. Going further would mean segmenting Chinese into words (e.g. jieba)
before matching — a dependency this demo does not take.

Reproduce everything: `uv run python scripts/eval_heuristic_zh.py`.
