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

# Results — to be added after the run. Everything above stays unchanged.
