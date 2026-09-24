"""Does a Chinese prompt score like its English original? The pre-registered test.

    uv run python scripts/eval_heuristic_zh.py
    uv run python scripts/eval_heuristic_zh.py --real ~/prompts.json   # optional: your own prompts

Zero API calls. Compares the 36 workload prompts with two independent Chinese
translations (tasks/workload.zh-formal.jsonl, tasks/workload.zh-casual.jsonl),
for the router as it is now and for the English-only router it replaced. Also
re-checks the English gate, and reports -- never as part of the verdict -- two
more pair sets: the first adversarial review's 250 zh/en pairs
(tasks/zh-review-pairs.jsonl, in-sample for the post-hoc fixes), and 208 natural
pairs written by agents who never saw the router (tasks/zh-natural-pairs.jsonl).

Pass bar: docs/heuristic-zh-prereg.md (written before any translation existed).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dms.routers import heuristic
from dms.routers.heuristic_zh import HAN, HAN_CHARS_PER_WORD, has_han

CUT = 1.0                 # score >= CUT goes to the high model, as in pass-through mode
PARITY_MIN = 33           # of 36, per translation set
DRIFT_MAX = 3             # |high count - English high count|
SETS = ("formal", "casual")

Scored = dict[str, tuple[float, frozenset[str]]]


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "tasks" / name).read_text().splitlines()]


def _load(name: str) -> dict[str, str]:
    return {row["id"]: row["prompt"] for row in _rows(name)}


@contextmanager
def _english_only() -> Iterator[None]:
    """The router before Chinese support: whitespace words, English markers only."""
    saved = heuristic.word_count, heuristic.chinese_signals, heuristic.code_view
    heuristic.word_count = lambda text: len(text.split())
    heuristic.chinese_signals = lambda text: frozenset()
    heuristic.code_view = lambda prompt: prompt
    try:
        yield
    finally:
        heuristic.word_count, heuristic.chinese_signals, heuristic.code_view = saved


def _score_all(prompts: dict[str, str]) -> Scored:
    router = heuristic.HeuristicRouter()
    out: Scored = {}
    for key, prompt in prompts.items():
        score, signals = router.score(prompt)
        out[key] = (score, frozenset(s.name for s in signals if s.hit))
    return out


def _high(scored: Scored) -> int:
    return sum(score >= CUT for score, _ in scored.values())


def _agreement(en: Scored, zh: Scored) -> int:
    return sum((en[k][0] >= CUT) == (zh[k][0] >= CUT) for k in en)


def _signal_agreement(en: Scored, zh: Scored) -> dict[str, str]:
    names = sorted({n for _, fired in en.values() for n in fired} | {n for _, f in zh.values() for n in f})
    return {n: f"{sum((n in en[k][1]) == (n in zh[k][1]) for k in en)}/{len(en)}" for n in names}


def _ratio(en: dict[str, str], zh: dict[str, str]) -> float:
    """Han characters per English word over whole prompts. Payloads (logs, JSON,
    code) stay English in the translations, so this understates the ratio."""
    han = sum(len(HAN.findall(p)) for p in zh.values())
    words = sum(len(p.split()) for p in en.values())
    return han / words


def _needed(en: Scored, zh: Scored, baseline: Scored) -> str:
    """Where the English-only router already agreed, Chinese support had nothing
    to do; the informative count is on the items it had to fix."""
    side = lambda s, k: s[k][0] >= CUT
    needed = [k for k in en if side(en, k) != side(baseline, k)]
    fixed = sum(side(en, k) == side(zh, k) for k in needed)
    broke = sum(side(en, k) != side(zh, k) for k in en if k not in needed)
    return f"needed on {len(needed)}, fixed {fixed}; broke {broke} of the {36 - len(needed)} that agreed anyway"


def _report_set(name: str, en: Scored, zh: Scored, baseline: Scored) -> bool:
    agree, high = _agreement(en, zh), _high(zh)
    drift = high - _high(en)
    passed = agree >= PARITY_MIN and abs(drift) <= DRIFT_MAX
    print(f"\n== {name} ==")
    print(f"  parity   {agree}/36 on the same side of {CUT}   (bar >= {PARITY_MIN})"
          f"   English-only router: {_agreement(en, baseline)}/36")
    print(f"  high     {high}/36 vs English {_high(en)}/36, drift {drift:+d}   (bar <= ±{DRIFT_MAX})"
          f"   English-only router: {_high(baseline)}/36")
    print(f"  lexicon  {_needed(en, zh, baseline)}")
    print(f"  signals  {_signal_agreement(en, zh)}")
    for key in en:
        if (en[key][0] >= CUT) != (zh[key][0] >= CUT):
            print(f"  differs  {key}: en {en[key][0]:+.1f} {sorted(en[key][1])}"
                  f"  zh {zh[key][0]:+.1f} {sorted(zh[key][1])}")
    print(f"  verdict  {'PASS' if passed else 'FAIL'}")
    return passed


def _english_gate() -> int:
    """Mismatches against the 60 English scores frozen before Chinese support."""
    golden = json.loads((ROOT / "tests/fixtures/heuristic_en_golden.json").read_text())
    router, mismatches = heuristic.HeuristicRouter(), 0
    for name in ("workload.jsonl", "exemplars.jsonl"):
        for i, row in enumerate(_rows(name)):
            score, signals = router.score(row.get("prompt") or row["text"])
            now = [score, sorted(s.name for s in signals if s.hit)]
            mismatches += now != golden[row.get("id") or f"x{i:02d}"]
    return mismatches


def _side(scored: Scored, key: str) -> bool:
    return scored[key][0] >= CUT


def _pair_line(label: str, keys: list[str], en: Scored, zh: Scored, before: Scored) -> str:
    agree = sum(_side(en, k) == _side(zh, k) for k in keys)
    over = sum(_side(zh, k) and not _side(en, k) for k in keys)
    under = sum(_side(en, k) and not _side(zh, k) for k in keys)
    base = sum(_side(en, k) == _side(before, k) for k in keys)
    return (f"  {label:<10} parity {agree}/{len(keys)} ({agree / len(keys):.0%})"
            f"   zh-only high {over}, en-only high {under}   English-only router: {base}/{len(keys)}")


def _score_pairs(rows: list[dict]) -> tuple[Scored, Scored, Scored]:
    en = _score_all({r["id"]: r["en"] for r in rows})
    zh = _score_all({r["id"]: r["zh"] for r in rows})
    with _english_only():
        before = _score_all({r["id"]: r["zh"] for r in rows})
    return en, zh, before


def _report_pairs() -> None:
    rows = _rows("zh-review-pairs.jsonl")
    en, zh, before = _score_pairs(rows)
    print(f"\n== adversarial review pairs ({len(rows)}; in-sample for the post-hoc fixes) ==")
    for source in ("fp-hunter", "fn-hunter"):
        keys = [r["id"] for r in rows if r["id"].startswith(source)]
        print(_pair_line(source, keys, en, zh, before))


def _report_natural() -> None:
    rows = _rows("zh-natural-pairs.jsonl")
    en, zh, before = _score_pairs(rows)
    print(f"\n== natural pairs ({len(rows)}; written blind to the router) ==")
    print(_pair_line("all", [r["id"] for r in rows], en, zh, before))
    for level in ("hard", "routine"):
        keys = [r["id"] for r in rows if r["difficulty"] == level]
        print(f"  {level:<10} sent high: English {sum(_side(en, k) for k in keys)}/{len(keys)}"
              f"   Chinese {sum(_side(zh, k) for k in keys)}/{len(keys)}"
              f"   English-only router on the Chinese {sum(_side(before, k) for k in keys)}/{len(keys)}")


def _report_real(path: Path, score: Callable[[dict[str, str]], Scored]) -> None:
    texts = [p for p in json.loads(path.read_text()) if has_han(p)]
    prompts = {str(i): p for i, p in enumerate(texts)}
    now = score(prompts)
    with _english_only():
        before = score(prompts)
    n = len(prompts)
    print(f"\n== your prompts ({n}, containing Chinese; aggregate only) ==")
    print(f"  high     before {_high(before)} ({_high(before) / n:.0%})"
          f"   after {_high(now)} ({_high(now) / n:.0%})")
    moved_up = sum(before[k][0] < CUT <= now[k][0] for k in prompts)
    moved_down = sum(now[k][0] < CUT <= before[k][0] for k in prompts)
    print(f"  moved    low->high {moved_up}   high->low {moved_down}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", type=Path, help="JSON list of your own Chinese prompts")
    args = parser.parse_args(argv)

    en_text = _load("workload.jsonl")
    en = _score_all(en_text)
    verdicts = []
    for name in SETS:
        zh_text = _load(f"workload.zh-{name}.jsonl")
        zh = _score_all(zh_text)
        with _english_only():
            baseline = _score_all(zh_text)
        print(f"\n[{name}] Han characters per English word over whole prompts: "
              f"{_ratio(en_text, zh_text):.2f} (payloads stay English, so this understates;"
              f" the router uses {HAN_CHARS_PER_WORD})")
        verdicts.append(_report_set(name, en, zh, baseline))

    mismatches = _english_gate()
    print(f"\nEnglish gate: {mismatches} of 60 frozen English prompts changed (must be 0)")
    passed = all(verdicts) and mismatches == 0
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}  (both sets and the English gate; see the pre-registration)")
    _report_pairs()
    _report_natural()
    if args.real:
        _report_real(args.real, _score_all)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
