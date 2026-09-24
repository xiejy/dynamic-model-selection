"""Chinese support in the heuristic router -- and proof it left English alone.

Pre-registered in docs/heuristic-zh-prereg.md.
"""
import json
from pathlib import Path

import pytest

from dms.routers.heuristic import HAN_CHARS_PER_WORD, HeuristicRouter, word_count
from dms.routers.heuristic_zh import chinese_signals

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests/fixtures/heuristic_en_golden.json").read_text())


def _english_prompts() -> list[tuple[str, str]]:
    rows = []
    for name in ("workload.jsonl", "exemplars.jsonl"):
        for i, line in enumerate((ROOT / "tasks" / name).read_text().splitlines()):
            row = json.loads(line)
            rows.append((row.get("id") or f"x{i:02d}", row.get("prompt") or row["text"]))
    return rows


def _fired(prompt: str) -> set[str]:
    return {s.name for s in HeuristicRouter().score(prompt)[1] if s.hit}


# ------------------------------------------------------ English is untouched


@pytest.mark.parametrize("key,prompt", _english_prompts(), ids=lambda v: v if len(v) < 8 else "")
def test_every_english_prompt_scores_exactly_as_before(key, prompt) -> None:
    """Hard gate: every benchmark number in this repo was produced by these
    scores. Frozen from the English-only router before Chinese was added."""
    score, signals = HeuristicRouter().score(prompt)

    assert [score, sorted(s.name for s in signals if s.hit)] == GOLDEN[key]


def test_the_golden_file_covers_the_whole_workload() -> None:
    assert len(GOLDEN) == len(_english_prompts()) == 60


# ------------------------------------------------------------------- length


def test_english_length_is_still_whitespace_words() -> None:
    assert word_count("explain the root cause of this bug") == 7


def test_han_characters_count_as_fractions_of_a_word() -> None:
    assert word_count("解" * 35) == pytest.approx(35 / HAN_CHARS_PER_WORD)


def test_mixed_text_counts_both() -> None:
    """Chinese engineers keep English terms inline: 这个 regex 为啥不匹配."""
    assert word_count("这个 regex 为啥不匹配") == pytest.approx(1 + 7 / HAN_CHARS_PER_WORD)


def test_chinese_punctuation_is_not_a_word() -> None:
    assert word_count("好的，谢谢。") == pytest.approx(4 / HAN_CHARS_PER_WORD)


def test_a_long_chinese_prompt_is_long() -> None:
    """Before: a 170-character question was one 'word' and never long."""
    prompt = "请详细说明这个服务在高并发下的表现" * 10  # 170 characters

    assert {"long_prompt", "very_long_prompt"} <= _fired(prompt)


def test_a_short_chinese_prompt_is_not_long() -> None:
    assert "long_prompt" not in _fired("帮我看下这个日志")


# ------------------------------------------------------------ Chinese markers
# Each English marker's Chinese renderings fire the same signal; the traps
# both vocabulary builders flagged do not.

FIRES = [
    ("reasoning_markers", "为什么这个 goroutine 会卡住"),
    ("reasoning_markers", "这一行为啥报错"),
    ("reasoning_markers", "所以为什么不行"),
    ("reasoning_markers", "解释一下这段代码"),
    ("reasoning_markers", "请说明原因"),
    ("reasoning_markers", "证明这个算法是正确的"),
    ("reasoning_markers", "推导一下时间复杂度"),
    ("reasoning_markers", "这两种方案怎么权衡"),
    ("reasoning_markers", "根因是什么"),
    ("reasoning_markers", "会不会死锁"),
    ("reasoning_markers", "这段代码存在 SQL 注入漏洞吗"),
    ("reasoning_markers", "最坏情况下要比较几次"),
    ("reasoning_markers", "排序的下界是多少"),
    ("reasoning_markers", "违反了哪种一致性保证"),
    ("reasoning_markers", "指出这个函数的 bug"),
    ("reasoning_markers", "帮我优化这个查询"),
    ("reasoning_markers", "理论上最快是多少"),
    ("reasoning_markers", "怎么会死锁"),
    ("multi_step", "重复执行 4 次"),
    ("multi_step", "一步一步推导"),
    ("multi_step", "逐步分析这个过程"),
    ("multi_step", "访问序列为 A B C"),
    ("multi_step", "一共发生多少次缓存未命中"),
    ("multi_step", "每次迭代执行常数量的工作"),
    ("multi_step", "从 256 个元素开始"),
    ("multi_step", "这个字符出现了几次"),
    ("multi_step", "跟踪一下变量的值"),
    ("multi_step", "x 的最终值是多少"),
    ("simple_markers", "请从下面的 JSON 中提取 retries 的值"),
    ("simple_markers", "哪种编程语言使用 .rs 文件扩展名"),
    ("simple_markers", "请将这行日志分类为 ERROR 或 INFO"),
    ("simple_markers", "给出撤销提交的 git 命令"),
    ("simple_markers", "0xFF 的十进制值是多少"),
    ("simple_markers", "一个 int 占多少字节"),
    ("technical_terms", "加个互斥锁"),
    ("technical_terms", "这个正则表达式对吗"),
    ("technical_terms", "PUT 是幂等的吗"),
    ("technical_terms", "用闭包实现"),
    ("technical_terms", "给这一列加索引"),
    ("technical_terms", "递归函数"),
    ("technical_terms", "只读副本延迟了"),
    ("technical_terms", "写个定时任务"),
]

TRAPS = [
    ("reasoning_markers", "你认为什么方案好"),          # 认为 + 什么: "what do you think"
    ("reasoning_markers", "它被称为什么"),              # 称为 + 什么: "called what"
    ("reasoning_markers", "作为什么角色参与"),          # 作为 + 什么: "as what role"
    ("reasoning_markers", "Python 解释器的版本"),       # interpreter
    ("reasoning_markers", "开一份在职证明"),            # certificate of employment
    ("reasoning_markers", "列表推导式怎么写"),          # list comprehension
    ("reasoning_markers", "交了保证金"),                # deposit
    ("reasoning_markers", "阅读使用说明"),              # manual (noun)
    ("reasoning_markers", "法律漏洞"),                  # legal loophole, no "has / exists"
    ("reasoning_markers", "做好最坏的打算"),            # everyday, not worst-case
    ("multi_step", "去除重复元素"),                     # duplicate, not repeat
    ("multi_step", "反序列化这个 JSON"),                # deserialise
    ("multi_step", "经济逐步恢复"),                     # gradually
    ("multi_step", "他从来不写测试"),                   # 从来: never
    ("multi_step", "追踪快递"),                         # parcel tracking
    ("simple_markers", "特征提取器的结构"),             # a feature extractor, not a directive
    ("simple_markers", "为什么这个 git 命令会失败"),    # a hard debugging question
    ("simple_markers", "商品分类页面"),                 # category (noun)
    ("simple_markers", "字节跳动的面试题"),             # ByteDance
    ("technical_terms", "L2 正则化的作用"),             # regularisation
    ("technical_terms", "周末去刷副本"),                # game dungeon
    ("technical_terms", "下标签页"),                    # "the lower tab"
]


@pytest.mark.parametrize("signal,prompt", FIRES)
def test_the_chinese_rendering_fires_its_signal(signal, prompt) -> None:
    assert signal in chinese_signals(prompt.lower())


@pytest.mark.parametrize("signal,prompt", TRAPS)
def test_the_trap_does_not_fire(signal, prompt) -> None:
    assert signal not in chinese_signals(prompt.lower())


def test_chinese_vocabulary_is_never_consulted_for_english_text() -> None:
    """ieee\\s?754 would otherwise widen English matching ("IEEE 754")."""
    assert chinese_signals("is 0.1 + 0.2 exact in ieee 754 doubles?") == frozenset()


def test_accidental_english_substrings_are_not_translated() -> None:
    """English 'prove' fires on 'improve'; that is an English-side accident,
    not a concept, so 改进 does not fire it."""
    assert "reasoning_markers" not in chinese_signals("改进一下这段代码")


def test_the_patterns_cannot_backtrack_catastrophically() -> None:
    import time

    hostile = ("从" + "，" * 5 + "的" * 30 + "以") * 3000
    started = time.perf_counter()
    chinese_signals(hostile)

    assert time.perf_counter() - started < 0.5


def test_a_hard_chinese_question_now_goes_high() -> None:
    """The point of the change: before, this scored 0 and went to the low model."""
    score, _ = HeuristicRouter().score("为什么这两个线程会死锁？请一步一步分析")

    assert score >= 1.0


def test_an_easy_chinese_lookup_stays_low() -> None:
    score, _ = HeuristicRouter().score("请从这行日志中提取 HTTP 状态码，只回答数字")

    assert score < 1.0
