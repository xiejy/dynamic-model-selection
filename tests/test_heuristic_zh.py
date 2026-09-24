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
    ("multi_step", "每一次迭代执行常数量的工作"),     # post-hoc: bare 每次 is also "every time"
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
    ("multi_step", "他从来没开始写测试"),               # 从来: never -- reaches the 从 guard
    ("multi_step", "追踪快递"),                         # parcel tracking
    ("simple_markers", "特征提取器的结构"),             # a feature extractor, not a directive
    ("simple_markers", "为什么 git 命令会失败"),        # a hard debugging question
    ("simple_markers", "商品分类页面"),                 # category (noun)
    ("simple_markers", "问了几个字节跳动的同事"),       # ByteDance, after 几个
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


ANCHORS = ("从", "以", "有", "存在", "跟踪", "指出", "给出", "最终", "易被", "约束", "容易被", "破坏了")


@pytest.mark.parametrize("anchor", ANCHORS)
@pytest.mark.parametrize("filler", ["的" * 40, "abc_def.ghi " * 8, "x" * 300])
def test_no_gap_can_backtrack_catastrophically(anchor, filler) -> None:
    """Every bounded gap after an anchor, fed ~50k characters that never close
    it -- Han filler, dotted identifiers, one endless token."""
    import time

    hostile = (anchor + filler) * (50_000 // (len(anchor) + len(filler)))
    started = time.perf_counter()
    chinese_signals(hostile)

    assert time.perf_counter() - started < 1.0


def test_a_hard_chinese_question_now_goes_high() -> None:
    """The point of the change: before, this scored 0 and went to the low model."""
    score, _ = HeuristicRouter().score("为什么这两个线程会死锁？请一步一步分析")

    assert score >= 1.0


def test_an_easy_chinese_lookup_stays_low() -> None:
    score, _ = HeuristicRouter().score("请从这行日志中提取 HTTP 状态码，只回答数字")

    assert score < 1.0


# ------------------------------------------------ post-hoc fixes (review, 2026-09-24)
# Added after the pre-registered measurement; reported as post-hoc in
# docs/heuristic-zh-prereg.md. Each case is a finding an independent verifier
# reproduced.

POST_HOC_FIRES = [
    ("reasoning_markers", "这个接口怎么这么慢"),                 # colloquial why
    ("reasoning_markers", "线上服务怎么又 OOM 了"),
    ("reasoning_markers", "这俩协程咋就卡住不动了"),
    ("reasoning_markers", "机器上咋这么多 TIME_WAIT"),
    ("reasoning_markers", "凭啥这里要加锁"),
    ("reasoning_markers", "这个操作为什么会失败"),               # 操作 + 为什么, not 作为
    ("reasoning_markers", "配置为什么不生效"),
    ("reasoning_markers", "生成为什么这么慢"),
    ("reasoning_markers", "名称为什么显示乱码"),
    ("reasoning_markers", "讲下这段代码"),
    ("reasoning_markers", "讲一讲原理"),
    ("reasoning_markers", "科普一下 CAP"),
    ("reasoning_markers", "怎么证这个结论"),
    ("reasoning_markers", "证一下这个不等式"),
    ("reasoning_markers", "指出以下 Python 函数中的缺陷"),       # gap longer than 8 characters
    ("reasoning_markers", "指出 parseConfig 里的 bug"),          # an identifier inside the gap
    ("reasoning_markers", "说一下原因"),
    ("reasoning_markers", "这两者之间怎么折中"),
    ("reasoning_markers", "理论 QPS 能到多少"),
    ("reasoning_markers", "这次改动破坏了循环不变式"),
    ("multi_step", "从 index.ts 这个文件开始看"),               # '.' inside a file name
    ("multi_step", "从 main.py 开始看"),
    ("multi_step", "跟踪 processIncomingRequest 的执行"),
    ("multi_step", "跟踪一下这个请求"),
    ("multi_step", "逐步排查一下"),
    ("multi_step", "从不同的节点开始遍历"),                     # 从不同 is not 从不
    ("multi_step", "从第 5 行起"),
    ("multi_step", "画个登录的时序图"),
    ("technical_terms", "多副本之间怎么同步"),
    ("technical_terms", "数据复制延迟很高"),
    ("technical_terms", "从库查询很慢"),
    # second pass over what the review pairs still missed
    ("reasoning_markers", "这个原子操作为何在 ARM 上不生效"),     # 操作 + 为何, not 作为
    ("reasoning_markers", "干嘛这么设计"),
    ("reasoning_markers", "请指出这段 C 代码中可能导致内存泄漏的缺陷"),   # an 18-unit gap
    ("multi_step", "跟一下这个请求的调用链"),
    ("multi_step", "这是报错堆栈，帮我看看"),                   # a stack trace
    ("multi_step", "分步拆解一下状态转移"),
    ("multi_step", "分步算一下要多少台机器"),
    ("multi_step", "x 最后的值是多少"),
    ("multi_step", "重复到收敛为止"),
]

POST_HOC_TRAPS = [
    ("technical_terms", "帮我改一下标题，改成部署指南"),         # 一下 + 标题 straddles 下标
    ("technical_terms", "看一下标准库里有没有 json 解析"),
    ("technical_terms", "国内有什么好用的搜索引擎"),             # 搜索引擎 straddles 索引
    ("technical_terms", "把 users 表的数据复制到 users_bak 表"),  # copy the data
    ("technical_terms", "下单后从库存里扣掉数量"),               # 从 + 库存 straddles 从库
    ("technical_terms", "关闭包括日志在内的输出"),               # 关闭 + 包括 straddles 闭包
    ("technical_terms", "这会改变基本的行为"),                   # 改变 + 基本 straddles 变基
    ("technical_terms", "docker 文件夹放哪"),
    ("technical_terms", "如果值为正则返回 1"),                   # 为正，则返回
    ("technical_terms", "把参数传递归一化之后的值"),             # 传递 + 归一化 straddles 递归
    ("technical_terms", "IEEE 754 双精度里 0.1 + 0.2 等于 0.3 吗"),  # English keys only "ieee-754"
    ("reasoning_markers", "调整一下界面的配色"),                 # 一下 + 界面 straddles 下界
    ("reasoning_markers", "这个默认行为什么时候改的"),           # 行为 + 什么时候: when, not why
    ("reasoning_markers", "timeout 设置为什么值比较合适"),       # 设置为 + 什么值
    ("reasoning_markers", "日志分为什么级别"),
    ("reasoning_markers", "因为什么都没做"),
    ("reasoning_markers", "因为毛利率下降，帮我翻译成英文"),     # 为毛 slang vs 毛利率
    ("reasoning_markers", "表单校验失败时给出错误提示"),         # emit an error, not name the bug
    ("reasoning_markers", "这种写法很容易写出 bug"),
    ("reasoning_markers", "这个包的使用说明怎么看"),             # 说明 the noun
    ("reasoning_markers", "我想了解读写分离怎么配"),             # 了解 + 读写 straddles 解读
    ("reasoning_markers", "我想了解释放锁的时机"),               # 了解 + 释放 straddles 解释
    ("reasoning_markers", "明天要给客户讲解决方案"),             # 讲 + 解决 straddles 讲解
    ("reasoning_markers", "帮我开一份实习证明"),
    ("reasoning_markers", "社保证明怎么开"),
    ("reasoning_markers", "昨天强推导致代码丢了"),               # 强推 + 导致 straddles 推导
    ("reasoning_markers", "进程被杀死锁住的文件没释放"),         # 杀死 + 锁住 straddles 死锁
    ("reasoning_markers", "把这个任务调优先级"),                 # 调 + 优先级 straddles 调优
    ("reasoning_markers", "这种病根据指南怎么治"),               # 病 + 根据 straddles 病根
    ("reasoning_markers", "说下进度"),                          # say, not explain
    ("reasoning_markers", "去办证一下"),
    ("reasoning_markers", "这个参数是干嘛用的"),                 # what for, not why
    ("multi_step", "把开发生产两套环境的配置分开"),             # 开发 + 生产 straddles 发生
    ("multi_step", "push 之后自动触发生成文档"),
    ("multi_step", "点击按钮后会出现一个弹窗"),                 # appear, not occur
    ("multi_step", "pandas 里怎么把浮点数列保留两位小数"),       # 浮点数 + 列 straddles 数列
    ("multi_step", "Excel 怎么隐藏所有奇数列"),
    ("multi_step", "手动执行一下这个脚本"),                     # run by hand, not trace
    ("multi_step", "以后开始用 pnpm"),                          # 以后 is not 以
    ("multi_step", "把汇总共享到群里"),                         # 汇总 + 共享 straddles 总共
    ("multi_step", "整合计费模块"),                             # 整合 + 计费 straddles 合计
    ("multi_step", "每次运行都报错"),                           # every time; English keys "each time"
    ("multi_step", "灰度逐步来吧"),                             # gradually
    ("simple_markers", "多了几个字节导致校验失败"),             # a few bytes
    ("simple_markers", "用哪个 git 命令可以撤销提交"),           # English keys only "give the git command"
    ("simple_markers", "把时间戳转成十进制再比较"),             # English keys only "decimal value of"
    # second pass
    ("reasoning_markers", "这个变量命名为什么比较好"),           # 命名为 + 什么
    ("reasoning_markers", "服务停了以后状态会变为什么"),         # 变为 + 什么
    ("reasoning_markers", "还有这个漏洞的 CVE 编号是多少"),      # 还有: "also"
    ("reasoning_markers", "没什么原因，就是想改"),               # "no particular reason"
    ("multi_step", "Excel 最后加一行合计"),                     # a totals row
    ("multi_step", "打完折最终价格是多少"),                     # a final price, not a final value
    ("multi_step", "我跟一下产品的值班"),                       # following up, not tracing
    ("multi_step", "最后的值班表发我一下"),
    ("multi_step", "我们分步走，先上 A 再上 B"),                 # phased rollout
    ("simple_markers", "为什么内核把这块内存归类为 cache"),     # a description, not a directive
]


@pytest.mark.parametrize("signal,prompt", POST_HOC_FIRES)
def test_post_hoc_rendering_fires(signal, prompt) -> None:
    assert signal in chinese_signals(prompt.lower())


@pytest.mark.parametrize("signal,prompt", POST_HOC_TRAPS)
def test_post_hoc_trap_does_not_fire(signal, prompt) -> None:
    assert signal not in chinese_signals(prompt.lower())


@pytest.mark.parametrize("text,words", [
    ("好的,谢谢.", 4 / HAN_CHARS_PER_WORD),                    # half-width punctuation by Han
    ("“登陆”改成“登录”", 6 / HAN_CHARS_PER_WORD),              # IME quotation marks
    ("改成 x = 3x + 1 吧", 5 + 3 / HAN_CHARS_PER_WORD),         # code operators still count
    ("用ｅｘｐｌａｉｎ看看", 1 + 3 / HAN_CHARS_PER_WORD),       # full-width letters are a word
])
def test_punctuation_next_to_chinese_is_not_a_word(text, words) -> None:
    assert word_count(text.lower()) == pytest.approx(words)


def test_full_width_brackets_typed_by_a_chinese_ime_are_code() -> None:
    """A Chinese IME types （） by default; English foo() fires contains_code."""
    assert "contains_code" in _fired("看下 foo（）的返回值")


def test_rare_han_extensions_are_han() -> None:
    from dms.routers.heuristic_zh import has_han

    assert has_han("\U00030000") and has_han("\U0002f800") and has_han("〇")


# ------------------------------------------ post-hoc, round 2 (fresh blind review)

ROUND2_FIRES = [
    ("reasoning_markers", "点了按钮之后为什么状态没有更新"),     # a noun right after 为什么 is its subject
    ("reasoning_markers", "为什么类型检查过不了"),
    ("reasoning_markers", "接口返回的为什么中文是乱码"),
    ("reasoning_markers", "为什么时间戳差了8小时"),
    ("reasoning_markers", "设置为什么不生效"),                   # 设置 + 为什么: still why
    ("reasoning_markers", "我不确定为什么会这样"),
    ("reasoning_markers", "改了配置怎么还是不生效"),             # colloquial why
    ("reasoning_markers", "这个服务怎么一直重启"),
    ("reasoning_markers", "日志怎么没有输出"),
    ("reasoning_markers", "除了解释原因之外，还要给出修复方案"),
    ("reasoning_markers", "请从原理解释一下 HTTPS 握手的过程"),
    ("reasoning_markers", "结合这张架构图解释一下数据流向"),
    ("reasoning_markers", "已知 a>0，b>0，求证 a+b>=2√(ab)"),
    ("reasoning_markers", "试证：任意连通无向图都存在生成树"),
    ("reasoning_markers", "请证明确实不存在更快的算法"),
    ("reasoning_markers", "这段代码会不会被 SQL 注入"),
    ("reasoning_markers", "按阿姆达尔定律，理论加速比是多少"),
    ("reasoning_markers", "volatile变量保证原子性吗"),
    ("reasoning_markers", "如何保证消息不丢失"),
    ("reasoning_markers", "说一下这个报错的原因\n日志在下面"),
    ("reasoning_markers", "说一下原因吧"),
    ("reasoning_markers", "线程 t1 和 t2 锁死了"),               # pangu spacing
    ("multi_step", "然后重复第 2 步，直到误差小于 0.01"),
    ("multi_step", "失败的话重复上一步"),
    ("multi_step", "重复直至收敛"),
    ("multi_step", "请分步解答这道动态规划题"),
    ("multi_step", "请逐步解答"),
    ("multi_step", "这是报错的堆栈，帮我看看"),
    ("multi_step", "堆栈信息如下"),
    ("multi_step", "跟踪 ctx 和 req 两个变量"),                 # pangu spacing
    ("multi_step", "从 0 开始计数"),
    ("multi_step", "从头开始"),
    ("multi_step", "请⼀步⼀步算这道题"),                       # Kangxi radical U+2F00 from a PDF
    ("technical_terms", "MySQL 主从延迟很高怎么排查"),
    ("technical_terms", "主从同步断了怎么恢复"),
    ("technical_terms", "读副本会读到旧数据吗"),
]

ROUND2_TRAPS = [
    ("reasoning_markers", "字符集应该设置为什么？"),             # set to what
    ("reasoning_markers", "初始化为什么？"),
    ("reasoning_markers", "翻译为什么比较好"),
    ("reasoning_markers", "把这个回调改为什么都不做"),
    ("reasoning_markers", "不管什么原因失败，都返回 500"),       # for whatever reason
    ("reasoning_markers", "java.lang.OutOfMemoryError: Metaspace 一般是啥原因"),   # "what causes"
    ("reasoning_markers", "跟产品讲下周的排期"),                 # 讲 + 下周
    ("reasoning_markers", "先说明一下，这个项目用的是 python 3.8"),   # a preamble
    ("reasoning_markers", "帮我写一份工作证明"),
    ("reasoning_markers", "TS 的类型推导不出来"),                # type inference
    ("reasoning_markers", "改完顺手推到 main 分支"),             # 顺手 + 推 (git push)
    ("reasoning_markers", "有没有好用的漏洞扫描工具推荐"),
    ("reasoning_markers", "改完之后保证测试能过"),               # make sure
    ("reasoning_markers", "尽量保证代码整洁"),
    ("reasoning_markers", "求这个函数的最优解"),                 # optimal is not optimi-
    ("reasoning_markers", "我去跟产品求证一下需求"),             # verify with someone
    ("reasoning_markers", "这个 bean 会被注入到哪个类"),         # dependency injection
    ("multi_step", "从上周开始就一直报 502"),                   # since
    ("multi_step", "从 v2.3 开始就不兼容了"),
    ("multi_step", "从这个版本起"),
    ("multi_step", "从日志看是 NPE 引起的"),
    ("multi_step", "Rust 的开发生态怎么样"),                    # 开发 + 生态
    ("multi_step", "跟下面的代码比较一下有什么区别"),           # 跟 = with
    ("multi_step", "df 里的分数列有空值"),                      # 分数 + 列
    ("multi_step", "按时间排序列出来"),                         # 排序 + 列出
    ("multi_step", "grpc 偶发 DEADLINE_EXCEEDED"),              # intermittent
    ("multi_step", "去除重复之前先排序"),                       # 重复 = duplicate
    ("technical_terms", "如果数字为正则输出 yes"),               # 为正，则输出
    ("technical_terms", "L2 正则系数怎么调"),                   # regularisation
    ("technical_terms", "2 的 10 次幂等于多少"),                 # 幂 + 等于
    ("technical_terms", "Excel 数据复制粘贴后格式乱了"),
    ("technical_terms", "从库里导入数据"),                      # from the library
    ("technical_terms", "探索引入新框架"),                      # 探索 + 引入
    ("technical_terms", "看下标红的地方"),                      # 看下 + 标红
    ("technical_terms", "上下标怎么打"),                        # sub/superscript
    ("technical_terms", "从服务器下载文件"),
    ("simple_markers", "为什么我的 VS Code 扩展名称在市场里显示不对"),   # an extension's name
]


@pytest.mark.parametrize("signal,prompt", ROUND2_FIRES)
def test_round2_rendering_fires(signal, prompt) -> None:
    assert signal in chinese_signals(prompt.lower())


@pytest.mark.parametrize("signal,prompt", ROUND2_TRAPS)
def test_round2_trap_does_not_fire(signal, prompt) -> None:
    assert signal not in chinese_signals(prompt.lower())


@pytest.mark.parametrize("prompt", ["这条SELECT语句为什么这么慢", "用def定义的函数", "grid【i】【j】 越界了"])
def test_code_glued_to_chinese_is_code(prompt) -> None:
    assert "contains_code" in _fired(prompt)


@pytest.mark.parametrize("prompt", ["我国的首都是（）", "超时时间（可选）（默认 30 秒）", "分⑴⑵两种情况"])
def test_chinese_typography_is_not_code(prompt) -> None:
    """（） is an exam blank and ）（ back-to-back asides; ⑴ is a numbered list."""
    assert "contains_code" not in _fired(prompt)


def test_punctuation_after_chinese_punctuation_is_not_a_word() -> None:
    assert word_count("（可选）, 默认开启") == pytest.approx(word_count("（可选），默认开启"))


def test_smart_apostrophes_inside_english_words_are_kept() -> None:
    """don’t is one word whether or not the message also contains Chinese."""
    assert word_count("don’t 中文") == pytest.approx(1 + 2 / HAN_CHARS_PER_WORD)


def test_kangxi_radicals_count_as_han() -> None:
    assert word_count("⼀步") == pytest.approx(2 / HAN_CHARS_PER_WORD)


def test_long_punctuation_runs_are_linear() -> None:
    import time

    started = time.perf_counter()
    word_count("中" + "!" * 40_000)

    assert time.perf_counter() - started < 0.5
