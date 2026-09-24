"""Chinese for the heuristic router: the same signals, and a length that counts.

The English router measures length in whitespace words and matches English
markers, so a Chinese prompt -- no spaces, no English -- scored 0 however hard
it was and always went to the low model.

Two changes, both inert on text without Han characters, so every English score
is exactly what it was (tests/test_heuristic_zh.py holds all 60 frozen):

* **Length.** A Han character counts as 1/1.5 of a word. Parallel corpora put
  English -> Chinese at 1.47-1.62 Han characters per English word (FLORES,
  WMT21-23, TICO-19, the Python docs translation) and natively written Chinese
  nearer 1.35; 1.5 is the central estimate. Punctuation is not counted.
* **Vocabulary.** Each English marker's Chinese renderings, merged from two
  vocabulary builders (precision-first and recall-first) who worked from the
  English lists alone and never saw the benchmark prompts. Parity is the
  target: the same concepts, no additions. Dropped from the merge:
  - renderings of *accidental* English hits ("improve" contains "prove",
    "consequence" contains "sequence") -- English-side bugs, not concepts;
  - the recall builder's looser forms of `simple_markers`, the one negative
    signal, where a false hit pushes a hard prompt down.
  Every pattern's guard is against a named trap (解释器 is an interpreter,
  保证金 a deposit, 序列化 serialisation, 刷副本 a game dungeon...).

Pre-registered test: docs/heuristic-zh-prereg.md.
"""
from __future__ import annotations

import re

HAN_CHARS_PER_WORD = 1.5

HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ebef]")
# Han characters and CJK / full-width punctuation: neither is a whitespace word.
_NOT_A_WORD = re.compile(
    r"[\u3000-\u303f\uff00-\uffef\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ebef]"
)

_P = r"[^，。,.；;？?！!\n]"  # one character inside the current clause: gaps never cross punctuation

_VOCABULARY: dict[str, tuple[str, ...]] = {
    "reasoning_markers": (
        # why. 认为/称为/视为/设为/作为/以为/成为 + 什么 are "think/called/as what", not "why".
        r"(?<![认認称稱视視设設作以成])[为為爲](?:什[么麼]|甚[么麼]|啥|嘛|毛)(?![样樣])",
        r"所以[为為](?:什[么麼]|啥|何)|因[为為]什[么麼](?![都也])",
        r"(?<![认認称稱视視设設作])[为為爲]何(?![种種物方])",
        r"怎[么麼][会會](?![用写寫])|[干幹][嘛吗麼](?:要|非|不|[还還]|老是|[总總]是)",
        # explain. Not 解释器 / 解释型语言 (interpreter), 解释权 (fine print), 讲解员.
        r"解[释釋](?!器|型|性?[语語]言|[执執]行|[权權])",
        r"[讲講](?:解(?![员員])|[讲講]|一下|清楚|明白|透)|[阐闡](?:述|明|[释釋])|解[读讀]",
        (r"(?:[请請]|并|並|[详詳][细細]|具[体體]|分[别別]|逐一|[帮幫]我)[说說]明"
         r"|[说說]明(?:一下|原因|理由|[为為]什|[为為]何|如何|怎[么麼]|是否)"),
        # prove. Not 在职证明 / 证明材料 (paperwork), nor 验证明文 across a word boundary.
        (r"(?<![开開具职職入历歷份住产產验驗认認论論凭憑签簽])[证證]明(?![书書材信函人天年日白确確细細显顯文])"
         r"|[论論][证證]"),
        r"推[导導](?!式)",                                   # not 列表推导式 (list comprehension)
        r"[权權]衡|取[舍捨]",                                # trade-off
        r"根本原因|根因|根源|病根",                          # root cause; bare 原因 is just "reason"
        r"[复複][杂雜][度性]",                               # complexity; bare 复杂 is "complicated"
        r"死[锁鎖]|(?:互相|相互|[线線]程|[进進]程).{0,4}[锁鎖]死",
        # vulnerable ("is vulnerable" = has a hole / is exposed to attack; bare 漏洞 is "vulnerability")
        (rf"易[受被]{_P}{{0,10}}攻[击擊]|容易(?:受到|遭受|被){_P}{{0,8}}(?:攻[击擊]|注入|利用)"
         rf"|(?:(?<!所)有|存在){_P}{{0,10}}漏洞"),
        r"最[坏壞差糟]糕?的?情[况況形]",                     # worst-case; not 做好最坏的打算
        r"下[确確]?界",                                      # lower bound; not 下限 (lower limit)
        # violated: past or passive, as English keys "violated" only
        (rf"(?:被|遭)[违違][反背]|[违違][反背]了"
         rf"|(?:不[变變](?:量|式|性)|[约約]束|契[约約]|前置条件|后置条件){_P}{{0,8}}破[坏壞]"),
        r"(?<![我量])保[证證](?![金书書人])",                # not 保证金 / 我保证 / 质量保证
        # name the cause / the bug: a request, not a narrated or negated event
        (rf"(?<![没沒未不])(?:指出|[说說]出|[点點]出|[给給]出|[写寫]出)(?!了){_P}{{0,8}}"
         r"(?:原因|起因|成因|bug|缺陷|[错錯][误誤]|毛病|[问問][题題]所在)"),
        r"[优優]化|最[优優](?![秀惠先美良雅质質])|最佳化|[调調][优優]",   # optimi-
        r"理[论論](?:上|值|性|最|极限|極限|峰值|上限|分析|[计計]算|[复複][杂雜]度|性能|吞吐|速度)",
    ),
    "multi_step": (
        # repeat. Bare 重复 is mostly "duplicate" (去除重复元素).
        (r"重[复複](?:地|[执執]行|[运運]行|[调調]用|操作|[进進]行|做|以上|上述|上面|前面|此|[这這]|[该該]"
         r"|步[骤驟]|[过過]程|直到|\s*[0-9a-z一二两兩三四五六七八九十百千几幾]+\s*[次遍轮輪])"
         r"|不[断斷]重[复複]|反[复覆](?!无常|無常|横跳|橫跳)|一[遍次]又一[遍次]"),
        # step by step. 逐步 alone is "gradually" (经济逐步恢复).
        r"一步一?步|分步[骤驟]|按步[骤驟]",
        (r"逐步地?(?:推[导導]|推理|分析|[计計]算|解[释釋]|[说說]明|演示|展示|思考|求解|[执執]行|模[拟擬]"
         r"|[讲講]解|[给給]出|列出|[跟追][踪蹤])"),
        # sequence. Not 序列化 (serialise), 序列号 (serial number), 时间序列 (time series).
        r"(?<!时间)(?<!時間)序列(?![化号號表出])|(?<![参參函变變整行])[数數]列(?![表举舉出数數])",
        (r"(?:依次|按?[顺順]序)(?:[执執]行|[进進]行|[调調]用|[应應]用|[处處]理)"
         r"|一系列(?:操作|指令|步[骤驟]|[调調]用|事件)"),
        # in total
        (r"一共(?![享同用识識存])|[总總]共|(?<!公)共[计計](?![划劃算])|[总總][计計](?![划劃算])"
         r"|合[计計](?![划劃算])|加起[来來]"),
        r"每一?次|每回|每[当當]",                            # each time
        # starting from / start with. Not 从来 / 从未 / 从不 (never), 从而, 从事.
        (rf"[从從](?![来來未不而事]){_P}{{0,12}}[开開]始"
         rf"|[从從]{_P}{{0,8}}(?:[节節][点點]|[顶頂][点點]|起[点點]|源[点點]|[状狀][态態]|根){_P}{{0,4}}出[发發]"
         rf"|(?<![可所])以{_P}{{1,10}}(?:[开開]始|[为為]起[点點])|(?:起始|[开開]始)[于於]|始[于於]"),
        # occur. Bare 出现 ("appear") is everywhere; only counting, position or modal forms.
        (r"(?:出[现現]|[发發]生)了?(?:[几幾]|多少)次|(?:出[现現]|[发發]生)的?(?:次[数數]|[频頻])"
         r"|(?:首次|第[一二三四五六七八九十0-9]+次|最[后後]一次|再次|每次)出[现現]"
         r"|出[现現]的?位置|出[现現]\s*[0-9一二两兩三四五六七八九十]+\s*次"
         r"|(?:[会會]|才|就|[将將]|可能)出[现現]|[发發]生(?!器|了?什[么麼])"),
        # trace: of a program, not a parcel (追踪快递) or a person (被人跟踪)
        (rf"[跟追][踪蹤]{_P}{{0,6}}(?:[执執]行|[调調]用|[变變]量|代[码碼]|程序|函[数數]|[栈棧]|循[环環]"
         r"|[过過]程|流程|每一步|的值)|堆[栈棧][跟追][踪蹤]|[栈棧]回溯"
         r"|走一遍|推演|(?:手[动動]|手工|人工|逐行)(?:模[拟擬]|[执執]行)"),
        rf"最[终終]的?值|最[终終]{_P}{{1,12}}的值|最[终終]{_P}{{0,6}}(?:等[于於]|[为為]多少|是多少)",
    ),
    # The one negative signal: the precision builder's forms only.
    "simple_markers": (
        r"(?:[请請]|中|里|裡)[提抽]取|[提抽]取(?:出|以下|下列|其中|所有|全部)",
        r"(?:哪|什[么麼])[一种種门門个個款]{0,2}(?:[编編]程|程序(?:[设設][计計])?|程式(?:[设設][计計])?)[语語]言",
        r"[扩擴]展名|[后後][缀綴]名|文件[后後][缀綴]|副[档檔]名",
        r"(?:[请請]|[进進]行|加以)[分归歸][类類]|(?<![可被以])[分归歸][类類](?:[为為]|成(?![本功熟员員])|到)",
        rf"(?:[给給]出|[写寫]出){_P}{{0,20}}(?:git|shell|bash|[终終]端|命令行)\s*(?:命令|指令)",
        (r"(?:哪[个個条條一]|(?<![为為])什[么麼])\s*(?:git|shell|bash)\s*(?:命令|指令)"
         r"|(?:git|shell|bash)\s*(?:命令|指令)(?:是|[为為])(?:什[么麼](?!意思)|哪)"),
        (r"十[进進][制位]的?(?:值|[数數]值|表示)|十[进進][制位](?:是|[为為])(?:多少|[几幾])"
         r"|[转轉](?:[换換])?(?:成|[为為])十[进進][制位]"),
        r"(?:多少|[几幾])[个個]?\s*(?:字[节節]|位元[组組組]|bytes?)",
        r"(?:多少|[几幾])[个個]?\s*kib",
    ),
    "technical_terms": (
        r"互斥(?:[锁鎖]|量|[体體])",                         # mutex; bare 互斥 is "mutually exclusive"
        r"(?<![a-z])go\s*[协協]程",                          # goroutine; bare 协程 is any coroutine
        r"ieee\s?754",
        r"正[则則](?![化项項])|正[规規]表[达達示]式",        # regex; not 正则化 (regularisation)
        r"最近最少使用|最近最久未使用",                      # lru
        r"(?<![a-z0-9])b\s?[+\-]?\s?[树樹]",                 # b-tree
        # replica; bare 副本 is a game dungeon (刷副本) or a document copy
        (r"副本(?:集|[节節][点點]|[数數]|同步|因子|延[迟遲])|只[读讀]副本|[复複]制集"
         r"|(?:主[从從]|[数數]据|[数數]據|[异異]步|同步|多主)[复複]制|[从從](?:库|庫|[节節][点點])"),
        r"[幂冪]等",                                         # idempotent
        # backtracking; bare 回溯 is "look back" (回溯历史)
        r"回溯(?:法|算法|搜索|搜尋|[实實]现)|(?:用|使用|灾难性|災難性|基于|基於)回溯",
        r"[闭閉]包",                                         # closure
        r"[变變]基|衍合",                                    # rebase
        r"[语語][义義意]化?版本",                            # semver
        r"定[时時]任[务務]",                                 # cron (a cron job)
        r"docker\s?文件",                                    # dockerfile
        r"索引|下[标標](?![签簽])",                          # index; not 下标签页 ("the lower tab")
        r"[递遞](?:[归歸]|迴)",                              # recursive
    ),
}

ZH_MARKERS: dict[str, re.Pattern[str]] = {
    signal: re.compile("|".join(f"(?:{p})" for p in patterns))
    for signal, patterns in _VOCABULARY.items()
}


def has_han(text: str) -> bool:
    return HAN.search(text) is not None


def chinese_signals(text: str) -> frozenset[str]:
    """Signals a Chinese marker fires in `text` (already lowercased). Text with
    no Han characters is never matched, so English scoring cannot change."""
    if not has_han(text):
        return frozenset()
    return frozenset(signal for signal, pattern in ZH_MARKERS.items() if pattern.search(text))


def word_count(text: str) -> float:
    """Whitespace words, with each Han character counted as 1/1.5 of a word."""
    han = len(HAN.findall(text))
    if not han:
        return len(text.split())
    return len(_NOT_A_WORD.sub(" ", text).split()) + han / HAN_CHARS_PER_WORD
