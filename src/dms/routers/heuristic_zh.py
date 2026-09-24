"""Chinese for the heuristic router: the same signals, and a length that counts.

The English router measures length in whitespace words and matches English
markers, so a Chinese prompt -- no spaces, no English -- scored 0 however hard
it was and always went to the low model.

Every change is inert on text without Han characters, so every English score is
exactly what it was (tests/test_heuristic_zh.py holds all 60 frozen):

* **Length.** A Han character counts as 1/1.5 of a word. Parallel corpora put
  English -> Chinese at 1.47-1.62 Han characters per English word (FLORES,
  WMT21-23, TICO-19, the Python docs translation) and natively written Chinese
  nearer 1.35; 1.5 is the central estimate. Punctuation is not counted.
* **Code.** foo（） typed with a Chinese IME is code; the exam blank （） is not.
* **Vocabulary.** Each English marker's Chinese renderings, merged from two
  vocabulary builders (precision-first and recall-first) who worked from the
  English lists alone and never saw the benchmark prompts. Parity is the
  target: the same concepts, no additions. Dropped from the merge:
  - renderings of *accidental* English hits ("improve" contains "prove",
    "consequence" contains "sequence") -- English-side bugs, not concepts;
  - the recall builder's looser forms of `simple_markers`, the one negative
    signal, where a false hit pushes a hard prompt down.

Chinese is written without spaces, so a two-character marker also matches where
one word ends and the next begins: 一下+标题 contains 下标 ("index"), 开发+生产
contains 发生 ("occur"). Each such straddle two adversarial reviews found is
guarded and pinned by a test (docs/heuristic-zh-prereg.md, post-hoc sections).
"""
from __future__ import annotations

import re
import unicodedata

HAN_CHARS_PER_WORD = 1.5

_HAN = (
    r"\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\U00020000-\U0002ee5f\U0002f800-\U0002fa1f\U00030000-\U000323af"
)
# CJK punctuation, the full-width punctuation block (not full-width letters or
# digits, which are words) and the middle dot of 迈克尔·乔丹.
_WIDE_PUNCT = (
    r"\u00b7\u3000-\u3006\u3008-\u303f"
    r"\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65"
)
# Punctuation that belongs to a Chinese sentence only where it touches one:
# ASCII, and General Punctuation (“ ” ’ … —), which also sits inside English words.
_NARROW_PUNCT = r"!-/:-@\[-^`{-~\u2000-\u206f"
HAN = re.compile(f"[{_HAN}]")
_NOT_A_WORD = re.compile(f"[{_HAN}{_WIDE_PUNCT}]")
# Anchored at the start of a run and possessive, so a long run costs linear time.
_PUNCT_BY_CHINESE = re.compile(
    f"(?<=[{_HAN}{_WIDE_PUNCT}])[{_NARROW_PUNCT}]++"
    f"|(?<![{_NARROW_PUNCT}])[{_NARROW_PUNCT}]++(?=[{_HAN}{_WIDE_PUNCT}])"
)
# Han counts as \w, so \bSELECT\b cannot see 这条SELECT语句: part the two scripts.
_HAN_ASCII_EDGE = re.compile(f"(?<=[{_HAN}])(?=[A-Za-z0-9_])|(?<=[A-Za-z0-9_])(?=[{_HAN}])")
# Kangxi / CJK radical code points that PDFs substitute for the ideographs they
# look like: fold them back so they read (and count) as Chinese.
_RADICALS = {
    c: unicodedata.normalize("NFKC", chr(c))
    for c in (*range(0x2E80, 0x2EF4), *range(0x2F00, 0x2FD6))
    if unicodedata.normalize("NFKC", chr(c)) != chr(c)
}
# Full-width brackets fold to ASCII only when attached to code: foo（）, a【i】.
_WIDE_BRACKETS = {
    "（": "(", "）": ")", "［": "[", "］": "]", "｛": "{", "｝": "}",
    "【": "[", "】": "]", "《": "<", "》": ">", "＝": "=", "＜": "<", "＞": ">",
}
_CODE_NEIGHBOURS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_()[]{}<>=-"
)

# One unit of a bounded gap: an ASCII token -- identifier, dotted name, version,
# path -- or one other character; spaces around units are free (pangu spacing
# puts one on each side of every Latin token). A gap stops at sentence and clause
# punctuation (，。；？！ , . ; ? !) and at a newline, never inside a token like
# main.py. The branches cannot match the same text and every repeat is
# possessive, so no gap can backtrack.
_UNIT = r"(?:[a-z0-9_$]++(?:[.\-/:][a-z0-9_$]++)*+|[^\sa-z0-9_$，。,.；;？?！!])"


def _gap(lo: int, hi: int) -> str:
    return rf"[ \t]*+(?:{_UNIT}[ \t]*+){{{lo},{hi}}}"


# ---------------------------------------------------------------- why (为什么)
# After a word ending in 为 -- 认为, 称为, 作为, 成为, 变为, 命名为 -- 为什么 is
# "think / called / as / become + what". Each is blocked unless a longer word
# makes it a why again: 确认为什么, 名称为什么, 操作为什么, 生成为什么.
_NOT_AS_WHAT = (
    r"(?<![^确確][认認])(?<!^[认認])"
    r"(?<![^名][称稱])(?<!^[称稱])"
    r"(?<![^操工动動合写寫创創制製运運][作])(?<!^作)"
    r"(?<![^完生组組造构構形合]成)(?<!^成)"
    r"(?<![^改][变變])(?<!^[变變])(?<![命取起改]名)"
    r"(?<![以因视視设設])"
)
_WHY = r"(?:什[么麼]|甚[么麼]|啥|嘛)"
# After 设置为 / 改为 / 转为 / 翻译为 / 初始化为 / 定义为 / 分为 / 替换为 / 映射为,
# 为什么 is "to what" when the clause ends or a noun follows -- and still "why"
# before a verb: 设置为什么不生效.
_TO_WHAT = (
    r"(?:(?<=[置转轉改换換分义義译譯化射])|(?<=[^确確]定))[为為爲](?:{why}|何)"
    r"(?=\s*(?:$|\n|[?？。，,!！]|比[较較]|合适|合適|好|呢|都|也|值(?!得)|[级級]别|[类類]型"
    r"|格式|名字|英文|中文|[东東]西|意思|[内內]容|[状狀][态態]|[颜顏]色|角色|程度|[样樣]))"
).replace("{why}", _WHY)

# "Starting from" a position -- 第 3 个, 头, a node, a file -- not "since" a
# time: 从上周开始 / 从 v2.3 开始 is "since last week / since v2.3".
_POSITION = (
    r"(?:第|[头頭]|首|末尾|[开開][头頭]|[这這][里裡儿兒]|哪[里裡儿兒]|元素|[节節][点點]|位置|索引|下[标標]"
    r"|行|列|[层層]|根|起[点點]|最[左右]|[左右]边|[a-z0-9_$]+\.[a-z]{1,5}(?![a-z0-9_]))"
)
_FROM = r"[从從](?![来來未而事]|不(?!同))"

_VOCABULARY: dict[str, tuple[str, ...]] = {
    "reasoning_markers": (
        # why
        rf"{_NOT_AS_WHAT}(?!{_TO_WHAT})[为為爲]{_WHY}(?![时時]候)",
        r"(?<!因)[为為爲]毛(?![利泽澤巾衣病])",               # slang; not 因为+毛利率
        r"所以[为為](?:什[么麼]|啥|何)|因[为為]什[么麼](?![都也])",
        rf"{_NOT_AS_WHAT}(?!{_TO_WHAT})[为為爲]何(?![种種物方时時])",
        (r"怎[么麼](?:[会會](?![用写寫])|[这這][么麼]|那[么麼]|又|突然|老是|[总總]是|就"
         r"|[还還](?:是|在|[会會]|[报報]|[没沒]|不)|一直|[没沒]有?|老(?!化)|[总總](?![结結体體共]))"
         r"|咋(?:就|[会會]|[这這][么麼]|那[么麼]|又|老|[总總]|突然|[还還])"
         r"|[干幹][嘛吗麼](?:要|非|不|[还還]|老是|[总總]是|[这這][么麼样樣]|那[么麼样樣])|凭(?:什[么麼]|啥)"),
        # explain. Not 解释器 (interpreter), 解释权, 讲解员, nor straddles: 了解+释放, 讲+下周.
        r"解[释釋](?!器|型|性?[语語]言|[执執]行|[权權]|放|[义義]|疑|然|[怀懷]|出|迦)",
        (r"[讲講](?:解(?![员員决決])|[讲講]|一下|下(?![去来來午载載周週个個班课課次])|一[讲講]|清楚|明白|透)"
         r"|[阐闡](?:述|明|[释釋])|(?<![了理注])解[读讀](?![写寫取])|科普"),
        # 说明 as a request -- not the noun (使用说明), nor a preamble (先说明一下)
        (r"(?:[请請]|并|並|[详詳][细細]|具[体體]|分[别別]|逐一|[帮幫]我)[说說]明"
         r"|(?<![用明书書有先充外便])[说說]明(?:一下|原因|理由|[为為]什|[为為]何|如何|怎[么麼]|是否)"),
        # prove. Not the paperwork 证明 (实习证明, 工作证明...), nor 考证+明朝, 无论+证书.
        (r"(?<![开開具职職入历歷份住产產验驗认認论論凭憑签簽习習保税稅款罪位读讀考作张張健康籍假])[证證]明"
         r"(?![书書材信函人天年日早晚白细細显顯文])"
         r"|(?<![无無讨討不评評谈談])[论論][证證](?![书書券件据據])"
         r"|怎[么麼]证(?![件书書明])|(?<![验驗认認保论論考查签簽凭憑求印办辦])证一?下"
         r"|求[证證]\s*[:：]|[，,。；;]\s*求[证證]|[试試][证證]"),
        # derive. Not 列表推导式, 类型推导 (type inference), 强推+导致, 顺手+推到.
        (r"(?<![强強型数數])推[导導](?![致式])"
         r"|(?<![顺順随隨快新助抓联聯])手推(?![车車荐薦送广廣进進出到给給流动動])"
         rf"|推一?下{_gap(0, 10)}(?:公式|期望|概率|梯度|通[项項]|方差|上界|下界|[结結]论|式子|[递遞]推)"),
        r"[权權]衡|取[舍捨]|怎[么麼]折[中衷]|折[中衷]一下|之[间間]的?折[中衷]",   # trade-off
        r"根本原因|根因|根源|病根(?![据據本])",               # root cause; bare 原因 is "reason"
        r"[复複][杂雜][度性]",                               # complexity; bare 复杂 is "complicated"
        rf"(?<![杀殺])死[锁鎖]|(?:互相|相互|[线線]程|[进進]程|[协協]程|事[务務]){_gap(0, 4)}[锁鎖]死",
        # vulnerable: "is vulnerable" = has a hole / is exposed to an attack. Bare 漏洞 is
        # "vulnerability"; not 有没有+漏洞扫描, 利用率, 依赖注入.
        (rf"易[受被]{_gap(0, 10)}攻[击擊]"
         rf"|容易(?:受到|遭受|被){_gap(0, 8)}(?:攻[击擊]|(?<![赖賴])注入|利用(?!率))"
         rf"|(?:(?<![所还還])有(?!没有|沒有|[点點])|存在){_gap(0, 10)}漏洞(?![扫掃报報库庫])"
         rf"|[会會](?:不[会會])?(?:被|受到){_gap(0, 6)}(?:sql|xss|csrf|命令|脚本|腳本)\s*(?:注入|攻[击擊])"),
        r"最[坏壞差糟]糕?的?情[况況形]",                     # worst-case; not 做好最坏的打算
        r"(?<![一以如])下[确確]?界(?![面限定线線])",         # lower bound; not 一下+界面
        # violated: past or passive, as English keys "violated" only
        (rf"(?:被|遭)[违違][反背]|[违違][反背]了"
         rf"|(?:不[变變](?:量|式|性)|[约約]束|契[约約]|前置条件|后置条件){_gap(0, 8)}破[坏壞]"
         rf"|破[坏壞]了?{_gap(0, 6)}(?:不[变變](?:量|式|性)|[约約]束|契[约約])"),
        # guarantee: asked about (如何保证, 能保证…吗) -- an imperative 保证 is "make sure"
        (r"(?:如何|怎[么麼样樣]|能否|能不能|是否|可以|无法|無法|不能|能)保[证證](?![金书書人明])"
         r"|(?<![我社医醫确確担擔环環参參投续續质質劳勞])(?<!尽量)(?<!儘量)(?<!质量)(?<!質量)"
         rf"保[证證](?![金书書人明]){_gap(0, 12)}(?:吗|嗎|么|麼|[?？])"),
        # name the cause / the bug: a request, not a narrated or negated event, and
        # not 给出错误 ("emit an error") or 写出 bug ("write a bug")
        (rf"(?<![没沒未不])(?:指出|[说說]出|[点點]出)(?!了){_gap(0, 18)}"
         r"(?:原因|起因|成因|bug|缺陷|[错錯][误誤]|毛病|[问問][题題]所在)"),
        rf"(?<![没沒未不])(?:[给給]出|[写寫]出)(?!了){_gap(0, 12)}(?:原因|起因|成因)",
        rf"[说說]一?下{_gap(0, 8)}(?:原因|bug|缺陷)(?=\s*(?:是[啥什]|在哪|$|\n|[?？。，,：:！!；;]|吧|呢))",
        # optimi- (optimise). Not 最优 -- "optimal" does not contain "optimi".
        r"[优優]化|最佳化|(?<![强強协協])[调調][优優](?![先势勢秀])",
        (r"理[论論](?:上|值|性|最|极限|極限|峰值|上限|下限|分析|[计計]算|[复複][杂雜]度|性能|吞吐|速度"
         r"|[带帶][宽寬]|延[迟遲]|加速比|依[据據]|基[础礎]|角度|[层層]面|并[发發]|並發|耗[时時]|模型"
         r"|\s*(?:qps|iops|tps))"),
    ),
    "multi_step": (
        # repeat. Bare 重复 is mostly "duplicate" (去除重复元素, 去除重复之前).
        (r"重[复複](?:地|[执執]行|[运運]行|[调調]用|操作|[进進]行|做|以上|上述|上面|前面|此|[这這]|[该該]"
         r"|步[骤驟]|[过過]程|直?到(?![达達])|直至|上一步|[刚剛]才|第\s*[0-9一二三四五六七八九十]+\s*步"
         r"|\s*[0-9a-z一二两兩三四五六七八九十百千几幾]+\s*[次遍轮輪])"
         r"|不[断斷]重[复複]|反[复覆](?!无常|無常|横跳|橫跳)|一[遍次]又一[遍次]"),
        # step by step. 逐步 alone is "gradually" (经济逐步恢复); 分步走 is a phased rollout.
        r"一步一?步|分步[骤驟]|按步[骤驟]",
        (r"(?:逐步地?|分步)(?:推[导導]?|推理|分析|[计計]?算|解[释釋]|[说說]明|演示|展示|思考|求解|[执執]行"
         r"|模[拟擬]|[讲講]解?|[给給]出|列出|[跟追][踪蹤]|排查|[调調]试|定位|拆解|解答)"),
        # sequence. Not 序列化, 序列号, 时间序列, 排序+列出, nor a column of numbers (浮点数+列).
        (r"(?<!时间)(?<!時間)(?<![排升降倒])序列(?![化号號表出])"
         r"|(?<![参參函变變整行点點小奇偶负負实實复複多少总總次指人分常系倍天页頁字帧幀位基])[数數]列(?![表举舉出数數])"
         r"|[时時]序[图圖]"),
        (r"(?:依次|按?[顺順]序)(?:[执執]行|[进進]行|[调調]用|[应應]用|[处處]理)"
         r"|一系列(?:操作|指令|步[骤驟]|[调調]用|事件)"),
        # in total. Not 汇总+共享, 整合+计费, nor a totals row (最后加一行合计).
        (r"一共(?![享同用识識存])|(?<![汇匯])[总總]共(?![享同])|(?<!公)共[计計](?![划劃算])"
         r"|(?<![整结結组組聚适適配混综綜联聯集符汇匯])[总總合][计計](?:[是为為])?(?:多少|[几幾])"),
        # each time. Bare 每次 is just as much "every time", which English does not key.
        r"每一次|每[当當]|每回",
        # starting from a position / start with. Not 从来 / 从未 / 从不 (never) -- but 从不同 is
        # "from different" -- nor "since" (从上周开始), nor 以后 / 以及 / 以下, 刚开始 / 一开始.
        (rf"{_FROM}{_gap(0, 6)}{_POSITION}{_gap(0, 6)}(?<![刚剛一])[开開]始(?![时時日])"
         rf"|[从從]\s*[0-9]+\s*(?:[个個号號位]\s*)?[开開]始"
         rf"|{_FROM}{_gap(0, 6)}{_POSITION}{_gap(0, 4)}(?<![一发發引拿提想兴興])起(?![来來床])"
         rf"|[从從]{_gap(0, 8)}(?:[节節][点點]|[顶頂][点點]|起[点點]|源[点點]|[状狀][态態]|根){_gap(0, 4)}出[发發]"
         rf"|(?<![可所难難加予])以(?![及后後前上下来來便免外为為致往]){_gap(1, 10)}(?:[开開]始|[为為]起[点點])"
         r"|(?:起始|[开開]始)[于於]|始[于於]"),
        # occur. Bare 出现 is "appear" (会出现一个弹窗); not 开发+生产, 开发+生态, 触发+生成.
        (r"(?:出[现現]|[发發]生)了?(?:[几幾]|多少)次|(?:出[现現]|[发發]生)的?(?:次[数數]|[频頻])"
         r"|(?:首次|第[一二三四五六七八九十0-9]+次|最[后後]一次|再次|每次)出[现現]"
         r"|出[现現]的?位置|出[现現]\s*[0-9一二两兩三四五六七八九十]+\s*次"
         r"|(?<![开開研触觸群转轉头頭批])[发發]生(?![产產成效命活器]|了?什[么麼])"),
        # trace: of a program -- not a parcel (追踪快递), 手动执行 ("run it by hand"), nor
        # 跟 = "with" (跟下面的代码)
        (rf"[跟追][踪蹤]{_gap(0, 6)}(?:[执執]行|[调調]用|[变變]量|代[码碼]|程序|函[数數]|[栈棧]|循[环環]"
         r"|[过過]程|流程|每一步|的值|[请請]求|[链鏈]路|指[针針])"
         r"|堆[栈棧][跟追][踪蹤]|[栈棧]回溯|推演|(?:手[动動]|手工|人工|逐行)模[拟擬]|逐行[执執]行"
         rf"|[跟追]一?下(?![面边邊方列一个個次游载載层層]){_gap(0, 10)}"
         r"(?:[调調]用|[执執]行|[变變]量|代[码碼]|[请請]求)"
         r"|(?:[报報][错錯]|异常|異常|[错錯][误誤]|崩[溃潰])的?堆[栈棧]|堆[栈棧]信息"),
        (rf"最[终終]的?值|最[终終]{_gap(1, 12)}的值|最[后後]的值(?!班)"
         r"|最[终終]\s*[a-z_][a-z0-9_]*\s*(?:等[于於]|[为為]多少|是多少)"),
    ),
    # The one negative signal: precision forms only, and no broader than English
    # ("give the git command", "decimal value of", "how many bytes").
    "simple_markers": (
        r"(?:[请請]|中|里|裡)[提抽]取|[提抽]取(?:出|以下|下列|其中|所有|全部)",
        r"(?:哪|什[么麼])[一种種门門个個款]{0,2}(?:[编編]程|程序(?:[设設][计計])?|程式(?:[设設][计計])?)[语語]言",
        r"[扩擴]展名(?![称稱字])|[后後][缀綴]名|文件[后後][缀綴]|副[档檔]名",
        (rf"(?:[请請]|[进進]行|加以)[分归歸][类類]"
         rf"|(?:[请請]|[帮幫]我){_gap(0, 20)}[分归歸][类類](?:[为為]|成(?![本功熟员員])|到)"),
        rf"(?:[给給]出|[写寫]出){_gap(0, 20)}(?:git|shell|bash|[终終]端|命令行)\s*(?:命令|指令)",
        r"十[进進][制位]的?(?:值|[数數]值|表示)|十[进進][制位](?:是|[为為])(?:多少|[几幾])",
        # how many bytes -- not 多了几个字节 ("a few bytes"), not 字节跳动 (ByteDance)
        r"(?:多少|(?<![了好这這那])[几幾])[个個]?\s*(?:字[节節](?!跳)|位元[组組組]|bytes?)",
        r"(?:多少|[几幾])[个個]?\s*kib",
    ),
    "technical_terms": (
        r"互斥(?:[锁鎖]|量|[体體])",                         # mutex; bare 互斥 is "mutually exclusive"
        r"(?<![a-z])go\s*[协協]程",                          # goroutine; bare 协程 is any coroutine
        # regex; not 正则化 / L2 正则 (regularisation), nor 为正，则输出
        (r"(?<![为為])(?<!l1 )(?<!l2 )(?<!l1)(?<!l2)正[则則](?![化项項返系强強参參惩懲])"
         r"|正[规規]表[达達示]式"),
        r"最近最少使用|最近最久未使用",                      # lru
        r"(?<![a-z0-9])b\s?[+\-]?\s?[树樹]",                 # b-tree
        # replica. Bare 副本 is a game dungeon (刷副本); 一副 is a measure word; 数据复制到 /
        # 复制粘贴 is copying; 从+库存 / 从+库里 is "from the inventory / library".
        (r"(?:[多两兩三几幾各主0-9二]|每[个個]?)副本(?![来來要打])"
         r"|副本(?:集|[节節][点點]|[数數]|同步|因子|延[迟遲]|之?[间間]|分配|机制|機制|分布)"
         r"|只?[读讀]副本|[复複][制製]集"
         r"|(?:主[从從]|[数數]据|[数數]據|[异異]步|同步|多主)[复複][制製](?![到给給进進粘黏])"
         r"|主[从從](?:延[迟遲]|同步|切[换換]|架构|架構)|[从從][库庫](?![存里裡房])"),
        r"[幂冪]等(?![于於价價同])",                         # idempotent; not 幂+等于
        # backtracking; bare 回溯 is "look back" (回溯历史, a retrospective)
        r"回溯(?:法|算法|搜索|搜尋|[实實]现|加|\+|\s*剪枝)|(?:用|使用|灾难性|災難性|基于|基於)回溯",
        r"(?<![关關封])[闭閉]包(?![括含])",                  # closure; not 关闭+包括
        r"(?<![改转轉突不演])[变變]基(?![本础礎因金])|衍合",   # rebase; not 改变+基本
        r"[语語][义義意]化?版本",                            # semver
        r"定[时時]任[务務]",                                 # cron (a cron job)
        r"docker\s?文件(?![夹夾系权權])",                    # dockerfile; not a docker folder
        # index; not 搜索引擎 / 探索+引入, nor 一下+标题, 看下+标红, 上下标
        (r"(?<![搜检檢探思摸勒线線])索引(?!擎)"
         r"|(?<![一以如录錄底之上])下[标標](?![签簽题題点點准準志誌记記注註识識语語红紅出])"),
        r"(?<![传傳])[递遞](?:[归歸](?!一)|迴)",             # recursive; not 传递+归一化
    ),
}

ZH_MARKERS: dict[str, re.Pattern[str]] = {
    signal: re.compile("|".join(f"(?:{p})" for p in patterns))
    for signal, patterns in _VOCABULARY.items()
}


def _fold(text: str) -> str:
    return text.translate(_RADICALS)


def has_han(text: str) -> bool:
    return HAN.search(_fold(text)) is not None


def chinese_signals(text: str) -> frozenset[str]:
    """Signals a Chinese marker fires in `text` (already lowercased). Text with
    no Han characters is never matched, so English scoring cannot change."""
    text = _fold(text)
    if not HAN.search(text):
        return frozenset()
    return frozenset(signal for signal, pattern in ZH_MARKERS.items() if pattern.search(text))


def code_view(prompt: str) -> str:
    """The prompt as the code detector sees it. With Han present, full-width
    brackets attached to code fold to ASCII -- foo（） -> foo() -- and Latin is
    parted from Han so keywords have word boundaries (这条SELECT语句)."""
    if not has_han(prompt):
        return prompt
    out: list[str] = []
    for ch in prompt:
        if ch in _WIDE_BRACKETS and out and out[-1] in _CODE_NEIGHBOURS:
            ch = _WIDE_BRACKETS[ch]
        out.append(ch)
    return _HAN_ASCII_EDGE.sub(" ", "".join(out))


def word_count(text: str) -> float:
    """Whitespace words, with each Han character counted as 1/1.5 of a word."""
    text = _fold(text)
    han = len(HAN.findall(text))
    if not han:
        return len(text.split())
    rest = _NOT_A_WORD.sub(" ", _PUNCT_BY_CHINESE.sub(" ", text))
    return len(rest.split()) + han / HAN_CHARS_PER_WORD
