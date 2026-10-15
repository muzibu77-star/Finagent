"""Turn the numbers of a free-text review into structured claims with a model.

The model is shown the review and the line items of the report's consolidated
statements and must name, for every amount or percentage, the line item it
refers to (or none), its fiscal year and whether it is an amount or a
year-on-year change.  It does not judge correctness; that is left to the
deterministic checker.  Each claim is tied back to the review by finding the
number's n-th occurrence in the text.
"""

import json
import re
import sqlite3

from src.verify.checker import STATEMENT, norm

PROMPT = '''下面是一篇关于{company}{year}年年度报告的点评。请找出文中每一个以“亿元”“万元”“百万元”或“%”表示的数字，逐个输出一个 JSON 对象，组成 JSON 数组。

每个对象的字段：
- "number"：原文中的数字，原样照抄，不含单位，例如 "1,741.44"；
- "unit"：亿元、万元、百万元或%；
- "occurrence"：同样写法的数字在文中第几次出现，从 1 开始；
- "phrase"：原文中说明这个数字是什么的指标词，原样照抄，例如 "归属于母公司股东权益"；
- "type"：
  "amount"（某科目的金额）；
  "growth"（某科目的同比、较上年、较年初的增减幅度，单位为%）；
  "change"（某科目较上年、较年初增加或减少的金额）；
  "ratio"（两个金额之比，如毛利率、费用率、资产负债率、占比，单位为%）；
  "other"（百分点变化、无法对应到报表科目的数字等）；
- "label"：amount、growth、change 时，这个数字对应下列报表科目中的哪一个，原样照抄科目名；找不到时填 null；
- "numerator"、"denominator"：ratio 时，分子和分母各由哪些科目相加得到，写成科目名列表，相减的科目前加 "-"，例如毛利率的分子为 ["营业收入", "-营业成本"]、分母为 ["营业收入"]；找不到对应科目时填 null；
- "direction"：growth、change 时，原文说的是增加（增长、提升、上升）填 "up"，减少（下降、减少、下滑）填 "down"，没说方向填 null；
- "year"：数字所属的财年（整数）；说的是上年或年初数时填上一年；
- "scope"："合并"，句中明确说是母公司时填 "母公司"。

只抽取，不要判断数字对错。只输出 JSON 数组，不要其他文字。

报表科目：
{labels}

点评：
{review}'''


def line_items(db: sqlite3.Connection, code: str, year: int) -> list[str]:
    """Distinct line items of the consolidated statements in that year's report."""
    rows = db.execute(
        "SELECT DISTINCT f.row_label, f.title FROM facts f JOIN documents d ON d.id = f.document_id "
        "WHERE d.code = ? AND d.fiscal_year = ? AND d.kind = 'annual' AND f.scale IS NOT NULL",
        (code, year)).fetchall()
    names = []
    for label, title in rows:
        if STATEMENT.search(title or '') and norm(label) and norm(label) not in names:
            names.append(norm(label))
    return names


def parse(content: str) -> list[dict]:
    """The JSON array in the answer; a malformed answer yields no claims."""
    found = re.search(r'\[.*\]', content, re.S)
    if not found:
        return []
    try:
        items = json.loads(found.group())
    except json.JSONDecodeError:
        return []
    return [item for item in items if isinstance(item, dict) and item.get('number')]


def locate(review: str, number: str, occurrence: int) -> int | None:
    """Start of the n-th standalone occurrence of ``number`` in the review."""
    pattern = re.compile(r'(?<![\d.,])' + re.escape(number) + r'(?![\d])')
    starts = [match.start() for match in pattern.finditer(review)]
    if not isinstance(occurrence, int) or not 1 <= occurrence <= len(starts):
        return starts[0] if len(starts) == 1 else None
    return starts[occurrence - 1]


# Nouns that decide what a line item is; a phrase naming one the line item lacks
# was matched to the wrong line item (equity against profit, say).
KEY_NOUNS = ('权益', '利润', '损益', '收入', '成本', '费用', '资产', '负债', '现金', '收益',
             '借款', '存款', '贷款', '应收', '应付', '存货', '投资', '股本', '公积', '准备')


# Different names for the same line item, mapped to one spelling before comparing.
SYNONYMS = (('净资产', '权益'), ('所有者权益', '股东权益'), ('总资产', '资产总计'),
            ('资产总额', '资产总计'))


def matches(phrase: str | None, label: str) -> bool:
    """Whether the review's own words are compatible with the chosen line item."""
    if not phrase:
        return False
    phrase, label = norm(phrase), norm(label)
    # Issuers report "营业总收入" next to "营业收入" (likewise "营业总成本" and
    # "营业成本") with different values, so a "总" that one name has and the other
    # lacks makes them different line items.  Names that both carry a "总", or
    # neither, are left to the rules below.
    if ('总' in phrase) != ('总' in label):
        total, plain = (phrase, label) if '总' in phrase else (label, phrase)
        if total.replace('总', '') == plain:
            return False
    if phrase in label or label in phrase:
        return True
    for a, b in SYNONYMS:
        phrase = phrase.replace(a, b)
        label = label.replace(a, b)
    nouns = [noun for noun in KEY_NOUNS if noun in phrase]
    if not nouns:
        return True  # "同比增长" names no line item, so it cannot contradict one
    if any(noun not in label for noun in nouns):
        return False
    # Abbreviations ("归母净利润") share few characters with the full name but
    # keep its deciding noun.
    if any(noun in phrase and noun in label for noun in KEY_NOUNS):
        return True
    pairs = {label[i:i + 2] for i in range(len(label) - 1)}
    shared = {phrase[i:i + 2] for i in range(len(phrase) - 1)} & pairs
    return bool(pairs) and len(shared) / len(pairs) >= 0.5


# Figures for part of the year or of the group, which the annual consolidated
# line items do not hold, and ratios with a defined method of their own.
NARROWER = re.compile(r'季度|一季|二季|三季|四季|上半年|下半年|单季|分部|板块')
OWN_METHOD = re.compile(r'加权|收益率|资本化|每股|周转|倍')


def checkable(phrase: str | None) -> bool:
    return not (phrase and (NARROWER.search(phrase) or OWN_METHOD.search(phrase)))
