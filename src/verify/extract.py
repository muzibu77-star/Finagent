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
- "type"："amount"（金额）、"growth"（同比、较上年、较年初的增减幅度）或 "other"（比率、占比、差额、百分点等）；
- "label"：这个数字对应下列报表科目中的哪一个，原样照抄科目名；找不到对应科目时填 null；
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
