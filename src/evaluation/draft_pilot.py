"""Pilot: how often does a model-written earnings review state unsupported numbers?

A general model is given the raw text of a report's consolidated statements and
asked for a short review.  Every amount and percentage in the review is then
checked by code against the facts of that report: an amount must equal some
disclosed value at the precision written, a percentage must equal a disclosed
percentage or the year-on-year change of some statement row.  Numbers without
support are candidates for manual review, not confirmed errors: a correctly
computed ratio the report never prints (a gross margin, say) is also flagged.
"""

import argparse
from decimal import Decimal, InvalidOperation
import json
import logging
from pathlib import Path
import re
import sqlite3

from src.evaluation.seed_claims import amount
from src.model.api_client import ApiClient

log = logging.getLogger(__name__)
STATEMENTS = ('合并资产负债表', '合并利润表', '合并现金流量表')
SCALES = {'亿元': Decimal('100000000'), '万元': Decimal('10000'), '百万元': Decimal('1000000')}
STATED = re.compile(r'(?<![\d.])(-?\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|百万元|%|个百分点)')
PROMPT = '''你是证券公司的行业分析师。下面是{company}{year}年年度报告中合并财务报表的原文（由PDF提取，排版可能错乱）。
请据此写一篇600字左右的年报点评，要求：
1. 覆盖收入、利润、主要费用、现金流和资产负债状况；
2. 金额统一换算为“亿元”，保留两位小数；同比变化用百分比，保留一位小数；
3. 只使用原文中的数据，不要引用原文之外的信息，不要给出投资评级；
4. 直接输出点评正文。

{text}'''

CLOSED_BOOK = '''你是证券公司的行业分析师。请写一篇600字左右的{company}{year}年年度报告点评，要求：
1. 覆盖收入、利润、主要费用、现金流和资产负债状况，并给出具体数字；
2. 金额统一用“亿元”，保留两位小数；同比变化用百分比，保留一位小数；
3. 不要给出投资评级；直接输出点评正文。'''


def statement_pages(db: sqlite3.Connection, document_id: str, limit: int = 12) -> list[int]:
    """Pages whose opening lines name a consolidated statement, plus the next page."""
    found = []
    for page, text in db.execute('SELECT page, text FROM pages WHERE document_id = ? '
                                 'ORDER BY page', (document_id,)):
        head = re.sub(r'\s', '', text[:400])
        if any(name in head for name in STATEMENTS) and '母公司' not in head[:60]:
            found.extend([page, page + 1])
    return sorted(set(found))[:limit]


def row_changes(db: sqlite3.Connection, document_id: str) -> list[Decimal]:
    """Year-on-year change of every table row that holds exactly two amounts.

    The first amount in reading order is taken as the current year, which is
    how comparative statements are laid out.
    """
    rows: dict[tuple, list[Decimal]] = {}
    for page, title, label, text, scale in db.execute(
            'SELECT page, title, row_label, text, scale FROM facts WHERE document_id = ? '
            "AND scale IS NOT NULL AND row_label != '' ORDER BY page, start", (document_id,)):
        rows.setdefault((page, title, label), []).append(amount(text, scale))
    return [(values[0] - values[1]) / abs(values[1]) * 100
            for values in rows.values() if len(values) == 2 and values[1] != 0]


def tolerance(number: str) -> Decimal:
    """Half a unit of the last written digit."""
    places = len(number.split('.')[1]) if '.' in number else 0
    return Decimal(1).scaleb(-places) / 2


def check(review: str, amounts: list[Decimal], percentages: list[Decimal]) -> list[dict]:
    """Each stated number with ``supported`` true or false."""
    checked = []
    for match in STATED.finditer(review):
        number, unit = match.group(1).replace(',', ''), match.group(2)
        try:
            stated = abs(Decimal(number))
        except InvalidOperation:
            continue
        allowed = tolerance(number)
        if unit in SCALES:
            supported = any(abs(abs(value) / SCALES[unit] - stated) <= allowed for value in amounts)
        elif unit == '%':
            supported = any(abs(abs(value) - stated) <= allowed for value in percentages)
        else:
            continue  # a difference in percentage points is derived from two ratios
        checked.append({'start': match.start(1), 'text': match.group(), 'unit': unit,
                        'supported': supported,
                        'context': review[max(0, match.start() - 28):match.end() + 6]})
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--splits', type=Path, default=Path('configs/issuer_splits.json'))
    parser.add_argument('--split', action='append', required=True,
                        help='issuer split to write for; may be repeated')
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--budget-usd', type=float, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--closed-book', action='store_true',
                        help='give the model no report text, only the company and year')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = ApiClient(args.model, args.budget_usd, args.output_dir / 'api_cache.jsonl')
    db = sqlite3.connect(f'file:{args.library}?mode=ro', uri=True)
    splits = json.loads(args.splits.read_text())
    codes = [row['code'] for name in args.split for row in splits[name]]
    records = []
    for code in codes:
        for document_id, company, year in db.execute(
                "SELECT id, publisher, fiscal_year FROM documents WHERE code = ? "
                "AND kind = 'annual' ORDER BY fiscal_year", (code,)).fetchall():
            pages = statement_pages(db, document_id)
            if not pages and not args.closed_book:
                records.append({'document_id': document_id, 'company': company, 'year': year,
                                'skipped': 'no consolidated statement page found'})
                continue
            text = '\n'.join(row[0] for page in pages for row in db.execute(
                'SELECT text FROM pages WHERE document_id = ? AND page = ?', (document_id, page)))
            prompt = (CLOSED_BOOK.format(company=company, year=year) if args.closed_book
                      else PROMPT.format(company=company, year=year, text=text))
            answer = client.chat([{'role': 'user', 'content': prompt}], max_tokens=2000)
            amounts = [amount(value, scale) for value, scale in db.execute(
                'SELECT text, scale FROM facts WHERE document_id = ? AND scale IS NOT NULL',
                (document_id,))]
            percentages = row_changes(db, document_id) + [
                Decimal(value.strip('()%-').replace(',', '')) for (value,) in db.execute(
                    "SELECT text FROM facts WHERE document_id = ? AND unit = '%'", (document_id,))
                if re.fullmatch(r'[\d,.]+', value.strip('()%-'))]
            numbers = check(answer['content'], amounts, percentages)
            records.append({'document_id': document_id, 'company': company, 'year': year,
                            'pages': pages, 'review': answer['content'],
                            'finish_reason': answer['finish_reason'], 'numbers': numbers})
            log.info('%s %s: %d numbers, %d unsupported, spent $%.3f', company, year, len(numbers),
                     sum(not n['supported'] for n in numbers), client.spent)
    (args.output_dir / 'records.json').write_text(json.dumps(records, ensure_ascii=False, indent=1))
    written = [r for r in records if 'numbers' in r]
    total = sum(len(r['numbers']) for r in written)
    unsupported = sum(not n['supported'] for r in written for n in r['numbers'])
    summary = {'model': args.model, 'split': args.split, 'closed_book': args.closed_book,
               'reviews': len(written),
               'skipped': len(records) - len(written), 'numbers': total,
               'unsupported': unsupported, 'cost_usd': round(client.spent, 4)}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
