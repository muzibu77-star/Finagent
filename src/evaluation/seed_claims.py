"""Seed claims for the evaluation set: correct statements and injected errors.

Statements are generated from main-statement facts of the source library, so
every claim carries the facts it rests on.  Each error is injected by one
recorded edit of a correct claim.  The sentences are templates; they test the
checking logic, not how real drafts are worded.
"""

import argparse
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
import json
from pathlib import Path
import random
import re
import sqlite3

STATEMENT = re.compile(r'合并(?:及(?:母)?公司)?(利润表|资产负债表|现金流量表)')
CURRENT = re.compile(r'本期|本年|期末|年末|本报告期')
PRIOR = re.compile(r'上期|上年|期初|年初')
YI = Decimal('100000000')
KEY_DATA = re.compile(r'主要会计数据')
# Ratios, per-share figures and counts are not amounts in yuan.
RATIO = re.compile(r'率|比|每股|倍|天数|人数|股数|户数|份额')
LATEST_YEAR = 2026
PLAIN_METRIC = re.compile(r'[\u4e00-\u9fff、（）()]{2,24}')


def role(header: str, fiscal_year: int) -> str | None:
    """Whether a column holds the report's own year or the comparative year."""
    if PRIOR.search(header):
        return 'prior'
    if CURRENT.search(header):
        return 'current'
    years = set(re.findall(r'(20\d{2})\s*年', header))
    if years == {str(fiscal_year)}:
        return 'current'
    return 'prior' if years == {str(fiscal_year - 1)} else None


def clean_metric(label: str) -> str:
    """A row label as an analyst would name it: no numbering or filing hints."""
    label = re.sub(r'\s', '', label)
    label = re.sub(r'[（(][^（()）]*(?:填列|列示|号)[^（()）]*[)）]', '', label)
    label = re.sub(r'^(?:[一二三四五六七八九十]+、|\d+[.、．]|[（(][一二三四五六七八九十\d]+[)）])', '', label)
    return re.sub(r'^(?:其中|加|减)[:：]', '', label).strip(':：')


def amount(text: str, scale: str) -> Decimal:
    """A disclosed table value in yuan; parentheses mean negative."""
    text = text.replace(',', '').strip()
    negative = text.startswith('(') and text.endswith(')')
    value = Decimal(text.strip('()')) * Decimal(scale)
    return -value if negative else value


def in_yi(value: Decimal) -> str:
    return str((value / YI).quantize(Decimal('0.01'), ROUND_HALF_UP))


def growth(current: Decimal, prior: Decimal) -> str:
    return str(((current - prior) / abs(prior) * 100).quantize(Decimal('0.1'), ROUND_HALF_UP))


def pairs(db: sqlite3.Connection) -> list[dict]:
    """Rows of consolidated main statements with both years on one page."""
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT f.id, f.page, f.text, f.digits, f.scale, f.row_label, f.column_header, f.title, "
        "d.id AS document_id, d.code, d.publisher, d.fiscal_year, d.published "
        "FROM facts f JOIN documents d ON d.id = f.document_id WHERE d.kind = 'annual' "
        "AND f.scale IS NOT NULL AND f.row_label != '' AND f.column_header IS NOT NULL "
        "AND (f.scope = '合并' OR f.title LIKE '%主要会计数据%') ORDER BY f.id").fetchall()
    grouped: dict[tuple, dict] = {}
    for row in rows:
        # The key-figures table near the front of every report is group-level and
        # comparative, so it can stand in where the statements were not bound.
        if not STATEMENT.search(row['title'] or '') and not KEY_DATA.search(row['title'] or ''):
            continue
        side = role(row['column_header'], row['fiscal_year'])
        if side is None:
            continue
        key = (row['document_id'], row['page'], row['title'], row['row_label'])
        # A label used twice on a page (e.g. "其他") cannot be told apart later.
        grouped.setdefault(key, {}).setdefault(side, []).append(row)
    found = []
    for (document_id, page, title, label), sides in grouped.items():
        if len(sides.get('current', [])) != 1 or len(sides.get('prior', [])) != 1:
            continue
        current, prior = sides['current'][0], sides['prior'][0]
        metric = clean_metric(label)
        # A label with digits or stray punctuation is a merged or broken cell.
        if not PLAIN_METRIC.fullmatch(metric) or RATIO.search(metric):
            continue
        found.append({
            'document_id': document_id, 'page': page, 'code': current['code'],
            'company': current['publisher'], 'fiscal_year': current['fiscal_year'],
            'as_of': current['published'],
            'statement': ('合并' + STATEMENT.search(title).group(1) if STATEMENT.search(title)
                          else '主要会计数据'),
            'metric': metric, 'current_fact': current['id'], 'prior_fact': prior['id'],
            'current': amount(current['text'], current['scale']),
            'prior': amount(prior['text'], prior['scale']),
            'current_digits': current['digits']})
    return found


def sentence(prefix: str, number: str, unit: str) -> dict:
    return {'text': f'{prefix}{number}{unit}。', 'span': [len(prefix), len(prefix) + len(number)],
            'stated': number, 'unit': unit}


def direct(pair: dict) -> dict | None:
    """"某公司2024年营业收入为1,741.44亿元。" """
    when = (f"{pair['fiscal_year']}年末" if pair['statement'] == '合并资产负债表'
            else f"{pair['fiscal_year']}年")
    if abs(pair['current']) < YI / 100:  # would read "0.00亿元"
        return None
    claim = sentence(f"{pair['company']}{when}{pair['metric']}为", in_yi(pair['current']), '亿元')
    return {**claim, 'kind': 'direct', 'label': 'consistent', 'error': None,
            'facts': [pair['current_fact']], 'expected': in_yi(pair['current'])}


def derived(pair: dict) -> dict | None:
    """"某公司2024年营业收入同比增长15.7%。"; needs a positive base year."""
    if pair['prior'] <= 0 or pair['statement'] == '合并资产负债表':
        return None
    change = Decimal(growth(pair['current'], pair['prior']))
    word = '同比增长' if change >= 0 else '同比下降'
    claim = sentence(f"{pair['company']}{pair['fiscal_year']}年{pair['metric']}{word}",
                     str(abs(change)), '%')
    return {**claim, 'kind': 'derived', 'label': 'consistent', 'error': None,
            'facts': [pair['current_fact'], pair['prior_fact']], 'expected': str(abs(change))}


def restate(claim: dict, number: str, unit: str, error: str, note: str) -> dict:
    prefix = claim['text'][:claim['span'][0]]
    return {**claim, **sentence(prefix, number, unit), 'label': 'inconsistent', 'error': error,
            'injected': note}


def wrong_digit(claim: dict, rng: random.Random) -> dict:
    """E1: one leading digit changed, far outside the rounding tolerance."""
    digits = [i for i, char in enumerate(claim['stated']) if char.isdigit()]
    position = rng.choice(digits[:max(1, len(digits) - 2)])
    old = claim['stated'][position]
    new = rng.choice([d for d in '123456789' if d != old])
    number = claim['stated'][:position] + new + claim['stated'][position + 1:]
    return restate(claim, number, claim['unit'], 'E1', f'digit {old}->{new} at {position}')


def wrong_unit(claim: dict, rng: random.Random) -> dict | None:
    """E2: the number is kept and the unit changed by orders of magnitude."""
    if claim['unit'] != '亿元':
        return None
    unit = rng.choice(['万元', '百万元', '万亿元'])
    return restate(claim, claim['stated'], unit, 'E2', f'unit 亿元->{unit}')


def wrong_period(claim: dict, pair: dict) -> dict | None:
    """E3: last year's amount stated as this year's."""
    if claim['kind'] != 'direct' or in_yi(pair['prior']) == in_yi(pair['current']):
        return None
    return {**restate(claim, in_yi(pair['prior']), claim['unit'], 'E3', 'prior-year amount'),
            'facts': [pair['current_fact'], pair['prior_fact']]}


def wrong_growth(claim: dict, pair: dict) -> dict | None:
    """E5: growth divided by the current year instead of the base year."""
    if claim['kind'] != 'derived':
        return None
    mistaken = ((pair['current'] - pair['prior']) / abs(pair['current']) * 100).quantize(
        Decimal('0.1'), ROUND_HALF_UP)
    if abs(abs(mistaken) - Decimal(claim['stated'])) <= Decimal('0.1'):
        return None
    return restate(claim, str(abs(mistaken)), '%', 'E5', 'divided by current-year amount')


def parent_amounts(db: sqlite3.Connection) -> dict[tuple, Decimal]:
    """Current-year amounts of parent-company statements, by report, statement and metric."""
    db.row_factory = sqlite3.Row
    found: dict[tuple, list[Decimal]] = {}
    for row in db.execute(
            "SELECT f.text, f.scale, f.row_label, f.column_header, f.title, d.id, d.fiscal_year "
            "FROM facts f JOIN documents d ON d.id = f.document_id WHERE d.kind = 'annual' "
            "AND f.scope = '母公司' AND f.scale IS NOT NULL AND f.column_header IS NOT NULL"):
        kind = re.search(r'(?:母公司|合并及(?:母)?公司)(利润表|资产负债表|现金流量表)', row['title'] or '')
        if kind and role(row['column_header'], row['fiscal_year']) == 'current':
            found.setdefault((row['id'], '合并' + kind.group(1), clean_metric(row['row_label'])),
                             []).append(amount(row['text'], row['scale']))
    return {key: values[0] for key, values in found.items() if len(values) == 1}


def wrong_scope(claim: dict, pair: dict, parents: dict[tuple, Decimal]) -> dict | None:
    """E4: the parent company's own amount stated as the group's."""
    parent = parents.get((pair['document_id'], pair['statement'], pair['metric']))
    if claim['kind'] != 'direct' or parent is None or in_yi(parent) == claim['stated']:
        return None
    return restate(claim, in_yi(parent), claim['unit'], 'E4', 'parent-company amount')


def repeated(claim: dict, rng: random.Random) -> dict | None:
    """E9: the same metric restated later in the passage with a different number."""
    if claim['kind'] != 'direct':
        return None
    other = wrong_digit(claim, rng)['stated']
    metric = claim['text'][:claim['span'][0]].rstrip('为')
    tail = f'综合来看，{metric}达'
    text = claim['text'] + tail + other + claim['unit'] + '。'
    start = len(claim['text']) + len(tail)
    return {**claim, 'text': text, 'first_span': claim['span'], 'span': [start, start + len(other)],
            'stated': other, 'label': 'inconsistent', 'error': 'E9',
            'injected': 'second mention differs from the first'}


MEETING = re.compile(r'新华社北京\d{1,2}月\d{1,2}日电\s*(?:（记者[^）]*）)?\s*国务院总理李强(\d{1,2})月(\d{1,2})日'
                     r'主持召开国务院常务会议，([^。]{6,120})。')


def meeting_claims(db: sqlite3.Connection, rng: random.Random) -> list[dict]:
    """Dated policy statements: the true meeting date, and E7 variants of it."""
    db.row_factory = sqlite3.Row
    records = []
    for row in db.execute(
            "SELECT d.id, d.published, p.text FROM documents d JOIN pages p "
            "ON p.document_id = d.id WHERE d.kind = 'executive-meetings' ORDER BY d.published"):
        match = MEETING.search(row['text'])
        if not match:
            continue
        year, month, day = int(row['published'][:4]), int(match.group(1)), int(match.group(2))
        agenda = match.group(3)

        def claim(stated_year: int, stated_day: int, error: str | None, note: str) -> dict:
            date = f'{stated_year}年{month}月{stated_day}日'
            return {'text': f'{date}，国务院常务会议{agenda}。', 'span': [0, len(date)],
                    'stated': date, 'unit': None, 'kind': 'date',
                    'label': 'inconsistent' if error else 'consistent', 'error': error,
                    'expected': f'{year}年{month}月{day}日', 'facts': [],
                    'document_id': row['id'], 'as_of': None, 'injected': note}
        # An old meeting passed off as recent news: a later year that has already begun.
        later = rng.choice(list(range(year + 1, LATEST_YEAR + 1)) or [year - 1])
        other_day = rng.choice([d for d in range(1, 29) if abs(d - day) >= 2])
        records.extend([claim(year, day, None, ''),
                        claim(later, day, 'E7', f'year {year}->{later}'),
                        claim(year, other_day, 'E7', f'day {day}->{other_day}')])
    return records


def report_digits(db: sqlite3.Connection) -> dict[tuple, set[str]]:
    """Every number written in each issuer's annual report, by fiscal year."""
    found: dict[tuple, set[str]] = {}
    for code, year, digits in db.execute(
            "SELECT d.code, d.fiscal_year, f.digits FROM facts f JOIN documents d "
            "ON d.id = f.document_id WHERE d.kind = 'annual'"):
        found.setdefault((code, year), set()).add(digits)
    return found


def cited(claim: dict, pair: dict, year: int, error: str | None, note: str) -> dict:
    """The claim with a source note naming one of the issuer's annual reports."""
    text = claim['text'].rstrip('。') + f"（数据来源：{pair['company']}{year}年年度报告）。"
    return {**claim, 'text': text, 'cited_report': [pair['code'], year],
            'label': 'inconsistent' if error else claim['label'], 'error': error or claim['error'],
            **({'injected': note} if error else {})}


def wrong_source(claim: dict, pair: dict, digits: dict[tuple, set[str]]) -> list[dict]:
    """A correct source note, and E8: a note naming an earlier report without the figure."""
    if claim['kind'] != 'direct':
        return []
    found = [cited(claim, pair, pair['fiscal_year'], None, '')]
    for year in (pair['fiscal_year'] - 1, pair['fiscal_year'] - 2):
        written = digits.get((pair['code'], year))
        if written is not None and pair['current_digits'] not in written:
            found.append(cited(claim, pair, year, 'E8', f'source report {year} lacks the figure'))
            break
    return found


def restatement_claims(db: sqlite3.Connection) -> list[dict]:
    """E6 from reviewed corrections: an original figure cited after it was corrected."""
    db.row_factory = sqlite3.Row
    records = []
    for correction in db.execute("SELECT * FROM documents WHERE kind = 'correction'").fetchall():
        original = db.execute("SELECT * FROM documents WHERE kind = 'annual' AND code = ? "
                              "AND fiscal_year = ? AND published < ?",
                              (correction['code'], correction['fiscal_year'],
                               correction['published'])).fetchone()
        if original is None:
            continue
        before = {row['row_label']: row for row in db.execute(
            'SELECT * FROM facts WHERE document_id = ?', (original['id'],))}
        # Only reviewed corrections: a full revised report repeats thousands of
        # labels, and matching by label alone would invent changes.
        for after in db.execute("SELECT * FROM facts WHERE document_id = ? AND unit_source = "
                                "'transcribed from scanned page'", (correction['id'],)):
            old = before.get(after['row_label'])
            if old is None or old['digits'] == after['digits']:
                continue
            old_value, new_value = in_yi(amount(old['text'], old['scale'])), in_yi(
                amount(after['text'], after['scale']))
            day_before = str(date.fromisoformat(correction['published']) - timedelta(days=1))
            text = f"{correction['publisher']}{correction['fiscal_year']}年末{after['row_label']}为"
            for as_of, stated, error, facts in (
                    (day_before, old_value, None, [old['id']]),
                    (correction['published'], new_value, None, [after['id']]),
                    (correction['published'], old_value, 'E6', [old['id'], after['id']])):
                records.append({
                    **sentence(text, stated, '亿元'), 'kind': 'direct',
                    'label': 'inconsistent' if error else 'consistent', 'error': error,
                    'expected': old_value if as_of == day_before else new_value, 'facts': facts,
                    'code': correction['code'], 'company': correction['publisher'],
                    'fiscal_year': correction['fiscal_year'], 'as_of': as_of,
                    'metric': after['row_label'], 'statement': '合并资产负债表',
                    'document_id': correction['id'], 'page': after['page'],
                    **({'injected': 'figure superseded by a correction'} if error else {})})
    return records


def build(library: Path, seed: int) -> list[dict]:
    db = sqlite3.connect(f'file:{library}?mode=ro', uri=True)
    rng = random.Random(seed)
    records = []
    parents = parent_amounts(db)
    digits = report_digits(db)
    for pair in pairs(db):
        for claim in (direct(pair), derived(pair)):
            if claim is None:
                continue
            variants = [claim, wrong_digit(claim, rng), wrong_unit(claim, rng),
                        wrong_period(claim, pair), wrong_growth(claim, pair),
                        wrong_scope(claim, pair, parents), repeated(claim, rng),
                        *wrong_source(claim, pair, digits)]
            for variant in variants:
                if variant is not None:
                    records.append({
                        'id': f'seed-{len(records):06d}', 'code': pair['code'],
                        'company': pair['company'], 'fiscal_year': pair['fiscal_year'],
                        'as_of': pair['as_of'], 'metric': pair['metric'],
                        'statement': pair['statement'], 'document_id': pair['document_id'],
                        'page': pair['page'], **variant})
    for claim in restatement_claims(db):
        records.append({'id': f'seed-{len(records):06d}', **claim})
    for claim in meeting_claims(db, rng):
        records.append({'id': f'seed-{len(records):06d}', 'code': None, 'company': None,
                        'fiscal_year': None, 'metric': '国务院常务会议', 'statement': None,
                        'page': 1, **claim})
    db.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'refusing to overwrite {args.output}')
    records = build(args.library, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    counts: dict[str, int] = {}
    for record in records:
        name = record['error'] or record['label']
        counts[name] = counts.get(name, 0) + 1
    print(json.dumps({'records': len(records), 'by_label': counts,
                      'issuers': len({r['code'] for r in records if r['code']})}, ensure_ascii=False))


if __name__ == '__main__':
    main()
