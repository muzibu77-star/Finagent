"""Check model-written reviews end to end and score against verified labels.

Pipeline: a model extracts structured claims from each review, the
deterministic checker judges each claim, and every verified number of the
review receives a verdict (numbers the extraction missed are handed to a
human).  The score that matters most is how many wrong numbers are passed as
consistent, since those would reach a published report.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3

from src.model.api_client import ApiClient
from src.verify.checker import SCALES, Checker, Claim
from src.verify.extract import PROMPT, checkable, line_items, locate, matches, parse
from src.evaluation.draft_pilot import statement_pages

DIRECT = '''下面是一篇关于{company}{year}年年度报告的点评，以及该年报合并财务报表的原文（由PDF提取，排版可能错乱）。
请逐个检查点评中每一个以“亿元”“万元”“百万元”或“%”表示的数字，判断它与原文是否一致（或能否由原文数据正确算出）。
输出 JSON 数组，每个数字一个对象：{{"number": 原文中的数字原样不含单位, "occurrence": 同样写法在点评中第几次出现(从1开始), "verdict": "正确"、"错误"或"无法判断"}}。只输出 JSON 数组。

点评：
{review}

年报原文：
{text}'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True, help='verified labels of the same reviews')
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--budget-usd', type=float, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--direct', action='store_true',
                        help='baseline: the model judges each number itself from the statements')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f'file:{args.library}?mode=ro', uri=True)
    checker = Checker(db)
    client = ApiClient(args.model, args.budget_usd, args.output_dir / 'api_cache.jsonl')
    labels = {}
    for line in args.labels.read_text().splitlines():
        row = json.loads(line)
        labels[(row['review_id'], row['start'])] = row['label']
    results, table = [], Counter()
    for line in args.reviews.read_text().splitlines():
        review = json.loads(line)
        year = review['fiscal_year']
        published = db.execute('SELECT published FROM documents WHERE id = ?',
                               (review['document_id'],)).fetchone()[0]
        verdicts = {}
        if args.direct:
            pages = statement_pages(db, review['document_id'])
            text = '\n'.join(db.execute('SELECT text FROM pages WHERE document_id = ? AND page = ?',
                                        (review['document_id'], page)).fetchone()[0] for page in pages)
            answer = client.chat([{'role': 'user', 'content': DIRECT.format(
                company=review['company'], year=year, review=review['text'],
                text=text or '（未找到合并报表页）')}], max_tokens=8000)
            meaning = {'正确': 'consistent', '错误': 'inconsistent', '无法判断': 'unsupported'}
            for item in parse(answer['content']):
                start = locate(review['text'], str(item['number']), item.get('occurrence'))
                if start is not None and start not in verdicts:
                    verdicts[start] = {'verdict': meaning.get(item.get('verdict'), 'unsupported')}
        items = [] if args.direct else line_items(db, review['code'], year)
        answer = None if args.direct else client.chat([{'role': 'user', 'content': PROMPT.format(
            company=review['company'], year=year, labels='\n'.join(items) or '（无）',
            review=review['text'])}], max_tokens=8000)
        for item in ([] if args.direct else parse(answer['content'])):
            start = locate(review['text'], str(item['number']), item.get('occurrence'))
            if start is None or start in verdicts:
                continue
            kind, unit, label = item.get('type'), item.get('unit'), item.get('label')
            year = item.get('year')
            if kind not in ('amount', 'growth', 'change', 'ratio') or not isinstance(year, int):
                verdicts[start] = {'verdict': 'unsupported', 'reason': 'not a checkable claim'}
                continue
            if not checkable(item.get('phrase')):
                verdicts[start] = {'verdict': 'unsupported', 'reason': 'quarter, segment or own-method ratio'}
                continue
            if item.get('scope') == '母公司':
                verdicts[start] = {'verdict': 'unsupported', 'reason': 'parent-company claim'}
                continue
            if kind == 'ratio':
                top, bottom = item.get('numerator'), item.get('denominator')
                if unit != '%' or not isinstance(top, list) or not isinstance(bottom, list):
                    verdicts[start] = {'verdict': 'unsupported', 'reason': 'ratio parts not named'}
                    continue
                claim = Claim(code=review['code'], metric='', fiscal_year=year,
                              stated=str(item['number']), unit=unit, as_of=published,
                              kind='ratio', numerator=top, denominator=bottom)
            else:
                expected_unit = unit == '%' if kind == 'growth' else unit in SCALES
                if not label or not expected_unit:
                    verdicts[start] = {'verdict': 'unsupported', 'reason': 'no line item or unit'}
                    continue
                if not matches(item.get('phrase'), label):
                    verdicts[start] = {'verdict': 'unsupported', 'label': label,
                                       'reason': 'line item does not fit the wording'}
                    continue
                claim = Claim(code=review['code'], metric=label, fiscal_year=year,
                              stated=str(item['number']), unit=unit, as_of=published,
                              kind={'amount': 'direct'}.get(kind, kind),
                              direction={'up': 1, 'down': -1}.get(item.get('direction')))
            if not re.fullmatch(r'-?[\d,，]*\.?\d+', claim.stated.strip()):
                verdicts[start] = {'verdict': 'unsupported', 'reason': 'number not written plainly'}
                continue
            verdict = checker.check(claim)
            verdicts[start] = {'verdict': verdict.verdict, 'error': verdict.error,
                               'expected': verdict.expected, 'label': label, 'year': year,
                               'type': kind, 'reason': verdict.reason}
        for number in review['numbers']:
            truth = labels[(review['id'], number['start'])]
            mine = verdicts.get(number['start'], {'verdict': 'missed'})
            table[(truth, mine['verdict'])] += 1
            results.append({'review_id': review['id'], 'start': number['start'],
                            'text': number['text'], 'truth': truth, 'closed_book': review['closed_book'],
                            'model': review['model'], **mine})
    (args.output_dir / 'records.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in results))
    by_truth = {truth: {v: n for (t, v), n in table.items() if t == truth}
                for truth in sorted({t for t, _ in table})}
    wrong, correct = by_truth.get('wrong', {}), by_truth.get('correct', {})
    summary = {
        'numbers': len(results), 'by_truth': by_truth,
        'wrong_numbers': sum(wrong.values()),
        'wrong_flagged': wrong.get('inconsistent', 0),
        'wrong_passed_as_consistent': wrong.get('consistent', 0),
        'correct_numbers': sum(correct.values()),
        'correct_flagged': correct.get('inconsistent', 0),
        'correct_confirmed': correct.get('consistent', 0),
        'cost_usd': round(client.spent, 4)}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
