"""Check model-written reviews end to end and score against verified labels.

Pipeline: a model extracts structured claims from each review, the
deterministic checker judges each claim, and every verified number of the
review receives a verdict (numbers the extraction missed are handed to a
human).  The score that matters most is how many wrong numbers are passed as
consistent, since those would reach a published report.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3

from src.model.api_client import ApiClient
from src.verify.checker import Checker
from src.verify.extract import PROMPT, line_items, locate, parse
from src.verify.pipeline import judge
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
    parser.add_argument('--url', default='https://api.deepseek.com/chat/completions',
                        help='chat completions endpoint; a local vLLM server for local models')
    parser.add_argument('--budget-usd', type=float, required=True)
    parser.add_argument('--oracle', type=Path,
                        help='extraction samples whose answers replace the model (checks the targets)')
    parser.add_argument('--workers', type=int, default=1,
                        help='concurrent extraction requests, for a local server')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--direct', action='store_true',
                        help='baseline: the model judges each number itself from the statements')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f'file:{args.library}?mode=ro', uri=True)
    checker = Checker(db)
    client = ApiClient(args.model, args.budget_usd, args.output_dir / 'api_cache.jsonl', url=args.url)
    labels = {}
    for line in args.labels.read_text().splitlines():
        row = json.loads(line)
        labels[(row['review_id'], row['start'])] = row['label']
    results, table = [], Counter()
    reviews = [json.loads(line) for line in args.reviews.read_text().splitlines()]
    oracle = {}
    if args.oracle:
        for line in args.oracle.read_text().splitlines():
            sample = json.loads(line)
            oracle[sample['review_id']] = sample['messages'][-1]['content']
    if args.workers > 1 and not args.direct:
        # Answers land in the client's cache; the loop below then reads them back.
        def extract(review: dict) -> None:
            items = line_items(db_for_threads(), review['code'], review['fiscal_year'])
            client.chat([{'role': 'user', 'content': PROMPT.format(
                company=review['company'], year=review['fiscal_year'],
                labels='\n'.join(items) or '（无）', review=review['text'])}], max_tokens=8000)

        def db_for_threads() -> sqlite3.Connection:
            return sqlite3.connect(f'file:{args.library}?mode=ro', uri=True)

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(extract, reviews))
    for review in reviews:
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
        answer = {'content': oracle[review['id']]} if oracle else None if args.direct else client.chat([{'role': 'user', 'content': PROMPT.format(
            company=review['company'], year=year, labels='\n'.join(items) or '（无）',
            review=review['text'])}], max_tokens=8000)
        if not args.direct:
            verdicts = judge(checker, review['text'], parse(answer['content']),
                             review['code'], published)
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
