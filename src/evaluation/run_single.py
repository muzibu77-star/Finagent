"""Run the single-agent control group over a review set and score it like the flow.

Every draft goes through ``review_draft_single``; the verdicts are scored against
the verified labels with the same counts as ``run_reviews`` (the single agent is
a drop-in for the ``--direct`` model baseline, which also judges numbers itself),
and each run's tool calls are written to ``<output-dir>/traces/<review_id>.json``.
The summary adds what only this arm has: tool calls and token use per review.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sqlite3

from src.agent.single import review_draft_single


def written(text: str) -> tuple[Decimal, Decimal] | None:
    """The value of a written number and one unit of its last written digit."""
    plain = text.replace(',', '').replace('，', '').rstrip('%')
    try:
        value = Decimal(plain)
    except InvalidOperation:
        return None
    decimals = len(plain.split('.')[1]) if '.' in plain else 0
    return value, Decimal(1).scaleb(-decimals)


def near(stated: str, expected: str) -> bool:
    """Whether a wrongly passed number is within one unit of its last written digit."""
    pair, wanted = written(stated), written(expected)
    return pair is not None and wanted is not None and abs(pair[0] - wanted[0]) <= pair[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True,
                        help='verified labels of the same reviews')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--url', default='https://api.deepseek.com/chat/completions')
    parser.add_argument('--budget-usd', type=float, required=True)
    parser.add_argument('--limit', type=int, help='only the first N reviews (a pilot run)')
    parser.add_argument('--workers', type=int, default=1,
                        help='concurrent reviews; each one appends its own answers to the cache')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'traces').mkdir(exist_ok=True)
    cache = args.output_dir / 'api_cache.jsonl'
    paid = len(cache.read_text().splitlines()) if cache.exists() else 0
    db = sqlite3.connect(f'file:{args.library}?mode=ro', uri=True)
    labels = {}
    for line in args.labels.read_text().splitlines():
        row = json.loads(line)
        labels[(row['review_id'], row['start'])] = row
    reviews = [json.loads(line) for line in args.reviews.read_text().splitlines()]
    if args.limit:
        reviews = reviews[:args.limit]
    table, results = Counter(), []
    tool_calls = prompt_tokens = completion_tokens = 0
    near_misses = 0

    jobs = [(review, db.execute('SELECT published FROM documents WHERE id = ?',
                                (review['document_id'],)).fetchone()[0])
            for review in reviews]

    def one(job: tuple[dict, str]) -> tuple[dict, dict]:
        review, published = job
        return review, review_draft_single(
            {'text': review['text'], 'code': review['code'], 'company': review['company'],
             'fiscal_year': review['fiscal_year'], 'as_of': published},
            library=args.library, model=args.model, url=args.url,
            budget_usd=args.budget_usd, cache=cache)

    with ThreadPoolExecutor(args.workers) as pool:
        runs = pool.map(one, jobs)
        for review, run in runs:
            (args.output_dir / 'traces' / f"{review['id']}.json").write_text(
                json.dumps(run['messages'], ensure_ascii=False, indent=1) + '\n')
            tool_calls += run['tool_calls']
            prompt_tokens += run['usage'].get('prompt_tokens', 0)
            completion_tokens += run['usage'].get('completion_tokens', 0)
            verdicts = {start: row['verdict'] for start, row in run['verdicts'].items()}
            for number in review['numbers']:
                truth = labels[(review['id'], number['start'])]['label']
                mine = verdicts.get(number['start'], 'missed')
                table[(truth, mine)] += 1
                if truth == 'wrong' and mine == 'consistent':
                    if near(number['text'], labels[(review['id'], number['start'])]['expected']):
                        near_misses += 1
                results.append({'review_id': review['id'], 'start': number['start'],
                                'text': number['text'], 'truth': truth,
                                'closed_book': review['closed_book'], 'model': review['model'],
                                'verdict': mine})
            print(f"{review['id']} {len(verdicts)}/{len(review['numbers'])} "
                  f"{run['tool_calls']} calls", flush=True)
    (args.output_dir / 'records.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in results))
    spent = [json.loads(line)['cost_usd'] for line in cache.read_text().splitlines()]
    by_truth = {truth: {v: n for (t, v), n in table.items() if t == truth}
                for truth in sorted({t for t, _ in table})}
    wrong, correct = by_truth.get('wrong', {}), by_truth.get('correct', {})
    summary = {
        'reviews': len(reviews), 'numbers': sum(table.values()), 'by_truth': by_truth,
        'wrong_numbers': sum(wrong.values()),
        'wrong_flagged': wrong.get('inconsistent', 0),
        'wrong_passed_as_consistent': wrong.get('consistent', 0),
        'wrong_passed_within_one_unit': near_misses,
        'correct_numbers': sum(correct.values()),
        'correct_flagged': correct.get('inconsistent', 0),
        'correct_confirmed': correct.get('consistent', 0),
        'tool_calls_per_review': round(tool_calls / len(reviews), 2),
        'prompt_tokens_per_review': round(prompt_tokens / len(reviews), 1),
        'completion_tokens_per_review': round(completion_tokens / len(reviews), 1),
        'cost_usd': round(sum(spent), 4), 'cost_usd_new': round(sum(spent[paid:]), 4)}
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
