"""Run the LangGraph review flow over a whole review set and score it.

Every draft of a frozen review set goes through ``review_draft``; its workpaper
is written to ``<output-dir>/workpapers/<review_id>.json`` and its verdicts are
scored against the verified labels with the same counts as ``run_reviews``.
The API cache of the extraction run comes first in the output directory (as
``api_cache.jsonl``), so extraction is a cache hit and only the revision calls
cost money.  Unless ``--no-investigate`` is given, the tool agent re-opens the
numbers the checker handed to a person and the summary counts the verdicts that
came from the agent.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

from src.agent.graph import plain, review_draft


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
    parser.add_argument('--no-investigate', action='store_true',
                        help='leave the numbers handed to a person with the checker')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'workpapers').mkdir(exist_ok=True)
    cache = args.output_dir / 'api_cache.jsonl'
    paid = len(cache.read_text().splitlines()) if cache.exists() else 0
    db = sqlite3.connect(f'file:{args.library}?mode=ro', uri=True)
    labels = {}
    for line in args.labels.read_text().splitlines():
        row = json.loads(line)
        labels[(row['review_id'], row['start'])] = row['label']
    reviews = [json.loads(line) for line in args.reviews.read_text().splitlines()]
    table = Counter()
    revisions = accepted = expected_shown = 0
    passed_nonrounding = 0
    agent_flagged, agent_hint = Counter(), Counter()
    errors = []
    for review in reviews:
        published = db.execute('SELECT published FROM documents WHERE id = ?',
                               (review['document_id'],)).fetchone()[0]
        paper = review_draft({'text': review['text'], 'code': review['code'],
                              'company': review['company'], 'fiscal_year': review['fiscal_year'],
                              'as_of': published},
                             library=args.library, model=args.model, url=args.url,
                             budget_usd=args.budget_usd, cache=cache,
                             investigate=not args.no_investigate)
        (args.output_dir / 'workpapers' / f"{review['id']}.json").write_text(
            json.dumps(paper, ensure_ascii=False, indent=1) + '\n')
        rows = {row['start']: row for row in paper['numbers']}
        verdicts = {start: row['verdict'] for start, row in rows.items()}
        errors += paper['errors']
        for number in review['numbers']:
            truth = labels[(review['id'], number['start'])]
            verdict = verdicts.get(number['start'], 'missed')
            table[(truth, verdict)] += 1
            if (truth == 'wrong' and verdict == 'consistent'
                    and rows.get(number['start'], {}).get('reason') != 'rounding difference'):
                passed_nonrounding += 1
        for start, row in rows.items():
            truth = labels.get((review['id'], start))
            if row['source'] == 'agent':
                agent_flagged[truth] += 1
            elif row.get('agent') and row['agent']['verdict'] == 'consistent':
                agent_hint[truth] += 1
        for row in paper['numbers']:
            if not row['suggestion']:
                continue
            revisions += 1
            if row['suggestion'].get('accepted'):
                accepted += 1
                expected_shown += plain(str(row['expected'])) in plain(row['suggestion']['sentence'])
        print(f"{review['id']} {json.dumps(paper['summary'], ensure_ascii=False)}", flush=True)
    spent = [json.loads(line)['cost_usd'] for line in cache.read_text().splitlines()]
    by_truth = {truth: {v: n for (t, v), n in table.items() if t == truth}
                for truth in sorted({t for t, _ in table})}
    wrong, correct = by_truth.get('wrong', {}), by_truth.get('correct', {})
    summary = {
        'reviews': len(reviews), 'numbers': sum(table.values()), 'by_truth': by_truth,
        'wrong_numbers': sum(wrong.values()),
        'wrong_flagged': wrong.get('inconsistent', 0),
        'wrong_passed_as_consistent': wrong.get('consistent', 0),
        'correct_numbers': sum(correct.values()),
        'correct_flagged': correct.get('inconsistent', 0),
        'correct_confirmed': correct.get('consistent', 0),
        'wrong_passed_nonrounding': passed_nonrounding,
        'agent_flagged': sum(agent_flagged.values()),
        'agent_flagged_by_truth': {truth: agent_flagged.get(truth, 0)
                                   for truth in ('wrong', 'correct', 'unverifiable')},
        'agent_hint_consistent': sum(agent_hint.values()),
        'agent_hint_by_truth': {truth: agent_hint.get(truth, 0)
                                for truth in ('wrong', 'correct', 'unverifiable')},
        'agent_errors': errors,
        'revisions': {'suggested': revisions, 'accepted': accepted,
                      'discarded': revisions - accepted,
                      'expected_in_rewrite': expected_shown},
        'cost_usd': round(sum(spent), 4), 'cost_usd_new': round(sum(spent[paid:]), 4)}
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
