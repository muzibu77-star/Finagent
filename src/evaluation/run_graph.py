"""Run the LangGraph review flow over a whole review set and score it.

Every draft of a frozen review set goes through ``review_draft``; its workpaper
is written to ``<output-dir>/workpapers/<review_id>.json`` and its verdicts are
scored against the verified labels with the same counts as ``run_reviews``.
The API cache of the extraction run comes first in the output directory (as
``api_cache.jsonl``), so extraction is a cache hit and only the revision calls
cost money.  Unless ``--no-investigate`` is given, the tool agent re-opens the
numbers the checker handed to a person and the summary counts the verdicts that
came from the agent.

The extraction model and the model of the investigate and revision nodes (the
agent model, which defaults to ``--model``) are metered apart: the agent answers
go to ``api_cache_<agent-model>.jsonl`` beside the extraction cache, both files
are named in the summary, and ``cost_usd_by_model`` splits the spend.

A local agent model is slow, so ``--shard I N`` runs only the I-th of N
contiguous chunks of the review set and prints the chunk it scored.  Each shard
gets its own ``--output-dir``, seeded with a copy of the shared extraction
cache, so the shards never write the same file.  Concatenating the shard caches
(one row per request, identical rows dropped) into ``api_cache.jsonl`` of a new
output directory and rerunning the whole set there scores it as one set at no
request cost - the sharding used for the independent check.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

from src.agent.graph import plain, review_draft


def cache_rows(path: Path) -> list[dict]:
    """The rows of a cache file; one that does not exist yet is empty."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def cache_files(output_dir: Path, model: str, agent_model: str | None) -> list[Path]:
    """The caches of a run: the extraction one, plus the agent one if it differs."""
    extract = output_dir / 'api_cache.jsonl'
    if agent_model in (None, model):
        return [extract]
    return [extract, output_dir / f'api_cache_{agent_model}.jsonl']


def shard_reviews(reviews: list[dict], index: int, count: int) -> list[dict]:
    """The ``index``-th (from 1) of ``count`` contiguous chunks of the review set."""
    if not 1 <= index <= count:
        raise SystemExit(f'shard {index} of {count}: the shard number must be in 1..{count}')
    return reviews[(index - 1) * len(reviews) // count:index * len(reviews) // count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True,
                        help='verified labels of the same reviews')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--url', default='https://api.deepseek.com/chat/completions')
    parser.add_argument('--agent-model',
                        help='model of the investigate and revision nodes; defaults to --model')
    parser.add_argument('--agent-url', help='endpoint of that model; defaults to --url')
    parser.add_argument('--budget-usd', type=float, required=True)
    parser.add_argument('--no-investigate', action='store_true',
                        help='leave the numbers handed to a person with the checker')
    parser.add_argument('--shard', nargs=2, type=int, metavar=('I', 'N'),
                        help='score only chunk I of N of the review set, for one parallel shard')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'workpapers').mkdir(exist_ok=True)
    caches = cache_files(args.output_dir, args.model, args.agent_model)
    paid = [len(cache_rows(path)) for path in caches]
    db = sqlite3.connect(f'file:{args.library}?mode=ro', uri=True)
    labels = {}
    for line in args.labels.read_text().splitlines():
        row = json.loads(line)
        labels[(row['review_id'], row['start'])] = row['label']
    reviews = [json.loads(line) for line in args.reviews.read_text().splitlines()]
    if args.shard:
        reviews = shard_reviews(reviews, *args.shard)
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
                             budget_usd=args.budget_usd, cache=caches[0],
                             investigate=not args.no_investigate,
                             agent_model=args.agent_model, agent_url=args.agent_url,
                             agent_cache=caches[-1])
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
    cached = [cache_rows(path) for path in caches]
    spent = [row['cost_usd'] for rows in cached for row in rows]
    fresh = [row['cost_usd'] for rows, first in zip(cached, paid) for row in rows[first:]]
    by_model = Counter()
    for rows in cached:
        for row in rows:
            by_model[row['model']] += row['cost_usd']
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
        'cache': [path.name for path in caches],
        'cost_usd': round(sum(spent), 4), 'cost_usd_new': round(sum(fresh), 4),
        'cost_usd_by_model': {model: round(cost, 4)
                              for model, cost in sorted(by_model.items())}}
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
