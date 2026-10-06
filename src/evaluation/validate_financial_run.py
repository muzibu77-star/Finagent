"""Audit complete paired runs and resolve every calculation citation to raw text."""

import argparse
from collections import Counter
import json
from pathlib import Path


def validate(run: Path, data: Path, tasks_path: Path) -> dict:
    rows = [json.loads(x) for x in (run / 'records.jsonl').read_text().splitlines()]
    tasks = json.loads(tasks_path.read_text())
    evidence = {r['evidence_id']: r for r in map(json.loads,
        (data / 'evidence.jsonl').read_text().splitlines())}
    gold = {r['task_id']: r['gold'] for r in map(json.loads,
        (data / 'gold.jsonl').read_text().splitlines())}
    conditions = ('direct', 'calculator', 'retrieval_calculator')
    expected_pairs = {(t['task_id'], c) for t in tasks for c in conditions}
    actual_pairs = [(r['task_id'], r['condition']) for r in rows]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError('missing or duplicate evaluation rows')
    citations = 0
    failures = Counter()
    for row in rows:
        expected = gold[row['task_id']]['exe_ans']
        if row['correct'] != ('error' not in row and row.get('prediction') == expected):
            raise ValueError('score disagrees with source gold')
        if not row['correct']:
            error = row.get('error', '')
            category = ('input_budget' if error == 'input_over_budget' else
                        'tool_or_output_contract' if error else
                        'retrieval' if not row['correct_evidence'] else 'numeric_or_semantic')
            failures[f"{row['condition']}:{category}"] += 1
        for fact in row.get('calculation', {}).get('facts', {}).values():
            if fact['evidence_id'] != row['selected_evidence']:
                raise ValueError('citation points outside selected document')
            doc, loc = evidence[fact['evidence_id']], fact['location']
            if 'row' in loc:
                text = str(doc['table'][loc['row']][loc['column']])
            elif loc['section'] == 'paragraphs':
                text = doc['paragraphs'][loc['paragraph']]['text']
            else:
                text = doc[loc['section']][loc['paragraph']]
            if text[loc['start']:loc['end']] != fact['raw']:
                raise ValueError('citation scalar differs from original span')
            citations += 1
    return {'passed': True, 'rows': len(rows), 'resolved_scalar_citations': citations,
            'failures': dict(failures),
            'meaning': 'Run completeness, score integrity and scalar provenance; not semantic accuracy.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/staged/m0_frozen_v1'))
    parser.add_argument('--tasks', type=Path, default=Path('artifacts/m1_baseline_v1/tasks.json'))
    args = parser.parse_args()
    result = validate(args.run, args.data, args.tasks)
    (args.run / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
