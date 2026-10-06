"""Independently validate staged counts, references, isolation and leakage."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from src.data.leakage import content_text, near_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    rows = {name: [json.loads(line) for line in (root / f'{name}.jsonl').read_text().splitlines()]
            for name in ('manifest', 'questions', 'gold', 'evidence')}
    evidence = {r['evidence_id']: r for r in rows['evidence']}
    assert len(evidence) == len(rows['evidence']) == 11038
    task_ids = {r['task_id'] for r in rows['manifest']}
    assert len(task_ids) == len(rows['manifest']) == 24833
    for name in ('questions', 'gold'):
        assert len(rows[name]) == len(task_ids)
        assert {r['task_id'] for r in rows[name]} == task_ids
    assert all(r['evidence_id'] in evidence for r in rows['questions'])
    assert all(set(r) <= {'evidence_id', 'table', 'paragraphs', 'pre_text', 'post_text'}
               for r in evidence.values())
    groups = defaultdict(set)
    selected = {}
    for row in rows['manifest']:
        if row['official_eligible']:
            assert row['exclusion_reason'] is None
            groups[row['source_report_id']].add(row['original_split'])
            selected[row['evidence_id']] = row
    assert all(len(splits) == 1 for splits in groups.values())
    selected = list(selected.values())
    for a, b, _ in near_pairs([content_text(evidence[r['evidence_id']]) for r in selected]):
        assert selected[a]['original_split'] == selected[b]['original_split']
    receipt = {'passed': True, 'questions': len(task_ids), 'evidence': len(evidence),
        'eligible': sum(r['official_eligible'] for r in rows['manifest']),
        'sha256': {name: hashlib.sha256((root / f'{name}.jsonl').read_bytes()).hexdigest()
                   for name in rows}}
    (root / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
