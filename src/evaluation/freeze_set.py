"""Freeze the evaluation set: injected seed claims and model-written reviews.

Records are assigned by issuer to the frozen training, development and
acceptance splits.  Training keeps every seed claim; the two held-out splits
keep a fixed-size sample per error type so that no type dominates.  Reviews are
kept whole, with the code-assigned support flag of each number, which is a weak
label and not a verified one.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

HELD_OUT = ('development', 'acceptance')


def split_of(record: dict, by_code: dict[str, str]) -> str:
    """The issuer's split; material outside the 30 issuers is spread by a stable hash."""
    if record.get('code') in by_code:
        return by_code[record['code']]
    bucket = int(hashlib.sha256(record['document_id'].encode()).hexdigest(), 16) % 30
    return 'acceptance' if bucket < 8 else 'development' if bucket < 14 else 'training'


def sample(records: list[dict], per_error: int, consistent: int, seed: int) -> list[dict]:
    """At most ``per_error`` claims of each error type and ``consistent`` correct ones."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record['error'] or 'consistent', []).append(record)
    kept = []
    for name in sorted(groups):
        rows = sorted(groups[name], key=lambda row: row['id'])
        random.Random(f'{seed}-{name}').shuffle(rows)
        kept.extend(rows[:consistent if name == 'consistent' else per_error])
    return sorted(kept, key=lambda row: row['id'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=Path, required=True)
    parser.add_argument('--drafts', type=Path, required=True,
                        help='directory with one draft_pilot output directory per condition')
    parser.add_argument('--splits', type=Path, default=Path('configs/issuer_splits.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--per-error', type=int, default=200)
    parser.add_argument('--consistent', type=int, default=600)
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    splits = json.loads(args.splits.read_text())
    by_code = {row['code']: name for name in ('training', *HELD_OUT) for row in splits[name]}
    claims: dict[str, list[dict]] = {name: [] for name in ('training', *HELD_OUT)}
    for line in args.seeds.read_text().splitlines():
        record = json.loads(line)
        claims[split_of(record, by_code)].append(record)
    for name in HELD_OUT:
        claims[name] = sample(claims[name], args.per_error, args.consistent, args.seed)
    reviews: dict[str, list[dict]] = {name: [] for name in claims}
    for directory in sorted(path for path in args.drafts.iterdir() if path.is_dir()):
        summary = json.loads((directory / 'summary.json').read_text())
        for record in json.loads((directory / 'records.json').read_text()):
            if 'review' not in record:
                continue
            code = record['document_id'].split('-')[0]
            reviews[split_of({'code': code, 'document_id': record['document_id']}, by_code)].append({
                'id': f"review-{directory.name}-{record['document_id']}", 'code': code,
                'company': record['company'], 'fiscal_year': record['year'],
                'document_id': record['document_id'], 'model': summary['model'],
                'closed_book': summary['closed_book'], 'text': record['review'],
                'numbers': record['numbers']})
    lock = {'seed': args.seed, 'per_error': args.per_error, 'consistent': args.consistent,
            'seeds_sha256': hashlib.sha256(args.seeds.read_bytes()).hexdigest(),
            'splits_sha256': hashlib.sha256(args.splits.read_bytes()).hexdigest(), 'files': {}}
    for kind, groups in (('claims', claims), ('reviews', reviews)):
        for name, rows in groups.items():
            path = args.output_dir / f'{name}_{kind}.jsonl'
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
            counts: dict[str, int] = {}
            for row in rows:
                label = (row['error'] or 'consistent') if kind == 'claims' else (
                    f"{row['model']}/{'closed' if row['closed_book'] else 'open'}")
                counts[label] = counts.get(label, 0) + 1
            lock['files'][path.name] = {
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'records': len(rows),
                'issuers': len({row['code'] for row in rows if row['code']}), 'counts': counts}
    (args.output_dir / 'lock.json').write_text(json.dumps(lock, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({name: {'records': info['records'], 'issuers': info['issuers']}
                      for name, info in lock['files'].items()}))


if __name__ == '__main__':
    main()
