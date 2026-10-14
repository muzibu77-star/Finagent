"""Assign issuers to training, development and acceptance splits.

Whole issuers are held out, so no report of an acceptance issuer is seen while
developing or training.  Issuers are interleaved by industry before cutting so
that each held-out split spans as many industries as it has room for.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

SIZES = (('acceptance', 8), ('development', 6), ('training', 16))


def assign(issuers: list[dict], seed: int) -> dict[str, list[dict]]:
    if len(issuers) != sum(size for _, size in SIZES):
        raise ValueError('split sizes do not match the number of issuers')
    rng = random.Random(seed)
    industries: dict[str, list[dict]] = {}
    for issuer in sorted(issuers, key=lambda row: row['code']):
        industries.setdefault(issuer['industry'], []).append(issuer)
    order = sorted(industries)
    rng.shuffle(order)
    for name in order:
        rng.shuffle(industries[name])
    interleaved = []
    while any(industries.values()):
        interleaved.extend(industries[name].pop() for name in order if industries[name])
    splits, start = {}, 0
    for name, size in SIZES:
        splits[name] = sorted(interleaved[start:start + size], key=lambda row: row['code'])
        start += size
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--issuers', type=Path, default=Path('configs/cninfo_issuers.json'))
    parser.add_argument('--output', type=Path, default=Path('configs/issuer_splits.json'))
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'refusing to overwrite {args.output}: the split is frozen')
    splits = assign(json.loads(args.issuers.read_text())['issuers'], args.seed)
    args.output.write_text(json.dumps({
        'seed': args.seed, 'issuers_sha256': hashlib.sha256(args.issuers.read_bytes()).hexdigest(),
        **{name: [{k: row[k] for k in ('code', 'name', 'industry')} for row in rows]
           for name, rows in splits.items()}}, ensure_ascii=False, indent=1) + '\n')
    for name, rows in splits.items():
        print(name, len(rows), [f"{row['name']}({row['industry']})" for row in rows])


if __name__ == '__main__':
    main()
