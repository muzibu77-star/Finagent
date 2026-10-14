"""Paired review-level bootstrap intervals for differences between two runs.

Both runs score the same verified numbers (``records.jsonl`` of
``run_reviews``).  Numbers of one review are correlated, so whole reviews are
resampled with replacement and the same resample is applied to both runs.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

RATES = {
    'false_alarm_rate': ('correct', {'inconsistent'}),
    'confirm_rate': ('correct', {'consistent'}),
    'flag_rate': ('wrong', {'inconsistent'}),
    'missed_rate': (None, {'missed'}),
}


def by_review(path: Path) -> dict[str, dict[int, dict]]:
    reviews: dict[str, dict[int, dict]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        reviews[row['review_id']][row['start']] = row
    return reviews


def counts(review: dict[int, dict]) -> dict[str, tuple[int, int]]:
    """``rate -> (hits, base)`` for one review."""
    result = {}
    for name, (truth, verdicts) in RATES.items():
        rows = [r for r in review.values() if truth is None or r['truth'] == truth]
        result[name] = (sum(r['verdict'] in verdicts for r in rows), len(rows))
    return result


def rate(parts: list[dict], name: str) -> float:
    hits = sum(p[name][0] for p in parts)
    base = sum(p[name][1] for p in parts)
    return hits / base if base else 0.0


def compare(a: Path, b: Path, samples: int = 10_000, seed: int = 0) -> dict:
    """Point estimates and 95% intervals of ``b - a`` for every rate."""
    left, right = by_review(a), by_review(b)
    if left.keys() != right.keys() or any(left[k].keys() != right[k].keys() for k in left):
        raise ValueError('the two runs do not score the same numbers')
    ids = sorted(left)
    first = {k: counts(left[k]) for k in ids}
    second = {k: counts(right[k]) for k in ids}
    rng = random.Random(seed)
    draws = {name: [] for name in RATES}
    for _ in range(samples):
        pick = [rng.choice(ids) for _ in ids]
        for name in RATES:
            draws[name].append(rate([second[k] for k in pick], name)
                               - rate([first[k] for k in pick], name))
    result = {}
    for name, values in draws.items():
        values.sort()
        result[name] = {
            'a': round(rate(list(first.values()), name), 4),
            'b': round(rate(list(second.values()), name), 4),
            'b_minus_a': round(rate(list(second.values()), name)
                               - rate(list(first.values()), name), 4),
            'ci95': [round(values[int(0.025 * samples)], 4),
                     round(values[int(0.975 * samples) - 1], 4)]}
    return {'reviews': len(ids), 'samples': samples, 'seed': seed, 'rates': result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, required=True, help='records.jsonl of run A')
    parser.add_argument('--b', type=Path, required=True, help='records.jsonl of run B')
    parser.add_argument('--samples', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(compare(args.a, args.b, args.samples, args.seed), indent=1))


if __name__ == '__main__':
    main()
