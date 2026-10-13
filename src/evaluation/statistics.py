"""Report-cluster bootstrap intervals with a fixed, reproducible random seed."""

from collections import defaultdict
import random
import statistics


def cluster_interval(rows: list[dict], metric: str, group: str = 'source_report_id',
                     repetitions: int = 2000, seed: int = 103) -> dict:
    if not rows or repetitions < 1:
        raise ValueError('nonempty observations and positive repetitions required')
    clusters = defaultdict(list)
    for row in rows:
        clusters[row[group]].append(float(row[metric]))
    keys = sorted(clusters)
    totals = [(sum(clusters[key]), len(clusters[key])) for key in keys]
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sample = rng.choices(totals, k=len(totals))
        draws.append(sum(total for total, _ in sample) / sum(n for _, n in sample))
    draws.sort()
    return {'mean': statistics.mean(float(row[metric]) for row in rows),
            'ci95': [draws[int(.025 * (repetitions - 1))], draws[int(.975 * (repetitions - 1))]],
            'questions': len(rows), 'clusters': len(keys), 'resamples': repetitions,
            'seed': seed, 'method': 'percentile bootstrap, resample source reports'}


def paired_interval(baseline: list[dict], candidate: list[dict], metric: str) -> dict:
    """Resample matched task differences, retaining report-level dependence."""
    def indexed(rows):
        result = {(r['task_id'], r['condition']): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate paired observations')
        return result
    left, right = indexed(baseline), indexed(candidate)
    if left.keys() != right.keys():
        raise ValueError('paired evaluation membership differs')
    differences = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a['source_report_id'] != b['source_report_id']:
            raise ValueError('paired source membership differs')
        differences.append({'source_report_id': a['source_report_id'],
                            'difference': float(b[metric])-float(a[metric])})
    return cluster_interval(differences, 'difference')
