"""Paired issuer-cluster intervals for audited frozen Agent experiments."""

import argparse
import hashlib
import json
from pathlib import Path

from src.evaluation.statistics import cluster_interval


def audited_rows(root: Path) -> tuple[dict, dict, dict]:
    validation = json.loads((root / 'validation.json').read_text())
    for name in ('records', 'turns'):
        digest = hashlib.sha256((root / f'{name}.jsonl').read_bytes()).hexdigest()
        if not validation['passed'] or digest != validation[f'{name}_sha256']:
            raise ValueError('Agent comparison requires a current independent audit')
    rows = [json.loads(s) for s in (root / 'records.jsonl').read_text().splitlines()]
    indexed = {r['task_id']: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('duplicate Agent comparison tasks')
    return indexed, json.loads((root / 'settings.json').read_text()), validation


def paired_quality(left: dict, right: dict, tasks: dict, ids: list[str]) -> dict:
    """Resample issuers so related reports and whole conversations stay together."""
    if not ids or any(key not in left or key not in right or key not in tasks for key in ids):
        raise ValueError('nonempty matched Agent tasks required')
    rows = [{'company': json.dumps(tasks[key]['company'], ensure_ascii=False),
             'baseline': float(left[key]['all_turns_passed']),
             'candidate': float(right[key]['all_turns_passed']),
             'difference': float(right[key]['all_turns_passed']) - float(left[key]['all_turns_passed'])}
            for key in ids]
    result = {}
    for name in ('baseline', 'candidate', 'difference'):
        interval = cluster_interval(rows, name, group='company')
        interval['method'] = 'percentile bootstrap, resample issuers; whole tasks/conversations'
        result[name] = interval
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    left, base_settings, base_audit = audited_rows(args.baseline)
    right, settings, audit = audited_rows(args.candidate)
    for key in ('lock', 'tasks_root', 'max_calls', 'task_seconds', 'max_input_tokens',
                'max_output_tokens', 'backend'):
        if settings.get(key) != base_settings.get(key):
            raise ValueError(f'Agent comparison protocol differs: {key}')
    root = Path(settings['tasks_root'])
    for name, digest in settings['lock']['sha256'].items():
        if hashlib.sha256((root / f'{name}.json').read_bytes()).hexdigest() != digest:
            raise ValueError('frozen Agent inputs changed')
    tasks = {t['task_id']: t for t in json.loads((root / 'tasks.json').read_text())}
    if left.keys() != right.keys() or left.keys() != tasks.keys():
        raise ValueError('Agent comparison membership differs')
    groups = {'all': list(tasks), 'cross_report_and_multi_turn': [
        key for key, task in tasks.items() if task['category'] in ('cross_evidence', 'multi_turn')]}
    groups.update({category: [key for key, task in tasks.items() if task['category'] == category]
                   for category in sorted({t['category'] for t in tasks.values()})})
    quality = {name: paired_quality(left, right, tasks, ids) for name, ids in groups.items()}
    fixed = {}
    for name, rows in [('baseline', left), ('candidate', right)]:
        control = {key: {'all_turns_passed': bool(rows[key]['fixed']['correct']
                    and rows[key]['fixed']['source_supported'])} for key in groups['calculation']}
        fixed[name] = paired_quality(control, rows, tasks, groups['calculation'])
    result = {'split': settings['lock']['split'], 'quality': quality,
              'fixed_single_question_comparison': fixed,
              'runtime': {'baseline': base_audit, 'candidate': audit},
              'limits': 'Custom research tasks, not an official benchmark. Fixed flow supports only '
                        'single-question comparison here; its lack of cross-report/multi-turn support '
                        'is not scored as an algorithmic improvement. Missing-company controls share '
                        'one cluster. This command does not select a model.'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
