"""Recompute full-test scores and matched confidence intervals from saved outputs."""

import argparse
import hashlib
import json
from pathlib import Path

from src.evaluation.full_benchmark import tatqa_prediction, tatqa_score
from src.evaluation.statistics import cluster_interval, paired_interval
from src.model.financial import decode_prediction
from src.tools.benchmark_tools import extract_numeric_facts


def validate_rows(records: list[dict], tasks: list[dict], gold: dict,
                  evidence: dict) -> int:
    expected = {(task['task_id'], condition) for task in tasks for condition in
                (('direct', 'calculator') if task['dataset'] == 'finqa' else ('direct',))}
    actual = [(r['task_id'], r['condition']) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('missing, duplicate or unexpected full-test observations')
    by_id = {task['task_id']: task for task in tasks}
    citations = 0
    for row in records:
        task, label = by_id[row['task_id']], gold[row['task_id']]
        if (row['dataset'] != task['dataset'] or row['source_report_id'] != task['source_report_id']
                or row['labels_available'] != label['labels_available']):
            raise ValueError('observation metadata differs from frozen membership')
        if not label['labels_available']:
            if any(k in row for k in ('em', 'f1', 'scale')):
                raise ValueError('unlabeled task was incorrectly scored')
            continue
        score = {'em': 0.0, 'f1': 0.0, 'scale': 0.0}
        if row.get('stopped_on_eos') and 'text' in row:
            if task['dataset'] == 'finqa':
                facts = extract_numeric_facts(evidence[task['evidence_id']])
                decoded = decode_prediction({'text': row['text'], 'stopped_on_eos': True},
                                            facts, row['condition'])
                if 'error' not in decoded:
                    score['em'] = float(decoded['prediction'] == label['gold']['exe_ans'])
                    score['f1'] = score['em']
                    if decoded.get('calculation') != row.get('calculation'):
                        raise ValueError('saved calculation differs from executed raw output')
                    citations += len(decoded.get('calculation', {}).get('facts', {}))
            else:
                try:
                    score = tatqa_score(tatqa_prediction(row['text']), label['gold'])
                except (ValueError, TypeError, KeyError):
                    pass  # Invalid output remains a scored failure.
        if any(row[k] != value for k, value in score.items()):
            raise ValueError('saved score differs from raw-output rescoring')
    return citations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    root = Path('data/staged/m0_frozen_v1')
    receipt = json.loads((root / 'validation.json').read_text())
    for name, digest in receipt['sha256'].items():
        if hashlib.sha256((root / f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise ValueError('frozen benchmark inputs changed')
    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]
    settings = json.loads((args.run / 'settings.json').read_text())
    split = settings.get('split', 'test')
    if split not in ('dev', 'test'):
        raise ValueError('invalid evaluation split')
    tasks = [r for r in rows(root / 'manifest.jsonl')
             if r['official_eligible'] and r['original_split'] == split]
    gold = {r['task_id']: r for r in rows(root / 'gold.jsonl')}
    evidence = {r['evidence_id']: r for r in rows(root / 'evidence.jsonl')}
    records = rows(args.run / 'records.jsonl')
    if settings['receipt'] != receipt:
        raise ValueError('run receipt differs from frozen data')
    citations = validate_rows(records, tasks, gold, evidence)
    summary = json.loads((args.run / 'summary.json').read_text())
    if summary['inputs'] != len(tasks) or summary['calls'] != len(records):
        raise ValueError('summary coverage differs from actual records')
    if set(summary['results']) != {'finqa/direct', 'finqa/calculator', 'tatqa/direct'}:
        raise ValueError('missing or unexpected summary groups')
    baseline = rows(args.baseline / 'records.jsonl') if args.baseline else None
    if baseline is not None:
        validate_rows(baseline, tasks, gold, evidence)
        base_settings = json.loads((args.baseline / 'settings.json').read_text())
        if base_settings.get('split', 'test') != split:
            raise ValueError('paired evaluation splits differ')
        for field in ('protocol', 'receipt', 'max_input_tokens', 'max_output_tokens',
                      'backend', 'runtime_precision', 'evaluation_policy', 'ci'):
            if base_settings[field] != settings[field]:
                raise ValueError(f'paired protocol differs: {field}')
        if base_settings.get('tatqa_system_suffix', '') != settings.get('tatqa_system_suffix', ''):
            raise ValueError('paired protocol differs: tatqa_system_suffix')
    paired = {}
    for name, result in summary['results'].items():
        dataset, condition = name.split('/')
        selected = [r for r in records if r['dataset'] == dataset and r['condition'] == condition]
        scored = [r for r in selected if r['labels_available']]
        if (result['inputs'], result['scored'], result['unscored']) != (
                len(selected), len(scored), len(selected) - len(scored)):
            raise ValueError('summary denominators changed')
        metric_names = {'em', 'f1', 'scale'} if dataset == 'tatqa' else {'em'}
        if set(result['metrics']) != metric_names:
            raise ValueError('missing or unexpected summary metrics')
        errors = sum(bool(r.get('error') or r.get('parse_error')
                          or not r.get('stopped_on_eos')) for r in selected)
        if result['errors'] != errors:
            raise ValueError('summary error count differs from actual records')
        for metric, expected in result['metrics'].items():
            if cluster_interval(scored, metric) != expected:
                raise ValueError('summary interval differs from saved observations')
            if baseline is not None:
                left = [r for r in baseline if r['dataset'] == dataset
                        and r['condition'] == condition and r['labels_available']]
                paired[f'{name}/{metric}'] = paired_interval(left, scored, metric)
    validation = {'passed': True, 'calls': len(records), 'resolved_citations': citations,
        'records_sha256': hashlib.sha256((args.run / 'records.jsonl').read_bytes()).hexdigest(),
        'paired_difference_candidate_minus_baseline': paired,
        'baseline': str(args.baseline) if args.baseline else None,
        'meaning': 'Completeness, raw-output scores, cited scalar execution and report-cluster intervals; not a new blind test.'}
    (args.run / 'validation.json').write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
