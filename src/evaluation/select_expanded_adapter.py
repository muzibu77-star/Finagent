"""Apply predeclared development-only adoption gates to expanded SFT models."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from src.evaluation.statistics import paired_interval


def development_rows(root: Path) -> tuple[list[dict], dict]:
    settings = json.loads((root / 'settings.json').read_text())
    if settings.get('split') != 'dev':
        raise ValueError('model selection requires development data, never test data')
    validation = json.loads((root / 'validation.json').read_text())
    if (not validation['passed'] or validation['records_sha256'] !=
            hashlib.sha256((root / 'records.jsonl').read_bytes()).hexdigest()):
        raise ValueError('model selection requires a current independent audit')
    return [json.loads(s) for s in (root / 'records.jsonl').read_text().splitlines()], settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(Path('configs/jd_evaluation_protocol.json').read_text())
    baseline, reference_settings = development_rows(args.root / 'dev_historical_action')
    regression = json.loads((args.root / 'regression_historical_action/summary.json').read_text())

    def primary(rows):
        return [r for r in rows if r['dataset'] == 'finqa' and r['condition'] == 'calculator']

    def tatqa_f1(rows):
        return statistics.mean(r['f1'] for r in rows if r['dataset'] == 'tatqa' and r['labels_available'])

    left = primary(baseline)
    base_latency = sum(r.get('latency_s', 0) for r in left)
    candidates = {}
    for name in ('expanded_action', 'expanded_answer'):
        rows, settings = development_rows(args.root / f'dev_{name}')
        for field in ('receipt', 'max_input_tokens', 'max_output_tokens', 'backend',
                      'runtime_precision', 'evaluation_policy', 'workers'):
            if settings[field] != reference_settings[field]:
                raise ValueError('candidate development conditions differ')
        if settings.get('tatqa_system_suffix', '') != reference_settings.get('tatqa_system_suffix', ''):
            raise ValueError('candidate TAT-QA prompts differ; not a model-only comparison')
        if settings['evaluation_policy'] != policy:
            raise ValueError('adoption policy changed after evaluation')
        candidate_regression = json.loads((args.root / f'regression_{name}/summary.json').read_text())
        if candidate_regression['frozen_sha256'] != regression['frozen_sha256']:
            raise ValueError('capability regression inputs differ')
        right = primary(rows)
        difference = paired_interval(left, right, 'em')
        latency = sum(r.get('latency_s', 0) for r in right) / base_latency
        gates = {'primary_quality': difference['ci95'][0] > 0,
                 'tatqa_non_decrease': tatqa_f1(rows) >= tatqa_f1(baseline),
                 'general_tools': candidate_regression['passed']['text'] >= regression['passed']['text'],
                 'synthetic_visual': candidate_regression['passed']['synthetic_image'] >= regression['passed']['synthetic_image'],
                 'latency': latency <= policy['adoption_gates']['maximum_calculator_latency_ratio']}
        candidates[name] = {'gates': gates, 'eligible': all(gates.values()),
            'primary_accuracy': statistics.mean(r['em'] for r in right),
            'paired_primary_difference': difference, 'latency_ratio': latency,
            'tatqa_f1': tatqa_f1(rows)}
    eligible = [name for name, value in candidates.items() if value['eligible']]
    selected = max(eligible, key=lambda name: (candidates[name]['primary_accuracy'],
                   -candidates[name]['latency_ratio'])) if eligible else 'historical_action'
    result = {'selected': selected, 'policy': policy, 'candidates': candidates,
              'scope': 'Development selection under the recorded evaluation backend and precision; full test has not been used.'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'decision.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
