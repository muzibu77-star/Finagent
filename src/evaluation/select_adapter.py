"""Apply recorded pilot gates; retain the base when any required gate fails."""

import json
import math
from pathlib import Path


def paired_counts(base: dict, candidate: dict) -> dict:
    if set(base) != set(candidate):
        raise ValueError('paired tasks differ')
    wins = sum(candidate[k] and not base[k] for k in base)
    losses = sum(base[k] and not candidate[k] for k in base)
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses)+1))/2**n) if n else 1.0
    return {'wins': wins, 'losses': losses, 'exact_mcnemar_p': p}


def main() -> None:
    policy = json.loads(Path('configs/m2_selection.json').read_text())
    condition = policy['primary_condition']
    base_dir = Path('artifacts/m1_native_v1')
    baseline = json.loads((base_dir/'summary.json').read_text())
    regression_base = json.loads(Path('artifacts/m2_regression_base/summary.json').read_text())
    def scores(path):
        return {r['task_id']:r['correct'] for r in map(json.loads,(path/'records.jsonl').read_text().splitlines())
                if r['condition']==condition}
    base = scores(base_dir)
    candidates = {}
    for arm, adapter in [('action','artifacts/m2_action_v2/step_0040'),
                         ('answer','artifacts/m2_answer_v1/step_0040')]:
        path = Path(f'artifacts/m2_eval_{arm}_v1')
        summary = json.loads((path/'summary.json').read_text())
        regression = json.loads(Path(f'artifacts/m2_regression_{arm}/summary.json').read_text())
        if regression['frozen_sha256'] != regression_base['frozen_sha256']:
            raise ValueError('regression inputs changed')
        if summary['task_sha256'] != baseline['task_sha256']:
            raise ValueError('development task inputs changed')
        paired = paired_counts(base, scores(path))
        ratio = summary['results'][condition]['latency_s']/baseline['results'][condition]['latency_s']
        gates = {'quality': paired['wins'] > paired['losses'] and paired['exact_mcnemar_p'] < policy['requires_paired_exact_mcnemar_p_below'],
                 'general': regression['passed']['text'] >= regression_base['passed']['text'],
                 'visual': regression['passed']['synthetic_image'] >= regression_base['passed']['synthetic_image'],
                 'latency': ratio <= policy['max_primary_latency_ratio_to_base']}
        candidates[arm] = {'adapter':adapter, **paired, 'latency_ratio':ratio,
                           'gates':gates,'eligible':all(gates.values()),
                           'primary_correct':summary['results'][condition]['correct']}
    eligible = [arm for arm in candidates if candidates[arm]['eligible']]
    selected = max(eligible,key=lambda arm:candidates[arm]['primary_correct']) if eligible else 'base'
    result = {'selected':selected,'adapter':candidates[selected]['adapter'] if selected!='base' else None,
              'policy':policy,'candidates':candidates,
              'limits':'Exploratory small development pilot; p-values unadjusted, not a confirmatory generalization claim.'}
    output = Path('artifacts/m2_provenance/decision.json')
    if output.exists(): raise ValueError('decision exists; preserve prior record')
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
