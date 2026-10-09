"""Recompute scalar scores and paired membership from saved M5 generations."""

import argparse
import hashlib
import json
from pathlib import Path

from src.evaluation.chinese_run import check
from src.evaluation.numeric_score import parse_numeric, scalar, score


def audit(run: Path, chinese: bool) -> dict:
    settings = json.loads((run / 'settings.json').read_text())
    root = Path(settings.get('tasks_root', 'data/staged/m5_chinese_v1')
                if chinese else settings['config']['input_root'])
    tasks = json.loads((root / 'tasks.json').read_text())
    gold = json.loads((root / 'gold.json').read_text())
    rows = [json.loads(line) for line in (run / 'records.jsonl').read_text().splitlines()]
    lock = settings['lock']
    for name, digest in lock['sha256' if chinese else 'hashes'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    field = 'mode' if chinese else 'condition'
    modes = settings.get('modes', ['text', 'vision']) if chinese else ['text', 'vision', 'hybrid']
    if chinese and (not modes or len(set(modes)) != len(modes)
                    or not set(modes) <= {'text', 'vision'}):
        raise ValueError('invalid frozen Chinese modes')
    expected = {(t['id'], mode) for t in tasks for mode in modes}
    if not chinese:
        expected.update((t['id'], 'occluded') for t in tasks[:3])
        monetary = [t for t in tasks if gold[t['id']]['answer_type'] == 'span'
                    and gold[t['id']]['scale'] in {'thousand', 'million', 'billion'}][:3]
        expected.update((t['id'], 'base_units') for t in monetary)
    assert len(rows) == len(expected)
    assert {(r['id'], r[field]) for r in rows} == expected
    for row in rows:
        if row.get('error') or row.get('parse_error') or not row.get('stopped_on_eos'):
            assert not row.get('passed', row.get('numeric_em', False))
            continue
        target = gold[row['id']]
        if chinese:
            result = check(row['text'], target)
            assert all(row[key] == value for key, value in result.items())
            assert row['passed'] == (result['numeric_correct'] and result['unit_correct'])
        elif row['condition'] == 'occluded':
            assert row['abstained'] == (json.loads(row['text']) == {'answer': None, 'scale': ''})
        else:
            if row['condition'] == 'base_units':
                target = {**target, 'answer': scalar(target['answer'][0], target['scale']), 'scale': ''}
            result = score(parse_numeric(row['text']), target)
            assert all(row[key] == value for key, value in result.items())
            override = settings['label_audit']['overrides'].get(row['id'])
            audited = {**target, 'scale': override['scale']} if override else target
            assert row['source_audited_numeric_em'] == score(parse_numeric(row['text']), audited)['numeric_em']
    result = {'passed': True, 'calls': len(rows), 'unique_pairs': len(expected),
              'scoring': 'Recomputed from saved raw generations; failures kept in denominator.'}
    (run / 'validation.json').write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--chinese', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.chinese)))


if __name__ == '__main__':
    main()
