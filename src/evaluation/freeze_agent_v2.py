"""Freeze new development/acceptance tasks, excluding previously exercised reports."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import zipfile

from src.data.build_manifest import financial_source, visible_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    args = parser.parse_args()
    root = Path('data/staged/m0_frozen_v1')
    receipt = json.loads((root / 'validation.json').read_text())
    for name, expected in receipt['sha256'].items():
        if hashlib.sha256((root / f'{name}.jsonl').read_bytes()).hexdigest() != expected:
            raise ValueError('frozen data changed')
    def rows(name):
        return [json.loads(line) for line in (root / f'{name}.jsonl').read_text().splitlines()]
    manifest = rows('manifest')
    evidence = {row['evidence_id']: row for row in rows('evidence')}
    questions = {row['task_id']: row['question'] for row in rows('questions')}
    gold = {row['task_id']: row['gold'] for row in rows('gold')}
    old_root = Path('data/staged/m3_acceptance_v1')
    old_corpus = json.loads((old_root / 'corpus.json').read_text())
    old_gold = json.loads((old_root / 'gold.json').read_text())
    old_sources = {old_corpus[key]['source_report_id'] for label in old_gold.values()
                   for turn in label['evidence_ids'] for key in turn}
    dev_tasks = json.loads(Path('artifacts/m1_baseline_v1/tasks.json').read_text())
    by_id = {row['task_id']: row for row in manifest}
    old_sources.update(by_id[t['task_id']]['source_report_id'] for t in dev_tasks)
    allowed = [row for row in manifest if row['dataset'] == 'finqa'
               and row['original_split'] == args.split and row['official_eligible']
               and row['source_report_id'] not in old_sources]
    random.Random(107).shuffle(allowed)
    snapshot = f'agent-v2-{args.split}'
    corpus = {row['evidence_id']: {'evidence': evidence[row['evidence_id']],
        'source_report_id': row['source_report_id'], 'company': row['company'],
        'period': str(row['report_period'])} for row in allowed}
    tasks, labels = [], {}
    def add(category, company, period, turns, expected, source_ids):
        key = f'{snapshot}-{len(tasks):03d}'
        tasks.append({'task_id': key, 'category': category, 'company': company,
                      'period': period, 'turns': turns, 'snapshot_id': snapshot})
        labels[key] = {'expected': expected, 'evidence_ids': source_ids}
    used = set()
    for row in allowed:
        if row['source_report_id'] in used:
            continue
        used.add(row['source_report_id'])
        add('calculation', row['company'], str(row['report_period']),
            [questions[row['task_id']]], [[gold[row['task_id']]['exe_ans']]], [[row['evidence_id']]])
        if len(tasks) == 30:
            break
    companies = defaultdict(dict)
    for row in allowed:
        companies[row['company']].setdefault(str(row['report_period']), row)
    cross_count = 0
    for company, periods in sorted(companies.items()):
        if len(periods) < 2:
            continue
        first, second = [periods[p] for p in sorted(periods)[:2]]
        prompt = (f"Answer both questions using a separate calculation for each. "
                  f"Report {first['report_period']}: {questions[first['task_id']]} "
                  f"Report {second['report_period']}: {questions[second['task_id']]}")
        add('cross_evidence', company, [str(first['report_period']), str(second['report_period'])],
            [prompt], [[gold[first['task_id']]['exe_ans'], gold[second['task_id']]['exe_ans']]],
            [[first['evidence_id'], second['evidence_id']]])
        cross_count += 1
        if cross_count == 20:
            break
    eligible_sources = {row['source_report_id'] for row in allowed}
    with zipfile.ZipFile('data/raw/convfinqa/data.zip') as archive:
        conversations = json.loads(archive.read('data/train.json')) + json.loads(archive.read('data/dev.json'))
    random.Random(107).shuffle(conversations)
    conversation_sources = set()
    for row in conversations:
        source = 'fin:' + financial_source(row['filename'])
        annotation = row['annotation']
        turns, answers = annotation['dialogue_break'], annotation.get('exe_ans_list', [])
        if source not in eligible_sources or source in conversation_sources or len(turns) < 2 or len(turns) != len(answers):
            continue
        key = 'conv:' + row['id']
        company, period = source.removeprefix('fin:').split('/')
        corpus[key] = {'evidence': {'evidence_id': key, **visible_evidence('finqa', row)},
                       'source_report_id': source, 'company': company, 'period': period}
        add('multi_turn', company, period, turns, [[answer] for answer in answers], [[key] for _ in turns])
        conversation_sources.add(source)
        if len(conversation_sources) == 15:
            break
    for i, row in enumerate(allowed[:15]):
        company = row['company'] if i >= 5 else None
        period = None if 5 <= i < 10 else ('2099' if i >= 10 else str(row['report_period']))
        outcome = 'insufficient_evidence' if i >= 10 else 'awaiting_input'
        add('insufficient_or_missing_scope', company, period, [questions[row['task_id']]], [outcome], [[]])
    if not tasks:
        raise ValueError('no independent source pool remains')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in [('tasks', tasks), ('gold', labels), ('corpus', corpus)]:
        (args.output_dir / f'{name}.json').write_text(json.dumps(value, indent=2))
    lock = {'seed': 107, 'split': args.split, 'tasks': len(tasks),
            'counts': {kind: sum(t['category'] == kind for t in tasks) for kind in
                       ('calculation', 'cross_evidence', 'multi_turn', 'insufficient_or_missing_scope')},
            'turns': sum(len(t['turns']) for t in tasks), 'corpus_entries': len(corpus),
            'excluded_historical_reports': sorted(old_sources),
            'source_reports': sorted(eligible_sources),
            'sha256': {name: hashlib.sha256((args.output_dir / f'{name}.json').read_bytes()).hexdigest()
                       for name in ('tasks', 'gold', 'corpus')},
            'limits': 'Counts reflect remaining source-isolated pool; no replacement with old acceptance tasks.'}
    (args.output_dir / 'lock.json').write_text(json.dumps(lock, indent=2))
    print(json.dumps({k: v for k, v in lock.items() if not k.endswith('reports')}))


if __name__ == '__main__':
    main()
