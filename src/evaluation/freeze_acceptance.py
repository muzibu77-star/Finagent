"""Freeze 80 independent tasks before agent tuning; keep labels outside the corpus."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import zipfile

from src.data.build_manifest import financial_source, visible_evidence


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    root=Path('data/staged/m0_frozen_v1')
    def rows(name):
        return list(map(json.loads,(root/f'{name}.jsonl').read_text().splitlines()))
    manifest=rows('manifest')
    evidence={r['evidence_id']:r for r in rows('evidence')}
    questions={r['task_id']:r['question'] for r in rows('questions')}
    gold={r['task_id']:r['gold'] for r in rows('gold')}
    candidates=[r for r in manifest if r['dataset']=='finqa' and r['original_split']=='test'
                and r['official_eligible']]
    random.Random(73).shuffle(candidates)
    corpus={r['evidence_id']:{'evidence':evidence[r['evidence_id']],
            'source_report_id':r['source_report_id'],'company':r['company'],
            'period':str(r['report_period'])} for r in candidates}
    tasks, labels = [], {}
    def add(category, company, period, turns, expected, source_ids):
        task_id=f'accept_{len(tasks):03d}'
        tasks.append({'task_id':task_id,'category':category,'company':company,
                      'period':period,'turns':turns,'snapshot_id':'m3-frozen-v1'})
        labels[task_id]={'expected':expected,'evidence_ids':source_ids}
    used=set()
    for r in candidates:
        if r['source_report_id'] in used: continue
        used.add(r['source_report_id'])
        add('calculation',r['company'],str(r['report_period']),[questions[r['task_id']]],
            [[gold[r['task_id']]['exe_ans']]],[[r['evidence_id']]])
        if len(tasks)==30: break
    by_company=defaultdict(dict)
    for r in candidates:
        by_company[r['company']].setdefault(str(r['report_period']),r)
    for company,reports in sorted(by_company.items()):
        if len(reports)<2: continue
        a,b=[reports[y] for y in sorted(reports)[:2]]
        question=(f"Answer both independent questions, with a separate calculation for each. "
                  f"Report {a['report_period']}: {questions[a['task_id']]} "
                  f"Report {b['report_period']}: {questions[b['task_id']]}")
        add('cross_evidence',company,[str(a['report_period']),str(b['report_period'])],
            [question],[[gold[a['task_id']]['exe_ans'],gold[b['task_id']]['exe_ans']]],
            [[a['evidence_id'],b['evidence_id']]])
        if len(tasks)==50: break
    if len(tasks)!=50: raise ValueError('not enough independent cross-report pairs')
    test_sources={r['source_report_id'] for r in manifest if r['group_split']=='test'
                  and r['dataset']=='finqa' and not r['exclusion_reason'] in
                  ('near_duplicate_cross_report','unresolved_2019_alias')}
    trained_sources={r['source_report_id'] for r in manifest if r['official_eligible']
                     and r['original_split']=='train'}
    development_sources={r['source_report_id'] for r in manifest if r['official_eligible']
                         and r['original_split']=='dev'}
    with zipfile.ZipFile('data/raw/convfinqa/data.zip') as archive:
        conversations=json.loads(archive.read('data/train.json'))+json.loads(archive.read('data/dev.json'))
    random.Random(73).shuffle(conversations)
    conversation_sources=set()
    for row in conversations:
        source='fin:'+financial_source(row['filename'])
        annotation=row['annotation']
        if (source not in test_sources or source in trained_sources|development_sources|conversation_sources
                or not annotation.get('exe_ans_list') or len(annotation['dialogue_break'])<2):
            continue
        eid='conv:'+row['id']
        context={'evidence_id':eid,**visible_evidence('finqa',row)}
        company,period=source.removeprefix('fin:').split('/')
        corpus[eid]={'evidence':context,'source_report_id':source,'company':company,'period':period}
        turns=annotation['dialogue_break']
        answers=annotation['exe_ans_list']
        if len(turns)!=len(answers): raise ValueError('conversation label mismatch')
        add('multi_turn',company,period,turns,[[a] for a in answers],[[eid] for _ in turns])
        conversation_sources.add(source)
        if len(tasks)==65: break
    if len(tasks)!=65: raise ValueError('not enough held-out labeled conversations')
    for i,r in enumerate(candidates[:15]):
        company=r['company'] if i>=5 else None
        period=None if 5<=i<10 else ('2099' if i>=10 else str(r['report_period']))
        outcome='insufficient_evidence' if i>=10 else 'awaiting_input'
        add('insufficient_or_missing_scope',company,period,[questions[r['task_id']]],
            [outcome],[[]])
    if len(tasks)!=80: raise ValueError('acceptance task count mismatch')
    if {r['source_report_id'] for r in corpus.values()} & (trained_sources|development_sources):
        raise ValueError('acceptance source leakage')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name,value in [('tasks',tasks),('gold',labels),('corpus',corpus)]:
        (args.output_dir/f'{name}.json').write_text(json.dumps(value,indent=2))
    lock={'seed':73,'tasks':80,'counts':{c:sum(t['category']==c for t in tasks)
        for c in sorted({t['category'] for t in tasks})},
        'turns':sum(len(t['turns']) for t in tasks),'corpus_entries':len(corpus),
        'sha256':{n:hashlib.sha256((args.output_dir/f'{n}.json').read_bytes()).hexdigest()
                  for n in ('tasks','gold','corpus')},
        'score':'all required values, source IDs and terminal business state; report separately by category',
        'limits':'Source-held-out custom tasks; not an official benchmark. Missing-scope controls are derived.'}
    (args.output_dir/'lock.json').write_text(json.dumps(lock,indent=2))
    print(json.dumps(lock))


if __name__=='__main__':
    main()
