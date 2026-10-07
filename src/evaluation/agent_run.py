"""Independent frozen 80-task/115-turn business evaluation, separate from fault tests."""

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import time

from src.agent.runner import Runner
from src.data.corpus import Corpus
from src.evidence.identity import evidence_identity
from src.harness.store import Store
from src.model.agent import ModelDriver
from src.model.financial import predict


def score_turn(row: dict, expected, evidence_ids: list[str], identities: dict) -> dict:
    if isinstance(expected, str):
        correct = row['execution']=='succeeded' and row['business']==expected
        if expected=='awaiting_input': correct = correct and row['report'] is None
        return {'numeric_correct':None,'source_supported':None,'passed':bool(correct)}
    calculations = (row.get('report') or {}).get('calculations',[])
    numeric = (row['execution']=='succeeded' and row['business']=='answered'
               and [c['value'] for c in calculations]==expected)
    supported = len(calculations)==len(evidence_ids)
    if supported:
        for calc, expected_id in zip(calculations,evidence_ids):
            cited={identities[f['evidence_id']] for f in calc['facts'].values()}
            supported = supported and cited=={identities[expected_id]}
    return {'numeric_correct':numeric,'source_supported':supported,'passed':numeric and supported}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    root=Path('data/staged/m3_acceptance_v1')
    lock=json.loads(Path('configs/m3_acceptance_lock.json').read_text())
    for name,digest in lock['sha256'].items():
        if hashlib.sha256((root/f'{name}.json').read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen acceptance inputs changed')
    tasks=json.loads((root/'tasks.json').read_text())
    gold=json.loads((root/'gold.json').read_text())
    if args.resume:
        if not args.output_dir.is_dir(): raise ValueError('resume directory missing')
    else:
        args.output_dir.mkdir(parents=True,exist_ok=False)
    settings={'adapter':str(args.adapter) if args.adapter else None,'lock':lock,
              'max_calls':10,'task_seconds':180,'max_input_tokens':8192,'max_output_tokens':512}
    settings_path=args.output_dir/'settings.json'
    if args.resume:
        if json.loads(settings_path.read_text())!=settings: raise ValueError('resume config mismatch')
    else: settings_path.write_text(json.dumps(settings,indent=2))
    corpus=Corpus(str(root/'corpus.json'),'m3-frozen-v1')
    identities={key:evidence_identity(doc['evidence'],doc['source_report_id'])
                for key,doc in corpus.documents.items()}
    store=Store(str(args.output_dir/'tasks.sqlite'))
    records_path=args.output_dir/'records.jsonl'
    records=list(map(json.loads,records_path.read_text().splitlines())) if records_path.exists() else []
    done={r['task_id'] for r in records}
    turn_path=args.output_dir/'turns.jsonl'
    old_turns=list(map(json.loads,turn_path.read_text().splitlines())) if turn_path.exists() else []
    completed_turns={(r['task_id'],r['turn']):r['result'] for r in old_turns}
    fixed_path=args.output_dir/'fixed.jsonl'
    old_fixed=list(map(json.loads,fixed_path.read_text().splitlines())) if fixed_path.exists() else []
    fixed_results={r['task_id']:r['result'] for r in old_fixed}
    started=time.monotonic()
    with ModelDriver(str(args.adapter) if args.adapter else None) as model:
        store.recover()  # The driver owns the exclusive process lock before recovery.
        runner=Runner(store,corpus,model)
        for task in tasks:
            task_id=task['task_id']
            if task_id in done: continue
            turns=[]
            fixed=None
            for i,question in enumerate(task['turns']):
                payload={k:task[k] for k in ('company','period','snapshot_id')}
                payload['question']=question
                request_id=f'{task_id}:turn{i}'
                if (task_id,i) in completed_turns:
                    turn=completed_turns[(task_id,i)]
                else:
                    receipt=store.submit(request_id,payload,task_id=f'{task_id}:turn0' if i else None)
                    if store.get(receipt['task_id'])['revision']!=receipt['revision']:
                        raise ValueError('unrecorded stale acceptance turn')
                    row=runner.run(receipt['task_id'])
                    turn=score_turn(row,gold[task_id]['expected'][i],gold[task_id]['evidence_ids'][i],identities)
                    turn.update(revision=row['revision'],execution=row['execution'],business=row['business'],
                                report=row['report'],attempts=row['state'].get('attempts',0))
                    with turn_path.open('a') as output:
                        output.write(json.dumps({'task_id':task_id,'turn':i,'result':turn})+'\n')
                turns.append(turn)
                if i==0 and task_id in fixed_results:
                    fixed=fixed_results[task_id]
                elif i==0 and task['category']=='calculation':
                    hits=corpus.search(question,corpus.allowed(payload))
                    if hits:
                        selected=hits[0]['evidence_id']
                        config=copy.deepcopy(model.config)
                        config['generation']['max_input_tokens']=8192
                        fixed=predict(model.model,model.tokenizer,corpus.documents[selected]['evidence'],
                                      question,'calculator',config)
                        fixed['source_supported']=identities[selected]==identities[gold[task_id]['evidence_ids'][0][0]]
                        fixed['correct']=('error' not in fixed and fixed.get('prediction')==gold[task_id]['expected'][0][0])
                    else: fixed={'error':'no retrieved evidence','correct':False,'source_supported':False}
                    with fixed_path.open('a') as output:
                        output.write(json.dumps({'task_id':task_id,'result':fixed})+'\n')
            record={'task_id':task_id,'category':task['category'],'turns':turns,
                    'all_turns_passed':all(t['passed'] for t in turns),'fixed':fixed}
            records.append(record)
            with records_path.open('a') as output: output.write(json.dumps(record)+'\n')
            elapsed=time.monotonic()-started
            completed=len(records)-len(done)
            logging.info('%d/%d tasks category=%s passed=%s elapsed=%.1fs ETA=%.1fs',
                len(records),len(tasks),task['category'],record['all_turns_passed'],elapsed,
                elapsed/max(1,completed)*(len(tasks)-len(records)))
    corpus.close()
    categories=sorted({r['category'] for r in records})
    summary={'settings':settings,'tasks':len(records),'turns':sum(len(r['turns']) for r in records),
        'categories':{c:{'passed':sum(r['all_turns_passed'] for r in records if r['category']==c),
                         'total':sum(r['category']==c for r in records)} for c in categories},
        'fixed_30_calculation_tasks':{'correct_with_source':sum(r['fixed']['correct'] and r['fixed']['source_supported'] for r in records if r['fixed']),
            'total':sum(r['fixed'] is not None for r in records)},
        'elapsed_this_process_s':time.monotonic()-started,
        'limits':['Frozen custom held-out task set, not official full benchmark.',
                  'Scope guards handle missing conditions without model calls.',
                  'Passing means numeric answer and exact local-source support; not audited financial interpretation.']}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
