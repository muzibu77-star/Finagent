"""Frozen 40-task FinQA development comparison with separate retrieval metrics."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import time

import torch
from transformers import AutoTokenizer
import yaml

from src.evaluation.m0_inference_check import generate, load_model
from src.retrieval.bm25 import EvidenceIndex
from src.tools.benchmark_tools import execute_action, extract_numeric_facts, parse_action

LOG = logging.getLogger(__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/staged/m0_frozen_v1'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    root = args.data_dir
    evidence = {r['evidence_id']: r for r in read_rows(root / 'evidence.jsonl')}
    questions = {r['task_id']: r for r in read_rows(root / 'questions.jsonl')}
    gold = {r['task_id']: r['gold'] for r in read_rows(root / 'gold.jsonl')}
    manifest = read_rows(root / 'manifest.jsonl')
    candidates = [r for r in manifest if r['dataset'] == 'finqa'
                  and r['original_split'] == 'dev' and r['official_eligible']]
    random.Random(29).shuffle(candidates)
    tasks, used_reports = [], set()
    for row in candidates:
        if row['source_report_id'] not in used_reports:
            tasks.append(row)
            used_reports.add(row['source_report_id'])
        if len(tasks) == 40:
            break
    if len(tasks) != 40:
        raise ValueError('need 40 distinct development reports')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'tasks.json').write_text(json.dumps(tasks, indent=2))
    index = EvidenceIndex(str(args.output_dir / 'retrieval.sqlite'))
    allowed = {row['evidence_id'] for row in candidates}
    for key in sorted(allowed):
        index.add(evidence[key])
    retrieval = []
    for row in tasks:
        hits = index.search(questions[row['task_id']]['question'])
        retrieval.append({'task_id': row['task_id'], 'hits': hits,
                          'hit_at_5': row['evidence_id'] in hits})
    index.close()
    (args.output_dir / 'retrieval.json').write_text(json.dumps(retrieval, indent=2))
    config = yaml.safe_load(Path('configs/m0_inference.yaml').read_text())
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['path'], local_files_only=True)
    model = load_model(config['model']['path'], None, config['model']['device'])
    records = []
    start = time.monotonic()
    for condition in ('direct', 'calculator'):
        for row in tasks:
            task_id = row['task_id']
            context = evidence[row['evidence_id']]
            facts = extract_numeric_facts(context)
            prompt = ('Use only the supplied financial evidence. Return no explanation. '
                      'Percentages in the numeric result should be fractions (25% = 0.25).\n')
            if condition == 'direct':
                prompt += 'Return exactly JSON {"answer": NUMBER_OR_YES_NO}.\n'
            else:
                prompt += (
                    'Return exactly JSON {"steps":[{"operation":"subtract","args":["f1","f0"]},'
                    '{"operation":"divide","args":["#0","f0"]}]}. '
                    'This is a schema example, not the solution. Use 1-8 steps. '
                    'Operations: add, subtract, multiply, divide, exp, greater. '
                    'Args must be fact IDs below, earlier #step results, or const_0, const_1, '
                    'const_2, const_3, const_4, const_100, const_m1. '
                    'Last step is the result. Never invent fact IDs or numeric literals.\n'
                )
            # Both conditions receive exactly the same evidence and fact catalogue.
            catalogue = {key: item['value_token'] + ' @ ' + (
                f"cell[{item['location']['row']},{item['location']['column']}]"
                if 'row' in item['location'] else
                f"{item['location']['section']}[{item['location']['paragraph']}]"
            ) for key, item in facts.items()}
            prompt += json.dumps({'evidence': context, 'facts': catalogue,
                                  'question': questions[task_id]['question']})
            record = generate(model, tokenizer, [{'role': 'user', 'content': prompt}],
                None, config['generation'], 512, 4096, config['model']['device'])
            record.update({'task_id': task_id, 'condition': condition, 'correct': False})
            if 'error' not in record:
                try:
                    if not record['stopped_on_eos']:
                        raise ValueError('output truncated')
                    action = parse_action(record['text'])
                    if condition == 'calculator':
                        calculation = execute_action(action, facts)
                        prediction = calculation['value']
                        record['calculation'] = calculation
                    else:
                        if set(action) != {'answer'}:
                            raise ValueError('expected answer only')
                        prediction = action['answer']
                        if isinstance(prediction, bool):
                            raise ValueError('boolean answer is not numeric')
                        if prediction not in ('yes', 'no'):
                            prediction = round(float(prediction), 5)
                    record['prediction'] = prediction
                    record['correct'] = prediction == gold[task_id]['exe_ans']
                    record['expected'] = gold[task_id]['exe_ans']
                except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
                    record['error'] = f'{type(exc).__name__}: {exc}'
            records.append(record)
            with (args.output_dir / 'records.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            elapsed = time.monotonic() - start
            LOG.info('%d/80 %s %s correct=%s elapsed=%.1fs ETA=%.1fs',
                len(records), condition, task_id, record['correct'], elapsed,
                elapsed / len(records) * (80 - len(records)))
    summary = {'tasks': 40, 'protocol': 'FinQA official execution value equality on filtered dev subset',
        'evidence_condition': 'same provided local evidence; retrieval scored separately',
        'retrieval_recall_at_5': sum(r['hit_at_5'] for r in retrieval) / len(retrieval),
        'retrieval_pool_documents': len(allowed),
        'results': {c: {'correct': sum(r['correct'] for r in records if r['condition'] == c),
                       'total': 40, 'errors': sum('error' in r for r in records if r['condition'] == c),
                       'latency_s': sum(r.get('latency_s', 0) for r in records if r['condition'] == c)}
                    for c in ('direct', 'calculator')},
        'config': config, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['FinQA only; no TAT-QA or full benchmark score claimed.',
            'Valid source references do not alone prove metric/period selection.',
            'Development tasks are not independent final acceptance tasks.']}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
