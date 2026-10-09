"""Paired native-tool and end-to-end retrieval evaluation on frozen tasks."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer
from peft import PeftModel
import yaml

from src.evaluation.m0_inference_check import load_model
from src.model.financial import TOOLS, decode_prediction, messages_for, predict
from src.model.http_driver import HttpDriver
from src.retrieval.bm25 import EvidenceIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, default=Path('artifacts/m1_baseline_v1/tasks.json'))
    parser.add_argument('--data-dir', type=Path, default=Path('data/staged/m0_frozen_v1'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--server-url')
    parser.add_argument('--served-model', default='base')
    args = parser.parse_args()
    if args.adapter and args.server_url:
        parser.error('use the declared served model for server-side adapters')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    def rows(name):
        return [json.loads(x) for x in (args.data_dir / f'{name}.jsonl').read_text().splitlines()]
    tasks = json.loads(args.tasks.read_text())
    evidence = {r['evidence_id']: r for r in rows('evidence')}
    questions = {r['task_id']: r for r in rows('questions')}
    gold = {r['task_id']: r['gold'] for r in rows('gold')}
    allowed = {r['evidence_id'] for r in rows('manifest') if r['dataset'] == 'finqa'
               and r['original_split'] == 'dev' and r['official_eligible']}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    index = EvidenceIndex(str(args.output_dir / 'retrieval.sqlite'))
    for key in sorted(allowed): index.add(evidence[key])
    config = yaml.safe_load(Path('configs/m0_inference.yaml').read_text())
    torch.manual_seed(0)
    driver = HttpDriver(args.server_url, args.served_model) if args.server_url else None
    if driver is None:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['path'], local_files_only=True)
        model = load_model(config['model']['path'], None, config['model']['device'])
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)
            model.eval()
    records = []
    started = time.monotonic()
    conditions = ('direct', 'calculator', 'retrieval_calculator')
    for condition in conditions:
        for task in tasks:
            question = questions[task['task_id']]['question']
            hits = index.search(question)
            selected = task['evidence_id'] if condition != 'retrieval_calculator' else (hits[0] if hits else None)
            if selected is None:
                record = {'error': 'no retrieved evidence'}
            else:
                task_condition = 'direct' if condition == 'direct' else 'calculator'
                if driver is None:
                    record = predict(model, tokenizer, evidence[selected], question, task_condition, config)
                else:
                    messages, facts = messages_for(evidence[selected], question, task_condition)
                    record = decode_prediction(driver.generate(messages,
                        TOOLS if task_condition == 'calculator' else None,
                        config['generation'].get('max_new_tokens', 512),
                        config['generation'].get('max_input_tokens', 4096)), facts, task_condition)
            expected = gold[task['task_id']]['exe_ans']
            record.update({'task_id': task['task_id'], 'condition': condition,
                'selected_evidence': selected, 'expected': expected,
                'correct': 'error' not in record and record.get('prediction') == expected,
                'correct_evidence': selected == task['evidence_id'],
                'retrieval_hit_at_5': task['evidence_id'] in hits})
            records.append(record)
            with (args.output_dir / 'records.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            elapsed = time.monotonic() - started
            total = len(tasks) * len(conditions)
            logging.info('%d/%d %s correct=%s elapsed=%.1fs ETA=%.1fs', len(records), total,
                condition, record['correct'], elapsed, elapsed / len(records) * (total-len(records)))
    index.close()
    summary = {'tasks': len(tasks), 'adapter': str(args.adapter) if args.adapter else None,
        'task_sha256': hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        'protocol': 'FinQA execution equality; source-isolated development subset',
        'results': {condition: {
            'correct': sum(r['correct'] for r in records if r['condition']==condition),
            'total': len(tasks),
            'correct_with_evidence': sum(r['correct'] and r['correct_evidence'] for r in records if r['condition']==condition),
            'errors': sum('error' in r for r in records if r['condition']==condition),
            'latency_s': sum(r.get('latency_s',0) for r in records if r['condition']==condition),
        } for condition in conditions},
        'config': config, 'elapsed_s': time.monotonic()-started,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if driver is not None:
        summary.update(backend='vllm', served_model=args.served_model,
                       server_url=args.server_url, lora_runtime_dtype='bfloat16')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
