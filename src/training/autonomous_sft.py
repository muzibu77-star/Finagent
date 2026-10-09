"""Collect actual search/read/calculate/submit traces from training-only sources."""

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import logging
from pathlib import Path
import random
import time

from transformers import AutoTokenizer
import yaml

from src.agent.runner import Runner, SCHEMAS, TOOLS
from src.data.corpus import Corpus
from src.evaluation.m0_inference_check import parse_tool_calls
from src.evidence.identity import evidence_identity
from src.harness.store import Store
from src.model.agent import ModelDriver
from src.model.http_driver import HttpDriver
from src.training.prepare_sft import encode_messages


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_training_parents(samples: list[dict], manifest: list[dict]) -> None:
    """Derived action IDs must resolve to eligible train questions and reports."""
    allowed = {r['task_id']: r['source_report_id'] for r in manifest
               if r['dataset'] == 'finqa' and r['original_split'] == 'train'
               and r['official_eligible']}
    ids = [sample['task_id'] for sample in samples]
    if not samples or len(ids) != len(set(ids)):
        raise ValueError('autonomous supervision must have unique action IDs')
    for sample in samples:
        parent = sample.get('parent_task_id')
        prefix = f'{parent}:autonomous:'
        suffix = sample['task_id'].removeprefix(prefix)
        if (parent not in allowed or sample.get('source_report_id') != allowed[parent]
                or not sample['task_id'].startswith(prefix) or not suffix.isdecimal()
                or str(int(suffix)) != suffix):
            raise ValueError('autonomous supervision has an invalid training parent or source')


def validate_lock(root: Path) -> dict:
    lock = json.loads((root / 'lock.json').read_text())
    for name, digest in lock['sha256'].items():
        if hashlib.sha256((root / f'{name}.json').read_bytes()).hexdigest() != digest:
            raise ValueError('autonomous training inputs changed')
    return lock


def freeze(output: Path, count: int) -> None:
    root = Path('data/staged/m0_frozen_v1')
    receipt = json.loads((root / 'validation.json').read_text())
    if not receipt['passed']:
        raise ValueError('source validation failed')
    for name, digest in receipt['sha256'].items():
        if hashlib.sha256((root / f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise ValueError('frozen source changed')
    manifest = read_rows(root / 'manifest.jsonl')
    allowed = [r for r in manifest if r['dataset'] == 'finqa'
               and r['original_split'] == 'train' and r['official_eligible']]
    if not 1 <= count <= len(allowed):
        raise ValueError('candidate count outside allowed training pool')
    evidence = {r['evidence_id']: r for r in read_rows(root / 'evidence.jsonl')}
    questions = {r['task_id']: r['question'] for r in read_rows(root / 'questions.jsonl')}
    gold = {r['task_id']: r['gold'] for r in read_rows(root / 'gold.jsonl')}
    snapshot = 'autonomous-train-v1'
    corpus = {r['evidence_id']: {'evidence': evidence[r['evidence_id']],
        'source_report_id': r['source_report_id'], 'company': r['company'],
        'period': str(r['report_period'])} for r in allowed}
    random.Random(109).shuffle(allowed)
    tasks, labels = [], {}
    for row in allowed[:count]:
        key = row['task_id']
        # No target page ID or answer is provided to the model.
        tasks.append({'task_id': key, 'question': questions[key], 'company': row['company'],
                      'period': str(row['report_period']), 'snapshot_id': snapshot})
        labels[key] = {'expected': gold[key]['exe_ans'], 'evidence_id': row['evidence_id'],
                       'source_report_id': row['source_report_id']}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('tasks', tasks), ('gold', labels), ('corpus', corpus)]:
        (output / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2))
    lock = {'seed': 109, 'candidates': count, 'snapshot_id': snapshot,
            'protocol': 'actual autonomous calls; labels used after execution only',
            'source_receipt': receipt, 'source_reports': sorted({r['source_report_id'] for r in allowed}),
            'sha256': {name: hashlib.sha256((output / f'{name}.json').read_bytes()).hexdigest()
                       for name in ('tasks', 'gold', 'corpus')}}
    (output / 'lock.json').write_text(json.dumps(lock, indent=2))


def successful_trace(row: dict, label: dict, identities: dict) -> bool:
    """Require a correct answer and exact source content, not execution alone."""
    calculations = (row.get('report') or {}).get('calculations', [])
    if (row['execution'] != 'succeeded' or row['business'] != 'answered'
            or len(calculations) != 1 or calculations[0]['value'] != label['expected']):
        return False
    facts = calculations[0]['facts']
    return bool(facts) and {identities[f['evidence_id']] for f in facts.values()} == {
        identities[label['evidence_id']]}


def collect(args) -> None:
    lock = validate_lock(args.tasks_root)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=args.resume)
    settings = {'lock': lock, 'adapter': str(args.adapter) if args.adapter else None,
                'server_url': args.server_url, 'served_model': args.served_model,
                'max_calls': 10, 'seconds': 180, 'context_management': True,
                'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                    for p in (__file__, 'src/agent/runner.py', 'src/agent/context.py')},
                'adapter_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in args.adapter.glob('adapter*') if p.is_file()} if args.adapter else {}}
    settings_path = output / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('trace resume settings mismatch')
    settings_path.write_text(json.dumps(settings, indent=2))
    tasks = json.loads((args.tasks_root / 'tasks.json').read_text())
    labels = json.loads((args.tasks_root / 'gold.json').read_text())
    corpus = Corpus(str(args.tasks_root / 'corpus.json'), lock['snapshot_id'])
    identities = {k: evidence_identity(v['evidence'], v['source_report_id'])
                  for k, v in corpus.documents.items()}
    store = Store(str(output / 'tasks.sqlite'))
    records_path, calls_path = output / 'records.jsonl', output / 'calls.jsonl'
    records = read_rows(records_path) if records_path.exists() else []
    done = {r['task_id'] for r in records}
    started = time.monotonic()
    driver_context = (HttpDriver(args.server_url, args.served_model) if args.server_url else
                      ModelDriver(str(args.adapter) if args.adapter else None))
    with driver_context as driver:
        store.recover()
        for task in tasks:
            key = task['task_id']
            if key in done:
                continue
            payload = {k: v for k, v in task.items() if k != 'task_id'}
            receipt = store.submit(key, payload)

            def captured(messages, tools, cancelled, deadline):
                visible = copy.deepcopy(messages)
                result = driver(messages, tools, cancelled, deadline)
                with calls_path.open('a') as handle:
                    handle.write(json.dumps({'task_id': key, 'messages': visible,
                                             'output': result}, ensure_ascii=False) + '\n')
                return result

            # Local driver OOM recovery retains the existing Runner contract.
            if hasattr(driver, 'health_check'):
                captured.health_check = driver.health_check
            row = Runner(store, corpus, captured, context_management=True).run(receipt['task_id'])
            events = store.events(key)
            observations = [e['body'] for e in events if e['kind'] == 'tool_observation']
            names = {e['tool'] for e in observations if 'error' not in e['observation']}
            accepted = successful_trace(row, labels[key], identities) and {
                'search_documents', 'get_evidence', 'calculate'} <= names
            record = {'task_id': key, 'accepted': accepted, 'execution': row['execution'],
                      'business': row['business'], 'attempts': row['state'].get('attempts', 0),
                      'stop_detail': row['state'].get('stop_detail'), 'report': row['report'],
                      'source_report_id': labels[key]['source_report_id']}
            with records_path.open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            records.append(record)
            logging.info('%d/%d accepted=%d elapsed=%.1fs records=%s', len(records), len(tasks),
                         sum(r['accepted'] for r in records), time.monotonic()-started, records_path)
    corpus.close()
    (output / 'summary.json').write_text(json.dumps({'candidates': len(tasks),
        'completed': len(records), 'accepted': sum(r['accepted'] for r in records),
        'failed': sum(not r['accepted'] for r in records)}, indent=2))


def prepare(args) -> None:
    lock = validate_lock(args.tasks_root)
    collection_settings = json.loads((args.run / 'settings.json').read_text())
    if collection_settings['lock'] != lock:
        raise ValueError('collection belongs to another frozen training pool')
    records = read_rows(args.run / 'records.jsonl')
    tasks = json.loads((args.tasks_root / 'tasks.json').read_text())
    if (len(records) != lock['candidates'] or len({r['task_id'] for r in records}) != len(records)
            or {r['task_id'] for r in records} != {t['task_id'] for t in tasks}):
        raise ValueError('complete unique trajectory collection required')
    labels = json.loads((args.tasks_root / 'gold.json').read_text())
    documents = json.loads((args.tasks_root / 'corpus.json').read_text())
    identities = {k: evidence_identity(v['evidence'], v['source_report_id'])
                  for k, v in documents.items()}
    model = yaml.safe_load(Path('configs/m0_inference.yaml').read_text())['model']
    tokenizer = AutoTokenizer.from_pretrained(model['path'], local_files_only=True)
    calls = defaultdict(list)
    for call in read_rows(args.run / 'calls.jsonl'):
        calls[call['task_id']].append(call)
    store = Store(str(args.run / 'tasks.sqlite'))
    samples, excluded = [], []
    action_counts = Counter()
    for record in records:
        key = record['task_id']
        events = store.events(key)
        names = {e['body']['tool'] for e in events if e['kind'] == 'tool_observation'
                 and 'error' not in e['body']['observation']}
        verified = successful_trace(store.get(key), labels[key], identities) and {
            'search_documents', 'get_evidence', 'calculate'} <= names
        if record['accepted'] != verified or record['source_report_id'] != labels[key]['source_report_id']:
            raise ValueError('saved trajectory score differs from actual state and source labels')
        if not record['accepted']:
            excluded.append({'task_id': key, 'reason': 'trajectory not correct and source-supported'})
            continue
        finished = [e['body'] for e in events if e['kind'] == 'generation_finished']
        trace = calls[key]
        if finished != [c['output'] for c in trace]:
            excluded.append({'task_id': key, 'reason': 'capture and committed generation differ'})
            continue
        # Reject the whole parent if any tool call failed; never teach a rejected action.
        if any(e['kind'] == 'tool_rejected' or (e['kind'] == 'tool_observation'
               and 'error' in e['body']['observation']) for e in events):
            excluded.append({'task_id': key, 'reason': 'trajectory contains rejected actions'})
            continue
        parent_samples = []
        for step, call in enumerate(trace):
            parsed, errors = parse_tool_calls(call['output'].get('text', ''), SCHEMAS)
            if errors or len(parsed) != 1 or not call['output'].get('stopped_on_eos'):
                raise ValueError('accepted trajectory contains invalid generation')
            action = parsed[0]
            messages = call['messages'] + [{'role': 'assistant', 'tool_calls': [
                {'type': 'function', 'function': {'name': action['name'],
                                                 'arguments': action['arguments']}}]}]
            encoded = encode_messages(tokenizer, messages, TOOLS)
            if len(encoded['input_ids']) > args.max_tokens:
                excluded.append({'task_id': key, 'step': step, 'reason': 'full state exceeds token budget'})
                parent_samples = []
                break
            parent_samples.append({'task_id': f'{key}:autonomous:{step}',
                'parent_task_id': key, 'source_report_id': record['source_report_id'],
                'tool': action['name'], **encoded})
        samples.extend(parent_samples)
        action_counts.update(s['tool'] for s in parent_samples)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    samples.sort(key=lambda s: len(s['input_ids']), reverse=True)
    (args.output_dir / 'action.json').write_text(json.dumps(samples))
    (args.output_dir / 'excluded.json').write_text(json.dumps(excluded, indent=2))
    summary = {'candidates': len(records), 'accepted_trajectories': sum(r['accepted'] for r in records),
        'selected_parents': len({s['parent_task_id'] for s in samples}), 'samples': len(samples),
        'tools': dict(action_counts), 'max_tokens': max((len(s['input_ids']) for s in samples), default=0),
        'excluded': dict(Counter(r['reason'] for r in excluded)), 'lock': lock,
        'collection_settings': collection_settings,
        'sha256': hashlib.sha256((args.output_dir / 'action.json').read_bytes()).hexdigest(),
        'supervision': 'assistant actions conditioned on actual deployment context; no given gold evidence'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    if not samples:
        raise ValueError('no valid complete autonomous trajectories; exclusions saved')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freezing = sub.add_parser('freeze')
    freezing.add_argument('--output-dir', type=Path, required=True)
    freezing.add_argument('--count', type=int, default=200)
    collecting = sub.add_parser('collect')
    collecting.add_argument('--tasks-root', type=Path, required=True)
    collecting.add_argument('--output-dir', type=Path, required=True)
    collecting.add_argument('--adapter', type=Path)
    collecting.add_argument('--server-url')
    collecting.add_argument('--served-model', default='base')
    collecting.add_argument('--resume', action='store_true')
    preparing = sub.add_parser('prepare')
    preparing.add_argument('--tasks-root', type=Path, required=True)
    preparing.add_argument('--run', type=Path, required=True)
    preparing.add_argument('--output-dir', type=Path, required=True)
    preparing.add_argument('--max-tokens', type=int, default=3072)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.command == 'freeze':
        freeze(args.output_dir, args.count)
    elif args.command == 'collect':
        if args.adapter and args.server_url:
            parser.error('use a declared served model for server adapters')
        collect(args)
    else:
        prepare(args)


if __name__ == '__main__':
    main()
