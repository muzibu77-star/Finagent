"""Execution-scored preference pairs, gated by the completed full SFT baseline."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import time

import torch
from transformers import AutoTokenizer
import yaml

from src.model.agent import ModelDriver
from src.model.financial import TOOLS, decode_prediction, messages_for
from src.model.http_driver import HttpDriver
from src.training.preference_loss import sequence_log_probability
from src.training.prepare_sft import encode_messages


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines()]


def require_full_baseline(root: Path) -> dict:
    """Do not collect preferences or optimize before audited P0-2 completion."""
    policy = json.loads(Path('configs/jd_evaluation_protocol.json').read_text())
    proof = {}
    for arm in ('action', 'answer'):
        path = root / arm / 'verified.json'
        if not path.exists() or not json.loads(path.read_text())['passed']:
            raise ValueError('P0-2 training verification is incomplete')
        selected = json.loads((root / arm / 'selection.json').read_text())
        if selected['selected'] < 2000:
            raise ValueError('P0-2 expanded training baseline is too small')
        proof[str(path)] = digest(path)
    decision_path = root / 'selection/decision.json'
    if not decision_path.exists():
        raise ValueError('P0-2 development selection is incomplete')
    decision = json.loads(decision_path.read_text())
    if decision['policy'] != policy:
        raise ValueError('P0-2 development policy changed')
    for model in policy['models']:
        for split, calls in [('dev', 1192), ('test', 3913)]:
            run = root / f'{split}_{model}'
            path = run / 'validation.json'
            if not path.exists():
                raise ValueError('P0-2 full baseline audit is incomplete')
            validation = json.loads(path.read_text())
            settings = json.loads((run / 'settings.json').read_text())
            summary = json.loads((run / 'summary.json').read_text())
            if (not validation['passed'] or validation['records_sha256'] != digest(run / 'records.jsonl')
                    or settings['split'] != split or settings['evaluation_policy'] != policy
                    or summary['calls'] != calls):
                raise ValueError('P0-2 baseline integrity check failed')
            proof[str(path)] = digest(path)
    name = decision['selected']
    adapters = {'historical_action': Path('artifacts/m2_action_v2/step_0040'),
                'expanded_action': root / 'action/step_0251',
                'expanded_answer': root / 'answer/step_0251'}
    if name not in adapters:
        raise ValueError('unrecognized baseline model')
    adapter = adapters[name]
    return {'model': name, 'adapter': str(adapter), 'baseline_audits': proof,
            'decision_sha256': digest(decision_path), 'adapter_sha256': {
                p.name: digest(p) for p in (adapter / 'adapter_config.json', adapter / 'adapter_model.safetensors')}}


def execution_reward(generation: dict, facts: dict, expected) -> tuple[int, dict]:
    decoded = decode_prediction(dict(generation), facts, 'calculator')
    reward = int('error' not in decoded and decoded.get('prediction') == expected
                 and bool(decoded.get('calculation', {}).get('facts')))
    return reward, decoded


def collect(args) -> None:
    baseline = require_full_baseline(args.baseline_root)
    policy = json.loads(Path('configs/jd_dpo_protocol.json').read_text())
    root = Path('data/staged/m0_frozen_v1')
    receipt = json.loads((root / 'validation.json').read_text())
    if not receipt['passed'] or any(digest(root / f'{name}.jsonl') != value
                                   for name, value in receipt['sha256'].items()):
        raise ValueError('frozen training sources changed')
    allowed = {r['task_id']: r for r in read_rows(root / 'manifest.jsonl')
               if r['original_split'] == 'train' and r['official_eligible'] and r['dataset'] == 'finqa'}
    evidence = {r['evidence_id']: r for r in read_rows(root / 'evidence.jsonl')}
    questions = {r['task_id']: r['question'] for r in read_rows(root / 'questions.jsonl')}
    gold = {r['task_id']: r['gold']['exe_ans'] for r in read_rows(root / 'gold.jsonl')
            if r['task_id'] in allowed}
    trace_path = Path('data/staged/jd_sft_crop_3072_v1/traces.jsonl')
    traces = read_rows(trace_path)
    if len({r['task_id'] for r in traces}) != len(traces) or not {r['task_id'] for r in traces} <= allowed.keys():
        raise ValueError('preference source is duplicated or outside allowed training')
    random.Random(policy['seed']).shuffle(traces)
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    settings = {'baseline': baseline, 'policy': policy, 'source_receipt': receipt,
                'traces_sha256': digest(trace_path), 'source_sha256': digest(Path(__file__)),
                'driver_sha256': digest(Path('src/model/http_driver.py')), 'server_url': args.server_url}
    settings_path = args.output_dir / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('preference collection resume settings differ')
    settings_path.write_text(json.dumps(settings, indent=2))
    records_path = args.output_dir / 'records.jsonl'
    rows = read_rows(records_path) if records_path.exists() else []
    done = {row['task_id'] for row in rows}
    selected, excluded = [], []
    with HttpDriver(args.server_url, baseline['model']) as driver:
        for trace in traces:
            key = trace['task_id']
            source = allowed[key]
            if trace['source_report_id'] != source['source_report_id'] or trace['evidence_id'] != source['evidence_id']:
                raise ValueError('preference source metadata mismatch')
            context = evidence[source['evidence_id']]
            messages, facts = messages_for(context, questions[key], 'calculator')
            tokens = driver.tokenizer.apply_chat_template(messages, tools=TOOLS,
                add_generation_prompt=True, enable_thinking=False, tokenize=True,
                return_dict=False)
            if len(tokens) > policy['max_prompt_tokens']:
                excluded.append({'task_id': key, 'reason': 'complete prompt exceeds preference budget'})
                continue
            visible = json.loads(messages[1]['content'])['facts']
            selected.append((trace, messages, {key: facts[key] for key in visible}))
            if len(selected) == policy['candidate_tasks']:
                break
        (args.output_dir / 'selection.json').write_text(json.dumps({
            'tasks': [r[0]['task_id'] for r in selected], 'excluded': excluded}, indent=2))
        if not done <= {r[0]['task_id'] for r in selected} or len(done) != len(rows):
            raise ValueError('invalid resumed preference membership')
        started = time.monotonic()
        with ThreadPoolExecutor(policy['draws']) as pool:
            for trace, messages, facts in selected:
                key = trace['task_id']
                if key in done:
                    continue
                def sample(seed):
                    return driver.generate(messages, TOOLS, policy['max_output_tokens'],
                        policy['max_prompt_tokens'], temperature=policy['temperature'], seed=seed)
                generations = list(pool.map(sample, range(policy['seed'], policy['seed'] + policy['draws'])))
                candidates = []
                for generation in generations:
                    reward, decoded = execution_reward(generation, facts, gold[key])
                    candidates.append({'generation': generation, 'reward': reward,
                                       'execution': decoded})
                row = {'task_id': key, 'source_report_id': trace['source_report_id'],
                       'messages': messages, 'visible_facts': facts, 'expected': gold[key],
                       'candidates': candidates}
                with records_path.open('a') as handle:
                    handle.write(json.dumps(row) + '\n')
                rows.append(row)
                logging.info('%d/%d preference tasks elapsed=%.1fs records=%s',
                             len(rows), len(selected), time.monotonic() - started, records_path)
    (args.output_dir / 'summary.json').write_text(json.dumps({
        'selected': len(selected), 'completed': len(rows), 'generation_calls': len(rows) * policy['draws'],
        'reward_counts': dict(Counter(c['reward'] for row in rows for c in row['candidates']))}, indent=2))


def prepare(args) -> None:
    baseline = require_full_baseline(args.baseline_root)
    settings = json.loads((args.run / 'settings.json').read_text())
    if settings['baseline'] != baseline:
        raise ValueError('preference baseline changed')
    policy = settings['policy']
    selection = json.loads((args.run / 'selection.json').read_text())
    rows = read_rows(args.run / 'records.jsonl')
    if len(rows) != len(selection['tasks']) or {r['task_id'] for r in rows} != set(selection['tasks']):
        raise ValueError('preference collection is incomplete or duplicated')
    source = Path('data/staged/m0_frozen_v1')
    receipt = settings['source_receipt']
    if not receipt['passed'] or any(digest(source / f'{name}.jsonl') != value
                                   for name, value in receipt['sha256'].items()):
        raise ValueError('preference scoring source changed')
    allowed = {r['task_id']: r for r in read_rows(source / 'manifest.jsonl')
               if r['original_split'] == 'train' and r['official_eligible'] and r['dataset'] == 'finqa'}
    evidence = {r['evidence_id']: r for r in read_rows(source / 'evidence.jsonl')}
    questions = {r['task_id']: r['question'] for r in read_rows(source / 'questions.jsonl')}
    gold = {r['task_id']: r['gold'] for r in read_rows(source / 'gold.jsonl')}
    config = yaml.safe_load(Path('configs/m0_inference.yaml').read_text())
    tokenizer = AutoTokenizer.from_pretrained(config['model']['path'], local_files_only=True)
    pairs, excluded = [], []
    for row in rows:
        key = row['task_id']
        if key not in allowed or row['source_report_id'] != allowed[key]['source_report_id']:
            raise ValueError('preference task is outside its allowed source')
        context = evidence[allowed[key]['evidence_id']]
        messages, facts = messages_for(context, questions[key], 'calculator')
        if (row['expected'] != gold[key]['exe_ans']
                or row['visible_facts'] != facts
                or row['messages'] != messages):
            raise ValueError('preference prompt, facts or label differ from the frozen source')
        winners, losers = [], []
        for candidate in row['candidates']:
            reward, _ = execution_reward(candidate['generation'], row['visible_facts'], row['expected'])
            if reward != candidate['reward']:
                raise ValueError('saved preference reward differs from tool execution')
            if candidate['generation'].get('stopped_on_eos') and 'error' not in candidate['generation']:
                (winners if reward else losers).append(candidate['generation']['text'])
        if not winners or not losers:
            excluded.append({'task_id': row['task_id'], 'reason': 'no complete unequal-reward pair'})
            continue
        encoded = {name: encode_messages(tokenizer, row['messages'] + [
            {'role': 'assistant', 'content': text}], TOOLS)
            for name, text in [('chosen', winners[0]), ('rejected', losers[0])]}
        if any(len(value['input_ids']) > policy['max_sequence_length'] for value in encoded.values()):
            excluded.append({'task_id': row['task_id'], 'reason': 'complete pair exceeds length limit'})
            continue
        prefixes = [value['input_ids'][:next(i for i, label in enumerate(value['labels'])
                    if label != -100)] for value in encoded.values()]
        if not prefixes[0] or prefixes[0] != prefixes[1]:
            excluded.append({'task_id': row['task_id'], 'reason': 'token boundary changes shared prompt'})
            continue
        pairs.append({'task_id': row['task_id'], 'source_report_id': row['source_report_id'],
                      'chosen_reward': 1, 'rejected_reward': 0, **encoded})
    pairs.sort(key=lambda row: sum(len(row[side]['input_ids']) for side in ('chosen', 'rejected')), reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'pairs.json').write_text(json.dumps(pairs))
    (args.output_dir / 'excluded.json').write_text(json.dumps(excluded, indent=2))
    (args.output_dir / 'settings.json').write_text(json.dumps({
        'collection': settings, 'records_sha256': digest(args.run / 'records.jsonl'),
        'pairs_sha256': digest(args.output_dir / 'pairs.json')}, indent=2))
    if not pairs:
        raise ValueError('no execution-ranked preference pairs; exclusions saved')


def reference(args) -> None:
    baseline = require_full_baseline(args.baseline_root)
    settings = json.loads((args.run / 'settings.json').read_text())
    if (settings['collection']['baseline'] != baseline
            or settings['pairs_sha256'] != digest(args.run / 'pairs.json')):
        raise ValueError('preference inputs or reference initialization changed')
    settings['reference_source_sha256'] = {name: digest(Path(name)) for name in
        ('src/training/preference_loss.py', 'src/training/loss.py', __file__, 'src/model/agent.py')}
    settings['reference_precision'] = {'base': 'bfloat16', 'lora': 'float32'}
    pairs = json.loads((args.run / 'pairs.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    settings_path = args.output_dir / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('reference cache resume mismatch')
    settings_path.write_text(json.dumps(settings, indent=2))
    path = args.output_dir / 'records.jsonl'
    cached = read_rows(path) if path.exists() else []
    if [r['task_id'] for r in cached] != [r['task_id'] for r in pairs[:len(cached)]]:
        raise ValueError('reference cache membership changed')
    started = time.monotonic()
    with ModelDriver(baseline['adapter']) as driver:
        if any(p.dtype != torch.float32 for name, p in driver.model.named_parameters() if 'lora_' in name):
            raise ValueError('reference LoRA precision differs from the training contract')
        driver.model.config.use_cache = False
        with torch.inference_mode():
            for pair in pairs[len(cached):]:
                record = dict(pair)
                for side in ('chosen', 'rejected'):
                    sample = pair[side]
                    ids = torch.tensor([sample['input_ids']], device=driver.config['model']['device'])
                    batch = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                             'labels': torch.tensor([sample['labels']], device=ids.device)}
                    value = float(sequence_log_probability(driver.model, batch))
                    if not math.isfinite(value):
                        raise ValueError('nonfinite reference probability')
                    record[f'reference_{side}_logp'] = value
                with path.open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                cached.append(record)
                logging.info('%d/%d reference pairs elapsed=%.1fs cache=%s',
                             len(cached), len(pairs), time.monotonic() - started, path)
    (args.output_dir / 'pairs.json').write_text(json.dumps(cached))
    policy = settings['collection']['policy']
    config = json.loads(Path('configs/jd_action_3072_training.json').read_text())
    config.update(prepared_samples=str(args.output_dir / 'pairs.json'),
                  samples_sha256=digest(args.output_dir / 'pairs.json'), samples=len(cached),
                  updates=math.ceil(len(cached) / config['gradient_accumulation']),
                  max_sequence_length=policy['max_sequence_length'], learning_rate=policy['learning_rate'],
                  training_objective='dpo', beta=policy['beta'], initial_adapter=baseline['adapter'],
                  initial_adapter_sha256=baseline['adapter_sha256'],
                  reference_adapter_sha256=baseline['adapter_sha256'], baseline_root=str(args.baseline_root))
    (args.output_dir / 'training.json').write_text(json.dumps(config, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('collect', 'prepare', 'reference'))
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--server-url', default='http://127.0.0.1:18081')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command != 'collect' and args.run is None:
        parser.error('--run is required for prepare/reference')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    {'collect': collect, 'prepare': prepare, 'reference': reference}[args.command](args)


if __name__ == '__main__':
    main()
