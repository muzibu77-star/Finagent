"""Single-GPU BF16 LoRA feasibility, checkpoint resume and reload verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import time

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from src.training.loss import assistant_loss

LOG = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare_samples(config: dict, tokenizer) -> tuple[list[dict], dict]:
    root = Path(config['data_dir'])
    summary = json.loads((root / 'summary.json').read_text())
    if not summary['split_protocol_frozen']:
        raise ValueError('training requires a frozen data protocol')
    receipt = json.loads((root / 'validation.json').read_text())
    if not receipt['passed']:
        raise ValueError('data validation did not pass')
    for name, expected in receipt['sha256'].items():
        if hashlib.sha256((root / f'{name}.jsonl').read_bytes()).hexdigest() != expected:
            raise ValueError(f'validated data changed: {name}')
    manifest = read_jsonl(root / 'manifest.jsonl')
    evidence = {r['evidence_id']: r for r in read_jsonl(root / 'evidence.jsonl')}
    questions = {r['task_id']: r for r in read_jsonl(root / 'questions.jsonl')}
    gold = {r['task_id']: r for r in read_jsonl(root / 'gold.jsonl')}
    if 'prepared_samples' in config:
        prepared = Path(config['prepared_samples'])
        if hashlib.sha256(prepared.read_bytes()).hexdigest() != config['samples_sha256']:
            raise ValueError('prepared supervision changed')
        samples = json.loads(prepared.read_text())
        allowed = {r['task_id'] for r in manifest if r['original_split'] == 'train'
                   and r['official_eligible']}
        source_reports = {r['task_id']: r['source_report_id'] for r in manifest}
        ids = [sample['task_id'] for sample in samples]
        if config.get('autonomous_supervision', False):
            from src.training.autonomous_sft import validate_training_parents
            validate_training_parents(samples, manifest)
        elif len(ids) != len(set(ids)) or not set(ids) <= allowed:
            raise ValueError('supervision contains duplicate or held-out tasks')
        for sample in samples:
            variants = [sample]
            if config.get('training_objective') == 'dpo':
                if (sample['chosen_reward'] != 1 or sample['rejected_reward'] != 0
                        or sample['source_report_id'] != source_reports[sample['task_id']]
                        or not all(math.isfinite(sample[key]) and sample[key] <= 0 for key in
                                   ('reference_chosen_logp', 'reference_rejected_logp'))
                        or config['reference_adapter_sha256'] != config['initial_adapter_sha256']):
                    raise ValueError('invalid execution preference or reference binding')
                variants = [sample['chosen'], sample['rejected']]
                prefixes = [v['input_ids'][:next((i for i, label in enumerate(v['labels'])
                            if label != -100), len(v['labels']))] for v in variants]
                if not prefixes[0] or prefixes[0] != prefixes[1]:
                    raise ValueError('preference completions must share exactly the same prompt')
            for variant in variants:
                tokens, labels = variant['input_ids'], variant['labels']
                if (not tokens or len(tokens) != len(labels)
                        or len(tokens) > config['max_sequence_length']
                        or not any(label != -100 for label in labels[1:])
                        or any(label not in (-100, token) for token, label in zip(tokens, labels))):
                    raise ValueError('invalid prepared token/mask contract')
        if len(samples) != config['samples']:
            raise ValueError('prepared sample count mismatch')
        lengths = ([len(s[side]['input_ids']) for s in samples for side in ('chosen', 'rejected')]
                   if config.get('training_objective') == 'dpo' else
                   [len(s['input_ids']) for s in samples])
        return samples, {'selected': len(samples),
                         'max_tokens': max(lengths),
                         'prepared_samples': str(prepared)}
    candidates = [r for r in manifest if r['original_split'] == 'train'
                  and r['official_eligible'] and gold[r['task_id']]['labels_available']]
    random.Random(config['seed']).shuffle(candidates)
    selected = []
    rejected = 0
    for row in candidates:
        task_id = row['task_id']
        question = questions[task_id]
        context = {k: v for k, v in evidence[row['evidence_id']].items()
                   if k != 'evidence_id'}
        user = json.dumps(context, ensure_ascii=False) + '\n' + question['question']
        answer = str(gold[task_id]['gold']['answer'])
        messages = [{'role': 'user', 'content': user}]
        prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                               enable_thinking=False, tokenize=False)
        full_text = tokenizer.apply_chat_template(messages + [
            {'role': 'assistant', 'content': answer}], enable_thinking=False,
            add_generation_prompt=False, tokenize=False)
        if not full_text.startswith(prefix):
            raise ValueError('assistant prefix mismatch')
        encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        full = encoded['input_ids']
        if len(full) > config['max_sequence_length']:
            rejected += 1
            continue
        # Offset boundaries remain valid when BPE merges across the prefix edge.
        labels = [token if start >= len(prefix) else -100
                  for token, (start, end) in zip(full, encoded['offset_mapping'])]
        if not any(x != -100 for x in labels):
            raise ValueError('empty assistant loss mask')
        selected.append({'task_id': task_id, 'input_ids': full, 'labels': labels})
    if len(selected) < config['samples']:
        raise ValueError('not enough complete eligible examples')
    # Include the longest admissible example to exercise the resource boundary.
    longest = max(selected, key=lambda row: len(row['input_ids']))
    samples = [longest] + [s for s in selected if s is not longest][:config['samples'] - 1]
    return samples, {'eligible': len(candidates), 'overlength_excluded': rejected,
                     'selected': len(samples), 'max_tokens': len(longest['input_ids'])}


def validated_initial_adapter(config: dict, model_config: dict) -> Path | None:
    """A new SFT phase may initialize from weights, with a fresh optimizer.

    This is distinct from --resume, which restores the complete phase state.
    The original adapter and its contract remain bound into the fingerprint.
    """
    if not config.get('initial_adapter'):
        return None
    path = Path(config['initial_adapter'])
    names = ('adapter_config.json', 'adapter_model.safetensors')
    digests = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in names}
    if digests != config.get('initial_adapter_sha256'):
        raise ValueError('initial adapter hashes differ')
    adapter = json.loads((path / 'adapter_config.json').read_text())
    expected = {'r': config['rank'], 'lora_alpha': 16, 'lora_dropout': 0.0,
                'target_modules': config['target_modules'], 'bias': 'none',
                'base_model_name_or_path': model_config['path'], 'peft_type': 'LORA',
                'rank_pattern': {}, 'alpha_pattern': {}, 'use_dora': False,
                'use_rslora': False, 'modules_to_save': None}
    if any(adapter.get(key) != value for key, value in expected.items()):
        raise ValueError('initial adapter is incompatible with the training model/config')
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/m0_training.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = json.loads(Path(args.config).read_text())
    if config.get('training_objective', 'sft') not in ('sft', 'dpo'):
        raise ValueError('unsupported training objective')
    if config.get('training_objective') == 'dpo':
        if not math.isfinite(config['beta']) or config['beta'] <= 0:
            raise ValueError('DPO beta must be finite and positive')
        from src.training.preference_data import require_full_baseline
        require_full_baseline(Path(config['baseline_root']))
    model_config = yaml.safe_load(Path(config['model_config']).read_text())['model']
    initial_adapter = validated_initial_adapter(config, model_config)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16-capable CUDA device required')
    tokenizer = AutoTokenizer.from_pretrained(model_config['path'], local_files_only=True)
    samples, selection = prepare_samples(config, tokenizer)
    fingerprint = hashlib.sha256(json.dumps(
        {'config': config, 'samples': samples}, sort_keys=True).encode()).hexdigest()
    (output / 'selection.json').write_text(json.dumps({**selection,
        'task_ids': [s['task_id'] for s in samples], 'fingerprint': fingerprint}, indent=2))
    base = AutoModelForMultimodalLM.from_pretrained(model_config['path'],
        dtype=torch.bfloat16, device_map={'': model_config['device']}, local_files_only=True)
    if args.resume:
        model = PeftModel.from_pretrained(base, args.resume, is_trainable=True)
    elif initial_adapter:
        model = PeftModel.from_pretrained(base, initial_adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=config['rank'], lora_alpha=16,
            lora_dropout=0.0, target_modules=config['target_modules'], bias='none'))
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('visual' in name or 'lora_' not in name for name, _ in trainable):
        raise RuntimeError('unexpected trainable modules')
    (output / 'trainable.json').write_text(json.dumps({
        'parameters': sum(p.numel() for _, p in trainable),
        'names': [name for name, _ in trainable],
        'dtypes': sorted({str(p.dtype) for _, p in trainable}),
        'model': model_config, 'gpu': torch.cuda.get_device_name(),
    }, indent=2))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: max(0.0, 1 - step / config['updates']))
    step = 0
    state = None
    if args.resume:
        state = torch.load(args.resume / 'state.pt', map_location='cpu', weights_only=False)
        if state['fingerprint'] != fingerprint:
            raise ValueError('checkpoint data/config mismatch')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng_cpu'])
        torch.cuda.set_rng_state_all(state['rng_cuda'])
        random.setstate(state['rng_python'])
        step = state['step']
    device = model_config['device']

    def batch(index: int, side: str | None = None) -> dict:
        sample = samples[index % len(samples)]
        if side is not None:
            sample = sample[side]
        ids = torch.tensor([sample['input_ids']], device=device)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                'labels': torch.tensor([sample['labels']], device=device)}

    def compute_loss(index: int):
        if config.get('training_objective') == 'dpo':
            from src.training.preference_loss import dpo_loss, sequence_log_probability
            sample = samples[index % len(samples)]
            chosen = sequence_log_probability(model, batch(index, 'chosen'))
            rejected = sequence_log_probability(model, batch(index, 'rejected'))
            return dpo_loss(chosen, rejected, sample['reference_chosen_logp'],
                            sample['reference_rejected_logp'], config['beta'])
        inputs = batch(index)
        return (assistant_loss(model, inputs) if config.get('selective_fp32_loss', False)
                else model(**inputs).loss)

    if args.verify_only:
        if state is None:
            raise ValueError('--verify-only requires --resume')
        model.eval()
        with torch.inference_mode():
            loss = float(compute_loss(0))
        if (not math.isfinite(loss) or not math.isfinite(state['reference_loss'])
                or abs(loss - state['reference_loss']) > 1e-5):
            raise RuntimeError('reload loss mismatch')
        if not optimizer.state or scheduler.last_epoch != step:
            raise RuntimeError('optimizer/scheduler resume mismatch')
        (output / 'verified.json').write_text(json.dumps({
            'step': step, 'reload_loss': loss, 'reference_loss': state['reference_loss'],
            'optimizer_restored': True, 'passed': step == config['updates']}, indent=2))
        return 0
    started = time.monotonic()
    start_step = step
    stop = args.stop_after or config['updates']
    model.train()
    torch.cuda.reset_peak_memory_stats()
    initial = [p.detach().clone() for _, p in trainable]
    while step < stop:
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(config['gradient_accumulation']):
            sample_index = step * config['gradient_accumulation'] + micro
            loss = compute_loss(sample_index)
            if not torch.isfinite(loss):
                sample = samples[sample_index % len(samples)]
                token_count = (sum(len(sample[side]['input_ids']) for side in ('chosen', 'rejected'))
                               if config.get('training_objective') == 'dpo' else len(sample['input_ids']))
                failure = {'step': step, 'micro': micro, 'task_id': sample['task_id'],
                           'tokens': token_count, 'error': 'nonfinite loss'}
                (output / 'failure.json').write_text(json.dumps(failure, indent=2))
                raise RuntimeError(str(failure))
            (loss / config['gradient_accumulation']).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError('nonfinite gradient norm')
        optimizer.step()
        scheduler.step()
        step += 1
        elapsed = time.monotonic() - started
        record = {'step': step, 'total': config['updates'], 'loss': sum(losses) / len(losses),
            'elapsed_s': elapsed, 'eta_s': elapsed / (step - start_step) * (stop - step),
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        LOG.info('%s checkpoint=%s', record, output)
    if not any(not torch.equal(a, p) for a, (_, p) in zip(initial, trainable)):
        raise RuntimeError('adapter parameters did not change')
    model.eval()
    with torch.inference_mode():
        reference_loss = float(compute_loss(0))
    if not math.isfinite(reference_loss):
        raise RuntimeError('nonfinite checkpoint verification loss')
    checkpoint = output / f'step_{step:04d}'
    checkpoint.mkdir(exist_ok=False)
    model.save_pretrained(checkpoint)
    torch.save({'step': step, 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'rng_cpu': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all(), 'rng_python': random.getstate(),
        'fingerprint': fingerprint, 'config': config, 'reference_loss': reference_loss,
        'sample_cursor': step * config['gradient_accumulation'],
        'sampler': 'fixed cyclic order recorded in selection.json'}, checkpoint / 'state.pt')
    (output / 'latest').write_text(str(checkpoint) + '\n')
    LOG.info('saved %s', checkpoint)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
