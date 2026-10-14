"""BF16 LoRA training, checkpoint resume and reload verification.

Runs on one GPU, or on several when started through ``torch.distributed.run``;
see ``src/training/distributed.py`` for why the two are the same training procedure.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import time

import torch
import yaml

from src.training import distributed
from src.training.chat_encoding import encode_messages

# peft initialises CUDA when imported, so a multi-GPU process is confined to its own
# GPU before the remaining imports.
distributed.isolate_gpu()

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForMultimodalLM, AutoTokenizer  # noqa: E402

from src.training.loss import assistant_loss  # noqa: E402

LOG = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def training_issuers(splits: Path) -> set[str]:
    return {row['code'] for row in json.loads(splits.read_text())['training']}


def review_issuer(review_id: str) -> str:
    """``review-<model>_<condition>-<code>-<announcement>`` names its issuer."""
    parts = review_id.split('-')
    if len(parts) < 4 or parts[0] != 'review':
        raise ValueError(f'not a review id: {review_id}')
    return parts[2]


def prepare_samples(config: dict, tokenizer) -> tuple[list[dict], dict]:
    """Chat samples ``{review_id, messages}``; only the final assistant turn is supervised.

    Every sample must come from a training-split issuer, so development and
    acceptance reports never reach the weights.
    """
    path = Path(config['chat_samples'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != config['samples_sha256']:
        raise ValueError('prepared supervision changed')
    allowed = training_issuers(Path(config['issuer_splits']))
    samples = []
    for row in read_jsonl(path):
        if review_issuer(row['review_id']) not in allowed:
            raise ValueError(f"sample outside the training split: {row['review_id']}")
        encoded = encode_messages(tokenizer, row['messages'], None, final_only=True)
        if len(encoded['input_ids']) > config['max_sequence_length']:
            raise ValueError(f"sample longer than max_sequence_length: {row['review_id']}")
        samples.append({'task_id': row['review_id'], **encoded})
    if len(samples) != config['samples'] or len({s['task_id'] for s in samples}) != len(samples):
        raise ValueError('prepared sample count mismatch or duplicate samples')
    # The longest sample goes first so the first update meets the memory peak;
    # the rest are in a fixed shuffled order.
    longest = max(samples, key=lambda s: len(s['input_ids']))
    rest = [s for s in samples if s is not longest]
    random.Random(config['seed']).shuffle(rest)
    samples = [longest] + rest
    lengths = [len(s['input_ids']) for s in samples]
    return samples, {'selected': len(samples), 'max_tokens': max(lengths),
                     'prepared_samples': str(path)}


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
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--dump-first-gradient', type=Path,
                        help='diagnostic: save the summed gradient of the first update of this run')
    parser.add_argument('--fsdp-frozen-weights', action='store_true',
                        help='multi-GPU only: shard the frozen base weights with FSDP; adapters stay replicated')
    parser.add_argument('--checkpointing-min-tokens', type=int, default=0,
                        help='recompute activations only for samples at least this long; shorter ones keep '
                             'them in memory, which needs the room FSDP sharding frees')
    parser.add_argument('--offload-unused-vision', action='store_true',
                        help='keep the vision tower on CPU; text-only supervision never calls it')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = json.loads(Path(args.config).read_text())
    if config.get('expandable_segments', False):
        if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') != 'expandable_segments:True':
            raise ValueError('training requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True')
    if config.get('deterministic_algorithms', False):
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.use_deterministic_algorithms(True)
    model_config = yaml.safe_load(Path(config['model_config']).read_text())['model']
    rank, world = distributed.setup()
    positions = distributed.local_positions(config['gradient_accumulation'], rank, world)
    sharded = args.fsdp_frozen_weights
    if sharded and (world == 1 or args.offload_unused_vision):
        raise ValueError('FSDP weight sharding needs several processes and manages the vision tower itself')
    if world > 1:
        if args.verify_only and not args.fsdp_frozen_weights:
            raise ValueError('verify replicated-weight checkpoints on one process')
        LOG.info('process %d of %d on physical GPU %s computes update positions %s',
                 rank, world, os.environ['CUDA_VISIBLE_DEVICES'], positions)
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
    if rank == 0:
        (output / 'selection.json').write_text(json.dumps({**selection,
            'task_ids': [s['task_id'] for s in samples], 'fingerprint': fingerprint}, indent=2))
    base = AutoModelForMultimodalLM.from_pretrained(model_config['path'],
        dtype=torch.bfloat16, device_map={'': model_config['device']}, local_files_only=True)
    if args.offload_unused_vision:
        # Placement only, so it is a run option rather than part of the checkpoint
        # fingerprint. The first-update probe runs before optimizer state exists and
        # therefore understates the memory later long samples need.
        def refuse(*_):
            raise RuntimeError('vision tower was called during text-only training')

        base.model.visual.register_forward_pre_hook(refuse)
        base.model.visual.to('cpu')
        LOG.info('vision tower kept on CPU: %.1f MiB',
                 sum(p.numel() * p.element_size() for p in base.model.visual.parameters()) / 2**20)
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
    if rank == 0:
        (output / 'trainable.json').write_text(json.dumps({
            'parameters': sum(p.numel() for _, p in trainable),
            'names': [name for name, _ in trainable],
            'dtypes': sorted({str(p.dtype) for _, p in trainable}),
            'model': model_config, 'gpu': torch.cuda.get_device_name(),
        }, indent=2))
    if sharded:
        units = distributed.shard_frozen_weights(base, [p for _, p in trainable])
        # Return the memory of the full weights to the driver. Left cached, it leaves no room
        # for cuBLAS to create its handle when the first backward pass is a long sample.
        torch.cuda.empty_cache()
        LOG.info('process %d: frozen weights sharded into %d FSDP units; resident %.0f MiB',
                 rank, len(units), torch.cuda.memory_allocated() / 2**20)
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
        # A checkpoint may be resumed on a machine with a different number of GPUs.
        torch.cuda.set_rng_state_all(state['rng_cuda'][:torch.cuda.device_count()])
        random.setstate(state['rng_python'])
        step = state['step']
    device = model_config['device']
    embedding = model.get_input_embeddings()
    # Sharded weights are placed by FSDP; the single-GPU memory offloads do not apply.
    embedding_offload = config.get('frozen_embedding_offload', False) and not sharded
    if embedding_offload:
        if (embedding.weight.requires_grad or embedding.weight.untyped_storage().data_ptr()
                == model.get_output_embeddings().weight.untyped_storage().data_ptr()):
            raise ValueError('embedding offload requires frozen, untied input weights')

    def needs_offload(index: int) -> bool:
        """Memory offloads cost time; ``offload_min_tokens`` limits them to long samples.

        Both offloads were verified to leave loss and gradients unchanged, so applying
        them per sample does not change the training result. Absent key: always offload.
        """
        sample = samples[index % len(samples)]
        tokens = len(sample['input_ids'])
        return not sharded and tokens >= config.get('offload_min_tokens', 0)

    def batch(index: int) -> dict:
        sample = samples[index % len(samples)]
        ids = torch.tensor([sample['input_ids']], device=device)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                'labels': torch.tensor([sample['labels']], device=device)}

    def compute_loss(index: int):
        inputs = batch(index)
        if args.checkpointing_min_tokens and model.training:
            # Recomputation changes time and memory only, not the result.
            if inputs['input_ids'].shape[1] >= args.checkpointing_min_tokens:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            else:
                model.gradient_checkpointing_disable()
        if embedding_offload:
            embedding.to(device)
        loss = (assistant_loss(model, inputs) if config.get('selective_fp32_loss', False)
                else model(**inputs).loss)
        if embedding_offload and needs_offload(index):
            # Decoder checkpoints start after embedding lookup. Its frozen,
            # untied weights are unused during backward; computations stay on GPU.
            embedding.to('cpu')
        return loss

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
        if rank == 0:
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
        failure = None
        for micro in positions:
            sample_index = step * config['gradient_accumulation'] + micro
            # Keep dense projection and saved tensor formats. The pinned-memory
            # path makes transposes contiguous and changes BF16 backward numerics.
            offload = (torch.autograd.graph.save_on_cpu(pin_memory=False)
                       if config.get('activation_cpu_offload', False) and needs_offload(sample_index)
                       else nullcontext())
            with offload:
                loss = compute_loss(sample_index)
            if not torch.isfinite(loss):
                sample = samples[sample_index % len(samples)]
                failure = {'step': step, 'micro': micro, 'task_id': sample['task_id'],
                           'tokens': len(sample['input_ids']), 'error': 'nonfinite loss'}
                if sharded:
                    # Peers are inside FSDP collectives; stop now and let the launcher end the group.
                    (output / f'failure_rank{rank}.json').write_text(json.dumps(failure, indent=2))
                    raise RuntimeError(str(failure))
                break
            (loss / config['gradient_accumulation']).backward()
            losses.append(float(loss.detach()))
        # Every process reports before any raises, so no peer waits on a dead collective.
        if distributed.reduce_scalar(float(failure is not None), world, 'max'):
            if failure:
                name = 'failure.json' if rank == 0 else f'failure_rank{rank}.json'
                (output / name).write_text(json.dumps(failure, indent=2))
            raise RuntimeError(str(failure or 'nonfinite loss on another process'))
        distributed.sync_gradients([p for _, p in trainable], world)
        if args.dump_first_gradient and rank == 0 and step == start_step:
            torch.save(torch.cat([p.grad.reshape(-1).cpu() for _, p in trainable]), args.dump_first_gradient)
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError('nonfinite gradient norm')
        optimizer.step()
        scheduler.step()
        step += 1
        elapsed = time.monotonic() - started
        mean_loss = distributed.reduce_scalar(sum(losses), world) / config['gradient_accumulation']
        peak = distributed.reduce_scalar(torch.cuda.max_memory_allocated() / 2**20, world, 'max')
        if rank:
            continue
        record = {'step': step, 'total': config['updates'], 'loss': mean_loss,
            'elapsed_s': elapsed, 'eta_s': elapsed / (step - start_step) * (stop - step),
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
        if world > 1:
            record.update(world_size=world, peak_allocated_mib_any_process=peak)
        if sharded:
            record['frozen_weights'] = 'fsdp'
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        LOG.info('%s checkpoint=%s', record, output)
    if not any(not torch.equal(a, p) for a, (_, p) in zip(initial, trainable)):
        raise RuntimeError('adapter parameters did not change')
    distributed.require_identical([p for _, p in trainable], world)
    if rank and not sharded:
        return 0
    model.eval()
    with torch.inference_mode():
        # With sharded weights every process must run this forward pass together.
        reference_loss = float(compute_loss(0))
    if rank:
        return 0
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
