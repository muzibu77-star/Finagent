"""LoRA SFT under DeepSpeed ZeRO-3, as an alternative multi-GPU backend.

Same data, fixed sample order, effective batch, AdamW, linear schedule, clipping
and loss as ``m0_lora_check``. DeepSpeed partitions the frozen and trainable
weights over the GPUs and owns gradient reduction, clipping and the optimizer
step:

    global_batch = micro_batch (1) x gradient_accumulation_steps (G / world) x world

Precision is left as loaded (BF16 base, FP32 adapters): DeepSpeed's own bf16
mode would cast the adapters and is not enabled. Checkpoints are DeepSpeed's,
plus an ordinary adapter export for serving and comparison.

DeepSpeed is not in the default environment; put its install directory on
PYTHONPATH (see memory-bank/environment.md).
"""

from __future__ import annotations

import argparse
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

# peft and deepspeed initialise CUDA when imported; confine this process to its GPU first.
distributed.isolate_gpu()
# DeepSpeed selects cuda:LOCAL_RANK; after isolation the only visible GPU is cuda:0.
os.environ['LOCAL_RANK'] = '0'

import deepspeed  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForMultimodalLM, AutoTokenizer  # noqa: E402

from src.training.loss import assistant_loss  # noqa: E402
from src.training.m0_lora_check import prepare_samples, validated_initial_adapter  # noqa: E402

LOG = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', help='tag of a DeepSpeed checkpoint under OUTPUT/deepspeed')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = json.loads(Path(args.config).read_text())
    if config.get('training_objective', 'sft') != 'sft' or not config.get('selective_fp32_loss'):
        raise ValueError('the DeepSpeed backend supports the selective-loss SFT recipe only')
    if config.get('deterministic_algorithms', False):
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.use_deterministic_algorithms(True)
    rank, world = distributed.setup()
    if world == 1:
        raise ValueError('start through torch.distributed.run with at least two processes')
    positions = distributed.local_positions(config['gradient_accumulation'], rank, world)
    model_config = yaml.safe_load(Path(config['model_config']).read_text())['model']
    initial_adapter = validated_initial_adapter(config, model_config)
    if not initial_adapter:
        raise ValueError('the DeepSpeed backend starts from an existing adapter')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(model_config['path'], local_files_only=True)
    samples, selection = prepare_samples(config, tokenizer)
    base = AutoModelForMultimodalLM.from_pretrained(model_config['path'], dtype=torch.bfloat16,
        device_map={'': model_config['device']}, local_files_only=True)
    model = PeftModel.from_pretrained(base, initial_adapter, is_trainable=True)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('visual' in name or 'lora_' not in name for name, _ in trainable):
        raise RuntimeError('unexpected trainable modules')
    dtypes = {str(p.dtype) for _, p in trainable}
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: max(0.0, 1 - step / config['updates']))
    accumulation = config['gradient_accumulation'] // world
    deepspeed_config = {
        'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': accumulation,
        'gradient_clipping': 1.0, 'zero_optimization': {'stage': 3},
        'zero_allow_untested_optimizer': True, 'steps_per_print': 10**9}
    engine, _, _, scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler, config=deepspeed_config)
    if {str(p.dtype) for _, p in trainable} != dtypes or next(base.parameters()).dtype != torch.bfloat16:
        raise RuntimeError('DeepSpeed changed parameter precision')
    step = 0
    if args.resume:
        # Checkpoints leave out the frozen base weights, so the module load cannot be strict.
        _, client = engine.load_checkpoint(str(output / 'deepspeed'), tag=args.resume,
                                           load_module_strict=False)
        step = client['step']
    if rank == 0:
        (output / 'selection.json').write_text(json.dumps({**selection, 'backend': 'deepspeed_zero3',
            'deepspeed': deepspeed.__version__, 'deepspeed_config': deepspeed_config,
            'task_ids': [s['task_id'] for s in samples]}, indent=2))
    device = model_config['device']

    def loss_for(index: int) -> torch.Tensor:
        sample = samples[index % len(samples)]
        ids = torch.tensor([sample['input_ids']], device=device)
        # The forward must go through the engine: that is where DeepSpeed advances its
        # micro-step counter. Calling the module directly makes every backward overwrite
        # the accumulated gradient, so only the last sample of each update would count.
        return assistant_loss(engine, {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                                       'labels': torch.tensor([sample['labels']], device=device)})

    stop = args.stop_after or config['updates']
    started, start_step = time.monotonic(), step
    engine.train()
    torch.cuda.reset_peak_memory_stats()
    while step < stop:
        losses = []
        for micro in positions:
            loss = loss_for(step * config['gradient_accumulation'] + micro)
            if not torch.isfinite(loss):
                # Peers are inside ZeRO collectives; stop now and let the launcher end the group.
                raise RuntimeError(f'nonfinite loss at update {step}, position {micro}')
            engine.backward(loss)  # scales by 1 / gradient_accumulation_steps
            engine.step()          # optimizer and schedule advance on the accumulation boundary
            losses.append(float(loss.detach()))
        step += 1
        mean_loss = distributed.reduce_scalar(sum(losses), world) / config['gradient_accumulation']
        peak = distributed.reduce_scalar(torch.cuda.max_memory_allocated() / 2**20, world, 'max')
        if rank:
            continue
        elapsed = time.monotonic() - started
        record = {'step': step, 'total': config['updates'], 'loss': mean_loss, 'elapsed_s': elapsed,
                  'eta_s': elapsed / (step - start_step) * (stop - step),
                  'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'world_size': world, 'peak_allocated_mib_any_process': peak,
                  'backend': 'deepspeed_zero3'}
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        LOG.info('%s checkpoint=%s', record, output)
    if not math.isfinite(mean_loss):
        raise RuntimeError('nonfinite training loss')
    tag = f'step_{step:04d}'
    engine.save_checkpoint(str(output / 'deepspeed'), tag=tag, client_state={'step': step},
                           exclude_frozen_parameters=True)
    # Materialise the partitioned adapter weights on every process, then export once.
    with deepspeed.zero.GatheredParameters([p for _, p in trainable], modifier_rank=None):
        if rank == 0:
            model.save_pretrained(output / tag)
    if rank == 0:
        (output / 'latest').write_text(str(output / tag) + '\n')
        LOG.info('saved %s', output / tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
