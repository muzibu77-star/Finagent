"""Multi-GPU gradient accumulation that keeps the single-GPU update definition.

One update still consumes the same ``samples_per_update`` consecutive samples
in the same fixed order. Each process handles the positions congruent to its
rank, the accumulated gradients are summed across processes once per update,
and every process then applies the identical optimizer step:

    global_batch = micro_batch (1) x (samples_per_update / world_size) x world_size

So a checkpoint, its data fingerprint and its sample cursor mean the same thing
for any world size; only floating-point summation order differs.
"""

import os

import torch
import torch.distributed as dist


def isolate_gpu() -> None:
    """Make only this process's GPU visible; call before anything initialises CUDA.

    Without this, seeding and RNG-state calls create a context of about 0.5 GiB on
    every GPU from every process, which is enough to push long samples out of memory.
    Afterwards the process addresses its GPU as ``cuda:0``.
    """
    if int(os.environ.get('WORLD_SIZE', '1')) == 1 or os.environ.get('FINAGENT_GPU_ISOLATED'):
        return
    if torch.cuda.is_initialized():
        raise RuntimeError('GPU isolation must happen before CUDA is initialised')
    local_rank = int(os.environ['LOCAL_RANK'])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = visible.split(',')[local_rank] if visible else str(local_rank)
    os.environ['FINAGENT_GPU_ISOLATED'] = '1'  # a second entry point importing this one must not redo it


def setup(backend: str | None = None) -> tuple[int, int]:
    """Join the process group described by the launcher; a plain run is (0, 1)."""
    if int(os.environ.get('WORLD_SIZE', '1')) == 1:
        return 0, 1
    if not dist.is_initialized():
        dist.init_process_group(backend or ('nccl' if torch.cuda.is_available() else 'gloo'))
    return dist.get_rank(), dist.get_world_size()


def local_positions(samples_per_update: int, rank: int, world_size: int) -> list[int]:
    """Positions inside one update that this process computes."""
    if not 0 <= rank < world_size:
        raise ValueError('rank outside the process group')
    if samples_per_update % world_size:
        raise ValueError(f'{samples_per_update} samples per update cannot be split evenly '
                         f'over {world_size} processes; the effective batch would change')
    return list(range(rank, samples_per_update, world_size))


def reduce_scalar(value: float, world_size: int, operation: str = 'sum') -> float:
    """Combine one number from every process; all processes must call it together."""
    if world_size == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.float64)
    if torch.cuda.is_available() and dist.get_backend() == 'nccl':
        tensor = tensor.cuda()
    dist.all_reduce(tensor, op={'sum': dist.ReduceOp.SUM, 'max': dist.ReduceOp.MAX,
                               'min': dist.ReduceOp.MIN}[operation])
    return float(tensor.item())


def sync_gradients(parameters: list[torch.nn.Parameter], world_size: int) -> None:
    """Replace every local accumulated gradient by the sum over all processes."""
    if world_size == 1:
        return
    if any(p.grad is None for p in parameters):
        raise RuntimeError('a process accumulated no gradient for a trainable parameter')
    flat = torch.cat([p.grad.reshape(-1) for p in parameters])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for parameter in parameters:
        count = parameter.grad.numel()
        parameter.grad.copy_(flat[offset:offset + count].view_as(parameter.grad))
        offset += count


def require_identical(parameters: list[torch.nn.Parameter], world_size: int) -> None:
    """Fail if the processes no longer hold the same trainable weights."""
    if world_size == 1:
        return
    checksum = float(sum(p.detach().double().sum() for p in parameters))
    magnitude = float(sum(p.detach().double().abs().sum() for p in parameters))
    for value in (checksum, magnitude):
        if reduce_scalar(value, world_size, 'max') != reduce_scalar(value, world_size, 'min'):
            raise RuntimeError('trainable weights diverged across processes')


def frozen_weight_units(base: torch.nn.Module, ignored: set[torch.nn.Parameter]) -> list[torch.nn.Module]:
    """Modules that each become one FSDP unit.

    Blocks of a ModuleList (decoder layers, vision blocks) are units; containers
    without parameters of their own are descended into; everything else is a unit.
    Units without any frozen parameter are dropped.
    """
    units = []

    def collect(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.ModuleList):
            units.extend(module)
            return
        children = [child for child in module.children() if any(True for _ in child.parameters())]
        if not children or any(p not in ignored for p in module.parameters(recurse=False)):
            units.append(module)
            return
        for child in children:
            collect(child)

    collect(base)
    units = [unit for unit in units if any(p not in ignored for p in unit.parameters())]
    covered = {p for unit in units for p in unit.parameters()}
    missing = sum(p.numel() for p in base.parameters() if p not in ignored and p not in covered)
    if missing:
        raise RuntimeError(f'{missing} frozen parameters are outside every FSDP unit')
    return units


def shard_frozen_weights(base: torch.nn.Module, trainable: list[torch.nn.Parameter]) -> list:
    """Shard the frozen base weights across processes with FSDP; keep adapters replicated.

    The frozen parameters of each unit are split over the processes and gathered only
    while that unit runs. The trainable adapter parameters are excluded, so they stay
    full copies on every process and keep using ``sync_gradients``: optimizer state,
    checkpoints and the update definition are the same as without sharding. Every
    forward and backward pass must then be executed by all processes together.
    """
    from torch.distributed.fsdp import fully_shard

    ignored = set(trainable)
    units = frozen_weight_units(base, ignored)
    for unit in units:
        fully_shard(unit, reshard_after_forward=True, ignored_params=ignored)
    return units
