"""Single-example causal loss without materialising full-sequence vocabulary logits."""

import torch
import torch.distributed as dist
from torch.nn import functional as F

from src.training.distributed import reduce_scalar

# Supervised positions per chunk of the vocabulary projection. Smaller chunks
# bound peak memory more tightly and cost more recomputation.
CHUNK_POSITIONS = 1024


def _chunk_cross_entropy_sum(hidden_chunk: torch.Tensor, targets_chunk: torch.Tensor,
                             head: torch.nn.Module) -> torch.Tensor:
    """FP32 cross-entropy sum of one chunk of supervised positions."""
    return F.cross_entropy(head(hidden_chunk).float(), targets_chunk, reduction='sum')


def agreed_chunk_count(positions: int) -> int:
    """Chunks this process runs, raised to the largest count in the process group.

    The output layer is frozen and FSDP-sharded, so every call to it all-gathers its
    weight; the processes hold different samples and must therefore issue the same
    number of calls. Shorter processes pad with empty chunks, which add nothing to
    the sum but keep both the forward and the recomputed backward collectives of the
    sharded head in step.
    """
    chunks = -(-positions // CHUNK_POSITIONS)
    if dist.is_available() and dist.is_initialized():
        chunks = int(reduce_scalar(float(chunks), dist.get_world_size(), 'max'))
    return chunks


def assistant_loss(model, batch: dict) -> torch.Tensor:
    """Mean FP32 CE over label[t+1] != -100, projecting the vocabulary head in chunks.

    Batch size is explicitly one. The selected positions are output positions,
    not input truncation; labels at position zero have no causal predecessor.

    The decoder's final normed hidden states, the tensor the full model feeds to
    its output head, are kept for the whole sequence; only the supervised rows are
    projected, one chunk at a time, and each chunk is wrapped in
    ``torch.utils.checkpoint`` so backward recomputes that chunk alone. The result
    equals upcasting the selected rows of the full-sequence logits, up to the
    order in which the per-position losses are summed and, under distributed
    training, up to the empty padding chunks.
    """
    labels = batch['labels']
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError('selective loss requires micro batch one')
    positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError('no supervised causal targets')
    inputs = {key: value for key, value in batch.items() if key != 'labels'}
    # The decoder submodule already applies the final norm; its head turns the
    # same states into logits in the untouched reference path.
    hidden = model.get_decoder()(**inputs, use_cache=False).last_hidden_state
    head = model.get_output_embeddings()
    targets = labels[0, positions + 1]
    total = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for index in range(agreed_chunk_count(positions.numel())):
        start = index * CHUNK_POSITIONS
        chunk = positions[start:start + CHUNK_POSITIONS]
        total = total + torch.utils.checkpoint.checkpoint(
            _chunk_cross_entropy_sum, hidden[0, chunk],
            targets[start:start + CHUNK_POSITIONS], head, use_reentrant=False)
    return total / positions.numel()
