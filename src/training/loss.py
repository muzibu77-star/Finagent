"""Equivalent single-example causal loss without upcasting masked vocabulary logits."""

import torch
from torch.nn import functional as F


def assistant_loss(model, batch: dict) -> torch.Tensor:
    """Mean FP32 CE over label[t+1] != -100, preserving full decoder input.

    Batch size is explicitly one. The selected positions are output positions,
    not input truncation; labels at position zero have no causal predecessor.
    """
    labels = batch['labels']
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError('selective loss requires micro batch one')
    positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError('no supervised causal targets')
    inputs = {key: value for key, value in batch.items() if key != 'labels'}
    # Preserve full BF16 projection/backward GEMM shape to avoid rounding drift.
    logits = model(**inputs, logits_to_keep=0).logits
    targets = labels[0, positions + 1]
    return F.cross_entropy(logits[0, positions].float(), targets, reduction='mean')
