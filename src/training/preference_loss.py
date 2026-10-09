"""Standard sigmoid DPO with summed, assistant-only causal log probabilities.

Reference: https://arxiv.org/abs/2305.18290, equation 7.
The fixed reference probabilities are computed before optimization, never updated.
"""

import torch
from torch.nn import functional as F

from src.training.loss import assistant_loss


def sequence_log_probability(model, batch: dict) -> torch.Tensor:
    count = torch.count_nonzero(batch['labels'][:, 1:] != -100)
    return -assistant_loss(model, batch) * count


def dpo_loss(chosen: torch.Tensor, rejected: torch.Tensor,
             reference_chosen: float, reference_rejected: float,
             beta: float) -> torch.Tensor:
    if beta <= 0 or chosen.ndim != 0 or rejected.ndim != 0:
        raise ValueError('DPO requires positive beta and scalar sequence log probabilities')
    margin = (chosen - rejected) - (reference_chosen - reference_rejected)
    return -F.logsigmoid(beta * margin)
