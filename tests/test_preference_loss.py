import math
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from src.training.preference_loss import dpo_loss, sequence_log_probability


class PreferenceLossTests(unittest.TestCase):
    def test_initial_margin_and_gradient_direction(self):
        chosen = torch.tensor(-4.0, requires_grad=True)
        rejected = torch.tensor(-7.0, requires_grad=True)
        loss = dpo_loss(chosen, rejected, -4.0, -7.0, .1)
        self.assertAlmostEqual(float(loss.detach()), math.log(2), places=6)
        loss.backward()
        self.assertAlmostEqual(float(chosen.grad), -.05, places=6)
        self.assertAlmostEqual(float(rejected.grad), .05, places=6)
        self.assertLess(float(dpo_loss(chosen + 1, rejected, -4.0, -7.0, .1).detach()), math.log(2))

    def test_only_causal_assistant_tokens_contribute_to_sum_and_gradient(self):
        torch.manual_seed(7)
        logits = torch.randn(1, 4, 8, requires_grad=True)
        batch = {'input_ids': torch.tensor([[0, 1, 2, 3]]),
                 'attention_mask': torch.ones(1, 4, dtype=torch.long),
                 'labels': torch.tensor([[-100, -100, 2, 3]])}
        model = lambda **kwargs: SimpleNamespace(logits=logits)
        actual = sequence_log_probability(model, batch)
        expected = F.log_softmax(logits[0, 1:3], dim=-1)[torch.arange(2), torch.tensor([2, 3])].sum()
        self.assertTrue(torch.allclose(actual, expected))
        left = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        right = torch.autograd.grad(expected, logits)[0]
        self.assertTrue(torch.allclose(left, right))
        self.assertEqual(float(left[0, 0].abs().sum()), 0)
        self.assertEqual(float(left[0, 3].abs().sum()), 0)
