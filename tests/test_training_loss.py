from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F
from src.training.loss import assistant_loss


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(13, 7)
        self.head = torch.nn.Linear(7, 13)

    def forward(self, input_ids, logits_to_keep):
        # Cumulative context makes masked input tokens influence later predictions.
        hidden = self.embed(input_ids).cumsum(1)
        indices = slice(None) if logits_to_keep == 0 else logits_to_keep
        return SimpleNamespace(logits=self.head(hidden[:, indices]))


class SelectiveLossTests(unittest.TestCase):
    def test_loss_and_all_parameter_gradients_match(self):
        torch.manual_seed(9)
        model = TinyModel()
        ids = torch.tensor([[1,2,3,4,5,6]])
        labels = torch.tensor([[-100,-100,3,-100,5,6]])
        full = model(ids, slice(None)).logits
        loss = F.cross_entropy(full[:, :-1].reshape(-1,13), labels[:,1:].reshape(-1))
        loss.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        selected = assistant_loss(model, {'input_ids':ids,'labels':labels})
        selected.backward()
        torch.testing.assert_close(loss, selected)
        for before, p in zip(gradients, model.parameters()):
            torch.testing.assert_close(before, p.grad)
        self.assertGreater(model.embed.weight.grad[1].abs().sum().item(), 0)

    def test_empty_and_multi_batch_rejected(self):
        model=TinyModel()
        for labels in (torch.full((1,3),-100), torch.ones((2,3),dtype=torch.long)):
            with self.assertRaises(ValueError):
                assistant_loss(model, {'input_ids':torch.ones_like(labels),'labels':labels})
