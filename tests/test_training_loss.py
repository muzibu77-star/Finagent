"""Chunked selective loss against the full-logits reference, on a tiny Qwen3.5."""

import unittest
from unittest import mock

import torch
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from src.training import loss as training_loss
from src.training.loss import assistant_loss

# Same adapters as the training entry, so the tested model is PEFT-wrapped there too.
TARGET_MODULES = (r"model\.language_model\.layers\.\d+\."
                  r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
                  r"linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|"
                  r"mlp\.(gate_proj|up_proj|down_proj))")


def tiny_lora_model() -> torch.nn.Module:
    """A random text-only Qwen3.5 small enough to run on a CPU."""
    text = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
                linear_num_key_heads=4, linear_num_value_heads=4)
    config = Qwen3_5Config(text_config=text, vision_config=dict(
        depth=1, hidden_size=16, num_heads=2, out_hidden_size=32))
    model = get_peft_model(Qwen3_5ForConditionalGeneration(config),
                           LoraConfig(r=4, lora_alpha=16, lora_dropout=0.0, bias='none',
                                      target_modules=TARGET_MODULES))
    model.config.use_cache = False
    return model


class SelectiveLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(9)
        cls.model = tiny_lora_model()
        cls.ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        cls.batch = {'input_ids': cls.ids, 'attention_mask': torch.ones_like(cls.ids),
                     'labels': torch.tensor([[-100, -100, 3, -100, 5, 6, -100, 9]])}

    def reference_loss(self) -> torch.Tensor:
        """The replaced implementation: full-sequence logits, then the selected rows."""
        logits = self.model(**self.batch, logits_to_keep=0).logits
        labels = self.batch['labels']
        positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
        return F.cross_entropy(logits[0, positions].float(), labels[0, positions + 1])

    def test_loss_and_lora_gradients_match_full_logits(self):
        reference = self.reference_loss()
        reference.backward()
        expected = {name: p.grad.clone() for name, p in self.model.named_parameters()
                    if p.requires_grad}
        self.assertTrue(expected)
        # Two chunks cross a chunk boundary, one chunk does not, and the last case
        # stands in for the padding a shorter process adds to keep FSDP collectives
        # in step: empty chunks must leave the value and the gradients untouched.
        schedules = ((1024, 0), (2, 0), (1, 0), (3, 2))
        for positions_per_chunk, padding in schedules:
            with self.subTest(positions_per_chunk=positions_per_chunk, padding=padding):
                self.model.zero_grad(set_to_none=True)
                with mock.patch.object(
                        training_loss, 'CHUNK_POSITIONS', positions_per_chunk), \
                        mock.patch.object(training_loss, 'agreed_chunk_count',
                                          lambda count: count + padding):
                    loss = assistant_loss(self.model, self.batch)
                loss.backward()
                torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-8)
                for name, p in self.model.named_parameters():
                    if p.requires_grad:
                        torch.testing.assert_close(p.grad, expected[name],
                                                   rtol=1e-5, atol=1e-8)

    def test_empty_and_multi_batch_rejected(self):
        # Both checks run before the model is touched, so nothing is forwarded here.
        for labels in (torch.full((1, 3), -100), torch.ones((2, 3), dtype=torch.long)):
            with self.assertRaises(ValueError):
                batch = {'input_ids': torch.ones_like(labels), 'labels': labels}
                assistant_loss(self.model, batch)
