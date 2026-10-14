import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.training.m0_lora_check import validated_initial_adapter


class TrainingInitializationTests(unittest.TestCase):
    def test_initial_weights_are_hash_and_model_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            adapter = {'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0, 'target_modules': 'linear',
                       'bias': 'none', 'base_model_name_or_path': '/model', 'peft_type': 'LORA',
                       'rank_pattern': {}, 'alpha_pattern': {}, 'use_dora': False,
                       'use_rslora': False, 'modules_to_save': None}
            (path / 'adapter_config.json').write_text(json.dumps(adapter))
            (path / 'adapter_model.safetensors').write_bytes(b'fixture; not loaded by hash validation')
            config = {'initial_adapter': str(path), 'rank': 8, 'target_modules': 'linear',
                      'initial_adapter_sha256': {
                          p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}}
            self.assertEqual(validated_initial_adapter(config, {'path': '/model'}), path)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                validated_initial_adapter(config, {'path': '/other-model'})
            (path / 'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hashes'):
                validated_initial_adapter(config, {'path': '/model'})
            self.assertIsNone(validated_initial_adapter({}, {'path': '/model'}))
