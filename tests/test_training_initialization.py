import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.training.m0_lora_check import prepare_samples, validated_initial_adapter


class TrainingInitializationTests(unittest.TestCase):
    def test_explicit_autonomous_parent_contract_and_legacy_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = [{'task_id': 'train', 'source_report_id': 'report', 'dataset': 'finqa',
                         'original_split': 'train', 'official_eligible': True}]
            data = {'manifest': manifest, 'evidence': [], 'questions': [], 'gold': []}
            for name, rows in data.items():
                (root / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (root / 'summary.json').write_text('{"split_protocol_frozen": true}')
            (root / 'validation.json').write_text(json.dumps({'passed': True, 'sha256': {
                name: hashlib.sha256((root / f'{name}.jsonl').read_bytes()).hexdigest() for name in data}}))
            sample = {'task_id': 'train:autonomous:0', 'parent_task_id': 'train',
                      'source_report_id': 'report', 'input_ids': [1, 2], 'labels': [-100, 2]}
            path = root / 'samples.json'
            path.write_text(json.dumps([sample]))
            config = {'data_dir': str(root), 'prepared_samples': str(path),
                      'samples_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'samples': 1, 'max_sequence_length': 2}
            with self.assertRaisesRegex(ValueError, 'held-out'):
                prepare_samples(config, None)
            samples, selected = prepare_samples({**config, 'autonomous_supervision': True}, None)
            self.assertEqual(samples, [sample])
            self.assertEqual(selected['selected'], 1)
            sample['task_id'] = 'train'
            path.write_text(json.dumps([sample]))
            config['samples_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(prepare_samples(config, None)[0], [sample])
            pair = {'task_id': 'train', 'source_report_id': 'report',
                    'chosen_reward': 1, 'rejected_reward': 0,
                    'chosen': {'input_ids': [1, 2], 'labels': [-100, 2]},
                    'rejected': {'input_ids': [1, 3], 'labels': [-100, 3]},
                    'reference_chosen_logp': -2.0, 'reference_rejected_logp': -3.0}
            config.update(training_objective='dpo', reference_adapter_sha256={'model': 'hash'},
                          initial_adapter_sha256={'model': 'hash'})
            path.write_text(json.dumps([pair]))
            config['samples_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(prepare_samples(config, None)[0], [pair])
            pair['rejected']['input_ids'][0] = 4
            path.write_text(json.dumps([pair]))
            config['samples_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'same prompt'):
                prepare_samples(config, None)

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
