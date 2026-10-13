import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.training.m0_lora_check import prepare_samples, read_jsonl


class PreparedReadBoundaryTests(unittest.TestCase):
    def test_prepared_training_only_decodes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {'manifest': [{'task_id': 'train', 'source_report_id': 'train-report',
                                 'original_split': 'train', 'official_eligible': True}],
                    'gold': [{'task_id': 'heldout', 'gold': 'must not decode'}],
                    'questions': [], 'evidence': []}
            for name, rows in data.items():
                (root / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (root / 'summary.json').write_text('{"split_protocol_frozen": true}')
            (root / 'validation.json').write_text(json.dumps({'passed': True, 'sha256': {
                n: hashlib.sha256((root / f'{n}.jsonl').read_bytes()).hexdigest() for n in data}}))
            sample = {'task_id': 'train', 'input_ids': [1, 2], 'labels': [-100, 2]}
            path = root / 'prepared.json'
            path.write_text(json.dumps([sample]))
            config = {'data_dir': str(root), 'prepared_samples': str(path),
                      'samples_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'samples': 1, 'max_sequence_length': 2}
            def manifest_only(path):
                self.assertEqual(path.name, 'manifest.jsonl')
                return read_jsonl(path)
            with patch('src.training.m0_lora_check.read_jsonl', side_effect=manifest_only):
                self.assertEqual(prepare_samples(config, None)[0], [sample])
