import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.validate_visual_run import audit


class ChineseAuditTests(unittest.TestCase):
    def test_expanded_text_only_membership_and_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, run = root / 'data', root / 'run'
            data.mkdir()
            run.mkdir()
            (data / 'tasks.json').write_text(json.dumps([{'id': 'new'}]))
            (data / 'gold.json').write_text(json.dumps({
                'new': {'answer': '2.0000', 'unit': '元', 'decimals': 4}}))
            lock = {'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in data.iterdir()}}
            (run / 'settings.json').write_text(json.dumps({
                'tasks_root': str(data), 'modes': ['text'], 'lock': lock}))
            row = {'id': 'new', 'mode': 'text', 'stopped_on_eos': True,
                   'text': '{"answer": 2, "unit": "元"}',
                   'numeric_correct': True, 'unit_correct': True, 'passed': True}
            (run / 'records.jsonl').write_text(json.dumps(row) + '\n')
            self.assertEqual(audit(run, True)['calls'], 1)
            (run / 'records.jsonl').write_text(json.dumps({**row, 'passed': False}) + '\n')
            with self.assertRaises(AssertionError):
                audit(run, True)
