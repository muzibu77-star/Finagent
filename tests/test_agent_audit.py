import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.agent.runner import Runner
from src.evaluation.validate_agent_run import audit
from src.harness.store import Store


class AgentAuditTests(unittest.TestCase):
    def test_new_frozen_root_and_summary_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, run = root / 'data', root / 'run'
            data.mkdir()
            run.mkdir()
            tasks = [{'task_id': 'new-task', 'category': 'clarification',
                      'turns': ['What was revenue?'], 'company': None, 'period': '2019'}]
            inputs = {'tasks': tasks, 'gold': {'new-task': {
                'expected': ['awaiting_input'], 'evidence_ids': [[]]}}, 'corpus': {}}
            for name, value in inputs.items():
                (data / f'{name}.json').write_text(json.dumps(value))
            lock = {'sha256': {name: hashlib.sha256((data / f'{name}.json').read_bytes()).hexdigest()
                               for name in inputs}}
            (data / 'lock.json').write_text(json.dumps(lock))
            (run / 'settings.json').write_text(json.dumps({'tasks_root': str(data), 'lock': lock}))
            store = Store(str(run / 'tasks.sqlite'))
            store.submit('new-task:turn0', {'question': 'What was revenue?',
                         'company': None, 'period': '2019', 'snapshot_id': 'test'})
            corpus = SimpleNamespace(fingerprint='test', companies={'issuer'})
            result = Runner(store, corpus, None).run('new-task:turn0')
            turn = {'execution': result['execution'], 'business': result['business'],
                    'report': result['report'], 'revision': result['revision'],
                    'attempts': 0, 'passed': True}
            (run / 'turns.jsonl').write_text(json.dumps({
                'task_id': 'new-task', 'turn': 0, 'result': turn}) + '\n')
            (run / 'records.jsonl').write_text(json.dumps({
                'task_id': 'new-task', 'category': 'clarification',
                'turns': [turn], 'all_turns_passed': True}) + '\n')
            summary = {'tasks': 1, 'turns': 1, 'categories': {
                'clarification': {'passed': 1, 'total': 1}},
                'fixed_30_calculation_tasks': {'total': 0, 'correct_with_source': 0}}
            (run / 'summary.json').write_text(json.dumps(summary))
            self.assertTrue(audit(run)['passed'])
            summary['categories']['clarification']['passed'] = 0
            (run / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(AssertionError):
                audit(run)
