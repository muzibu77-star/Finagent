import json
from pathlib import Path
import tempfile
import unittest

from src.agent.fixed_runner import FixedRunner
from src.data.corpus import Corpus
from src.harness.store import Store
from tests.test_runner import ScriptedModel, tool


class FixedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / 'corpus.json').write_text(json.dumps({'e1': {'company': 'A', 'period': '2023',
            'source_report_id': 'A/2023', 'evidence': {'evidence_id': 'e1', 'table': [['Revenue', '120', '150']]}}}))
        self.corpus = Corpus(str(root / 'corpus.json'), 's1')
        self.store = Store(str(root / 'tasks.sqlite'))
        self.payload = {'company': 'A', 'period': '2023', 'snapshot_id': 's1', 'question': 'Revenue change?'}
        self.model = ScriptedModel([tool('calculate', steps=[{'operation': 'subtract', 'args': ['f1', 'f0']}])])

    def tearDown(self):
        self.corpus.close()
        self.temp.cleanup()

    def test_single_call_and_atomic_report(self):
        self.store.submit('r1', self.payload)
        row = FixedRunner(self.store, self.corpus, self.model).run('r1')
        self.assertEqual(row['report']['calculations'][0]['value'], 30)
        self.assertEqual(row['report']['workflow'], 'fixed-v1')
        self.assertEqual(row['state']['attempts'], 1)
        self.assertEqual(FixedRunner(self.store, self.corpus, self.model).run('r1')['report'], row['report'])

    def test_failed_call_is_not_replayed_after_crash(self):
        self.store.submit('r1', self.payload)
        self.store.claim('r1')
        self.store.save('r1', 1, {'attempts': 1}, 'generation_started', {})
        self.store.recover()
        row = FixedRunner(self.store, self.corpus, ScriptedModel([])).run('r1')
        self.assertEqual(row['execution'], 'failed')
        self.assertIsNone(row['report'])

    def test_saved_generation_can_finish_after_crash(self):
        self.store.submit('r1', self.payload)
        self.store.claim('r1')
        record = next(self.model.outputs)
        self.store.save('r1', 1, {'attempts': 1, 'actual_output_tokens': 15,
            'prediction_record': record}, 'generation_finished', record)
        self.store.recover()
        row = FixedRunner(self.store, self.corpus, ScriptedModel([])).run('r1')
        self.assertEqual(row['report']['calculations'][0]['value'], 30)

    def test_actual_cancellation(self):
        self.store.submit('r1', self.payload)
        def cancel(messages, tools, cancelled, deadline):
            self.store.cancel('r1')
            self.assertTrue(cancelled())
            return {'text': 'partial', 'stopped_on_eos': False, 'new_tokens': 7}
        row = FixedRunner(self.store, self.corpus, cancel).run('r1')
        self.assertEqual(row['execution'], 'cancelled')
        self.assertEqual(row['state']['actual_output_tokens'], 7)
        self.assertIsNone(row['report'])


if __name__ == '__main__':
    unittest.main()
