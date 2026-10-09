"""Keep complete Unicode document evidence without changing the legacy prompt."""
import json
from pathlib import Path
import tempfile
import unittest

from src.agent.fixed_runner import FixedRunner
from src.data.corpus import Corpus
from src.harness.store import Store
from src.model.financial import messages_for
from src.service.server import Service
from tests.test_runner import tool


class DocumentPromptTests(unittest.TestCase):
    def test_unicode_preserves_evidence_and_followup_history(self):
        for unicode_context in (False, True):
            with self.subTest(unicode_context=unicode_context), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                evidence = {'evidence_id': 'e1',
                            'table': [['Revenue 营业收入', '120', '150']],
                            'post_text': ['金额以百万元计；上年和本年口径一致。']}
                path = root / 'corpus.json'
                path.write_text(json.dumps({'e1': {'company': 'A', 'period': '2023',
                    'source_report_id': 'A/2023', 'evidence': evidence}}))
                corpus = Corpus(str(path), 's1')
                store = Store(str(root / 'state.sqlite'))
                captured = []

                def model(messages, tools, cancelled, deadline):
                    captured.append(messages)
                    return tool('calculate', steps=[
                        {'operation': 'subtract', 'args': ['f1', 'f0']}])

                runner = FixedRunner(store, corpus, model, unicode_context=unicode_context)
                payload = {'company': 'A', 'period': '2023', 'snapshot_id': 's1',
                           'question': 'Revenue 营业收入增加多少？'}
                try:
                    store.submit('first', payload)
                    first = runner.run('first')
                    self.assertEqual(first['report']['calculations'][0]['value'], 30)
                    original, facts = messages_for(evidence, payload['question'], 'calculator')
                    self.assertEqual(json.loads(captured[0][-1]['content']),
                                     json.loads(original[-1]['content']))
                    if not unicode_context:
                        self.assertEqual(captured[0], original)
                    else:
                        self.assertIn('For a single-value lookup, use multiply',
                                      captured[0][0]['content'])
                        self.assertIn('营业收入', captured[0][-1]['content'])
                        self.assertNotIn('\\u8425', captured[0][-1]['content'])
                    self.assertEqual(first['state']['read']['e1'], list(facts))
                    store.submit('followup', {**payload, 'question': 'Revenue 再核对一次。'}, 'first')
                    second = runner.run('first')
                    self.assertEqual(second['report']['calculations'][0]['value'], 30)
                    content = json.loads(captured[1][-1]['content'])
                    history = json.loads(content['question'])['history']
                    self.assertEqual(content['evidence'], evidence)
                    self.assertEqual(history[0]['question'], payload['question'])
                    if unicode_context:
                        self.assertNotIn('\\u8425', captured[1][-1]['content'])
                finally:
                    corpus.close()

    def test_report_prompt_version_prevents_old_database_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'corpus.json'
            path.write_text('{}')
            database = str(root / 'state.sqlite')
            service = Service(database, str(path), 's1', None,
                              reports=True)
            with service.store.transaction() as db:
                body = json.loads(db.execute('SELECT body FROM service_config WHERE id=1').fetchone()['body'])
                self.assertEqual(body['prompt_serialization'], 'unicode-v1')
                self.assertEqual(body['document_tool_contract'], 'lookup-v1')
                del body['prompt_serialization']
                db.execute('UPDATE service_config SET body=? WHERE id=1',
                           (json.dumps(body, sort_keys=True),))
            with self.assertRaisesRegex(ValueError, 'different source snapshot'):
                Service(database, str(path), 's1', None,
                        reports=True)


if __name__ == '__main__':
    unittest.main()
