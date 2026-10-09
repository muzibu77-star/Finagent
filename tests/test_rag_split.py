import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.data.leakage import content_text
from src.evaluation import rag_generation, validate_rag_generation
from src.evaluation.retrieval_run import metrics


class RagSplitTests(unittest.TestCase):
    def test_split_membership_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [{'id': split, 'question': 'Revenue?', 'query': 'Revenue?',
                      'split': split, 'company': 'issuer', 'source_report_id': 'report',
                      'relevant': ['p']} for split in ('dev', 'test')]
            evidence = {'evidence_id': 'p', 'table': [],
                        'paragraphs': [{'text': 'Revenue is 30.'}]}
            corpus = root / 'corpus.json'
            corpus.write_text(json.dumps({'p': {'evidence': evidence}}))
            gold = {split: {'answer': '30', 'answer_type': 'arithmetic', 'scale': ''}
                    for split in ('dev', 'test')}
            for name, value in [('tasks', tasks), ('gold', gold)]:
                (root / f'{name}.json').write_text(json.dumps(value))
            lock = {'corpus_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
                    'sha256': {name: hashlib.sha256((root / f'{name}.json').read_bytes()).hexdigest()
                               for name in ('tasks', 'gold')}}
            (root / 'lock.json').write_text(json.dumps(lock))
            text = content_text(evidence)
            results = {mode: {'ranking': ['p'], 'chunks': [
                {'page_id': 'p', 'start': 0, 'end': len(text), 'text': text}],
                **metrics(['p'], ['p'])} for mode in validate_rag_generation.MODES}
            retrieval = root / 'retrieval'
            retrieval.mkdir()
            (retrieval / 'records.jsonl').write_text(''.join(json.dumps(
                {'id': t['id'], 'results': results}) + '\n' for t in tasks))
            for split, expected in [('dev', {'dev'}), ('test', {'test'}), (None, {'dev', 'test'})]:
                output = root / (split or 'all')
                arguments = ['rag_generation', '--tasks-dir', str(root), '--corpus', str(corpus),
                             '--retrieval-dir', str(retrieval), '--output-dir', str(output),
                             '--server-url', 'http://127.0.0.1:18081']
                if split:
                    arguments += ['--split', split]
                with patch('sys.argv', arguments), patch.object(rag_generation, 'HttpDriver') as driver:
                    model = driver.return_value.__enter__.return_value
                    model.generate.side_effect = lambda *a: {'text': json.dumps(
                        {'answer': 30, 'scale': '', 'citations': ['p']}), 'stopped_on_eos': True}
                    rag_generation.main()
                    self.assertEqual(model.generate.call_count, len(expected) * 4)
                rows = [json.loads(line) for line in (output / 'records.jsonl').read_text().splitlines()]
                self.assertEqual({row['id'] for row in rows}, expected)
                with patch('sys.argv', ['audit', '--run', str(output), '--tasks-dir', str(root),
                                       '--corpus', str(corpus), '--retrieval-dir', str(retrieval)]):
                    validate_rag_generation.main()
                audit = json.loads((output / 'validation.json').read_text())
                self.assertTrue(audit['passed'])
                self.assertEqual(audit['calls'], len(expected) * 4)


if __name__ == '__main__':
    unittest.main()
