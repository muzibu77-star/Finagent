"""Publication cutoff and immutable snapshot checks using explicit synthetic versions."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.data.report_corpus import ReportCorpus
from src.agent.fixed_runner import FixedRunner
from src.retrieval.report_index import ReportIndex
from tests.test_runner import ScriptedModel, tool
from src.harness.store import Store
from src.retrieval.report_index import terms


def document(key, published, supersedes=None, ingested='2026-01-01T00:00:00+00:00'):
    return {'company': 'A', 'period': '2023', 'report_period': '2023',
            'source_report_id': key, 'version': key, 'published_at': published,
            'ingested_at': ingested, 'supersedes': supersedes,
            'evidence': {'evidence_id': key, 'table': [['收入', '100']], 'pre_text': []}}


class ReportCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.docs = {'old': document('old', '2024-03-01'),
                     'new': document('new', '2024-05-01', 'old'),
                     'unknown': document('unknown', None)}
        self.payload = {'question': '收入？', 'company': 'A', 'period': '2023',
                        'snapshot_id': 'snapshot', 'as_of': '2024-04-01'}

    def tearDown(self):
        self.temp.cleanup()

    def corpus(self, docs=None, name='snapshot'):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        path = directory / 'corpus.json'
        path.write_text(json.dumps(self.docs if docs is None else docs))
        (directory / 'receipt.json').write_text(json.dumps({
            'snapshot_id': name, 'corpus_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
        return ReportCorpus(str(path), name)

    def test_cutoff_and_replacement(self):
        corpus = self.corpus()
        try:
            self.assertEqual(corpus.allowed(self.payload), {'old'})
            self.assertEqual(corpus.allowed({**self.payload, 'as_of': '2024-05-01'}), {'new'})
            self.assertEqual(corpus.allowed({**self.payload, 'as_of': '2024-02-29'}), set())
            self.assertEqual(corpus.allowed({**self.payload, 'as_of': None}), {'new', 'unknown'})
            self.assertEqual(set(corpus.documents), {'old', 'new', 'unknown'})
            for mode in ('bm25', 'fusion', 'rerank'):
                self.assertEqual(corpus.index.search('收入', corpus.allowed(self.payload), mode), ['old'])
            with self.assertRaisesRegex(ValueError, 'outside'):
                corpus.read('new', corpus.allowed(self.payload))
        finally:
            corpus.close()

    def test_ingestion_is_not_publication(self):
        corpus = self.corpus({'old': self.docs['old']}, 'early')
        later = self.corpus(self.docs, 'later')
        try:
            self.assertEqual(corpus.allowed({**self.payload, 'snapshot_id': 'early', 'as_of': '2024-06-01'}), {'old'})
            self.assertEqual(later.allowed({**self.payload, 'snapshot_id': 'later', 'as_of': '2024-06-01'}), {'new'})
            self.assertNotEqual(corpus.fingerprint, later.fingerprint)
        finally:
            corpus.close()
            later.close()

    def test_invalid_dates_or_snapshot(self):
        corpus = self.corpus()
        try:
            for change in ({'as_of': '20240301'}, {'as_of': '2024-03-01T12:00:00Z'},
                           {'snapshot_id': 'other'}):
                with self.assertRaises(ValueError):
                    corpus.allowed({**self.payload, **change})
        finally:
            corpus.close()

    def test_tampering_rejected(self):
        corpus = self.corpus()
        corpus.close()
        path = self.root / 'snapshot/corpus.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'hash'):
            ReportCorpus(str(path), 'snapshot')

    def test_invalid_replacement(self):
        for key, value in [('supersedes', 'absent'), ('published_at', '2024-01-01'),
                           ('company', 'B'), ('ingested_at', '2026-01-01')]:
            docs = copy.deepcopy(self.docs)
            docs['new'][key] = value
            with self.assertRaises(ValueError):
                self.corpus(docs)

    def test_scope_change_clears_old_facts_history_and_cache(self):
        store = Store(str(self.root / 'tasks.sqlite'))
        receipt = store.submit('r1', self.payload)
        store.claim(receipt['task_id'])
        store.save('r1', 1, {'facts': {'f0': 'old'}, 'history': ['prior']}, 'test', {})
        store.finish('r1', 1, 'answered', {'calculations': []})
        store.cached({'as_of': self.payload['as_of'], 'snapshot': 'old'}, {'id': 'old'})
        store.submit('r2', {**self.payload, 'as_of': '2024-05-01'}, task_id='r1')
        self.assertEqual(store.get('r1')['state'], {'history': []})
        self.assertIsNone(store.cached({'as_of': '2024-05-01', 'snapshot': 'old'}))
        self.assertIsNone(store.cached({'as_of': self.payload['as_of'], 'snapshot': 'new'}))

    def test_future_document_does_not_influence_ranking_statistics(self):
        permitted = {'a': document('a', '2024-01-01'), 'b': document('b', '2024-01-01')}
        permitted['b']['evidence']['table'] = [['收入', '利润', '200']]
        extra = document('future', '2025-01-01')
        extra['evidence']['table'] = [['收入'] * 1000]
        baseline, expanded = ReportIndex(permitted), ReportIndex({**permitted, 'future': extra})
        try:
            for mode in ('bm25', 'fusion', 'rerank'):
                self.assertEqual(baseline.search('收入 利润', set(permitted), mode),
                                 expanded.search('收入 利润', set(permitted), mode))
        finally:
            baseline.close()
            expanded.close()

    def test_fixed_workflow_cites_only_current_cutoff(self):
        self.docs['new']['evidence']['table'] = [['收入', '110']]
        corpus = self.corpus()
        store = Store(str(self.root / 'tasks.sqlite'))
        action = tool('calculate', steps=[{'operation': 'multiply', 'args': ['f0', 'const_1']}])
        try:
            store.submit('r1', self.payload)
            old = FixedRunner(store, corpus, ScriptedModel([action])).run('r1')
            self.assertEqual(old['report']['calculations'][0]['value'], 100)
            store.submit('r2', {**self.payload, 'as_of': '2024-05-01'}, 'r1')
            new = FixedRunner(store, corpus, ScriptedModel([action])).run('r1')
            self.assertEqual(new['report']['calculations'][0]['value'], 110)
            self.assertEqual(new['report']['request_scope']['as_of'], '2024-05-01')
            self.assertEqual(new['report']['calculations'][0]['facts']['f0']['evidence_id'], 'new')
        finally:
            corpus.close()

    def test_chinese_tokenization(self):
        self.assertEqual(terms('收入2023 Growth'), ['收入', '2023', 'growth'])
        self.assertIn('收入', terms('營運收入'))


if __name__ == '__main__':
    unittest.main()
