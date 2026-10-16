"""Tests for the whole-set runner: per-model caches, shards and the summary."""

import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

from src.data.library import SCHEMA
from src.evaluation import run_graph

REVIEWS = [{'id': f'rev-{number}', 'document_id': 'r2024', 'code': '600000',
            'company': '某公司', 'fiscal_year': 2024, 'text': '营业收入1,745.44亿元。',
            'numbers': [{'start': 4}]} for number in range(3)]


def paper(review: dict) -> dict:
    """A workpaper with one number the checker settled and no revision."""
    return {'numbers': [{'start': review['numbers'][0]['start'], 'verdict': 'inconsistent',
                         'source': 'checker', 'agent': None, 'suggestion': {}}],
            'errors': [], 'summary': {'numbers': 1}}


class ShardTest(unittest.TestCase):
    def test_contiguous_chunks_cover_the_set_once(self):
        reviews = list(range(7))
        chunks = [[row for row in run_graph.shard_reviews(reviews, index, 3)]
                  for index in (1, 2, 3)]
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 3])
        self.assertEqual([row for chunk in chunks for row in chunk], reviews)

    def test_six_shards_of_192_are_even(self):
        self.assertEqual([len(run_graph.shard_reviews([{}] * 192, index, 6))
                          for index in range(1, 7)], [32] * 6)

    def test_a_shard_number_out_of_range_is_refused(self):
        for index, count in ((0, 3), (4, 3)):
            with self.assertRaises(SystemExit):
                run_graph.shard_reviews([{}], index, count)


class CacheFilesTest(unittest.TestCase):
    def test_the_agent_model_gets_a_cache_of_its_own(self):
        output = Path('out')
        self.assertEqual(run_graph.cache_files(output, 'deepseek-flash', None),
                         [output / 'api_cache.jsonl'])
        self.assertEqual(run_graph.cache_files(output, 'deepseek-flash', 'deepseek-flash'),
                         [output / 'api_cache.jsonl'])
        self.assertEqual(run_graph.cache_files(output, 'qwen3.5-9b-extract', 'qwen3.5-9b'),
                         [output / 'api_cache.jsonl', output / 'api_cache_qwen3.5-9b.jsonl'])


class SummaryTest(unittest.TestCase):
    """The summary names the caches and splits the spend by model."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.library = self.root / 'library.sqlite'
        db = sqlite3.connect(self.library)
        db.executescript(SCHEMA)
        db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                   ('r2024', 'annual', '某公司', '600000', 2024, '年度报告', 'http://x',
                    '2025-03-30', 'h', 1))
        db.commit()
        db.close()
        self.reviews = self.root / 'reviews.jsonl'
        self.reviews.write_text('\n'.join(json.dumps(row, ensure_ascii=False)
                                         for row in REVIEWS) + '\n')
        self.labels = self.root / 'labels.jsonl'
        self.labels.write_text('\n'.join(json.dumps(
            {'review_id': row['id'], 'start': row['numbers'][0]['start'], 'label': 'wrong'})
            for row in REVIEWS) + '\n')

    def tearDown(self):
        self.directory.cleanup()

    def run_main(self, *extra: str) -> dict:
        argv = ['run_graph', '--library', str(self.library), '--reviews', str(self.reviews),
                '--labels', str(self.labels), '--output-dir', str(self.root / 'out'),
                '--budget-usd', '1.0', *extra]
        with mock.patch('src.evaluation.run_graph.review_draft', self.fake), \
                mock.patch.object(sys, 'argv', argv), \
                mock.patch('builtins.print'):
            run_graph.main()
        return json.loads((self.root / 'out' / 'summary.json').read_text())

    def fake(self, draft: dict, *, library, model, url, budget_usd, cache, investigate=True,
             agent_model=None, agent_url=None, agent_cache=None) -> dict:
        """One cache row per client, so the summary can tell the two models apart."""
        clients = [(Path(cache), model, 0.5)]
        if Path(agent_cache) != Path(cache):
            clients.append((Path(agent_cache), agent_model, 0.25))
        for path, name, cost in clients:
            with path.open('a') as handle:
                handle.write(json.dumps({'request_sha256': name + str(cost), 'model': name,
                                         'cost_usd': cost}) + '\n')
        return paper(REVIEWS[0])

    def test_the_two_models_are_cached_and_billed_apart(self):
        summary = self.run_main('--model', 'qwen3.5-9b-extract',
                                '--agent-model', 'qwen3.5-9b')
        self.assertEqual(summary['cache'],
                         ['api_cache.jsonl', 'api_cache_qwen3.5-9b.jsonl'])
        self.assertEqual(summary['cost_usd_by_model'],
                         {'qwen3.5-9b': 0.75, 'qwen3.5-9b-extract': 1.5})
        self.assertEqual(summary['cost_usd_new'], 2.25)
        self.assertEqual(summary['reviews'], 3)

    def test_one_model_keeps_one_cache_and_one_bill(self):
        summary = self.run_main('--model', 'qwen3.5-9b-extract')
        self.assertEqual(summary['cache'], ['api_cache.jsonl'])
        self.assertEqual(summary['cost_usd_by_model'], {'qwen3.5-9b-extract': 1.5})
        self.assertEqual(summary['cost_usd_new'], 1.5)

    def test_a_shard_scores_only_its_chunk(self):
        summary = self.run_main('--model', 'qwen3.5-9b-extract', '--shard', '2', '3')
        self.assertEqual(summary['reviews'], 1)
        self.assertEqual(summary['numbers'], 1)
        papers = sorted(path.name for path in (self.root / 'out' / 'workpapers').iterdir())
        self.assertEqual(papers, ['rev-1.json'])


if __name__ == '__main__':
    unittest.main()
