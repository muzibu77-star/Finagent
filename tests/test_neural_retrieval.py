"""CPU checks for provenance-preserving chunks and ranking aggregation."""

import unittest
from types import SimpleNamespace

import torch

from src.retrieval.neural import NeuralIndex, page_ranking, reciprocal_rank_fusion, split_page


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class NeuralRetrievalTests(unittest.TestCase):
    def test_chunk_spans_cover_tail_and_overlap(self):
        self.assertEqual(split_page('abcdefghij', CharacterTokenizer(), 4, 1),
                         [(0, 4), (3, 7), (6, 10)])
        self.assertEqual(split_page('', CharacterTokenizer(), 4, 1), [])
        with self.assertRaises(ValueError):
            split_page('a', CharacterTokenizer(), 4, 4)

    def test_fusion_does_not_count_duplicate_hits(self):
        self.assertEqual(reciprocal_rank_fusion([['a', 'a', 'b'], ['b', 'c']]),
                         reciprocal_rank_fusion([['a', 'b'], ['b', 'c']]))

    def test_page_collapse_keeps_highest_ranked_chunk(self):
        chunks = {'a': {'page_id': 'p1'}, 'b': {'page_id': 'p2'}, 'c': {'page_id': 'p1'}}
        self.assertEqual(page_ranking(chunks, ['c', 'b', 'a']), ['p1', 'p2'])

    def test_invalid_embeddings_cannot_enter_the_cache(self):
        class Batch(dict):
            def to(self, device):
                return self
        index = NeuralIndex.__new__(NeuralIndex)
        index.device = 'cpu'
        index.config = {'embedding': {'dimension': 2}}
        index.tokenizer = lambda *args, **kwargs: Batch(input_ids=torch.ones(1, 1, dtype=torch.long))
        for value in (float('nan'), 0.0):
            index.encoder = lambda **kwargs: SimpleNamespace(last_hidden_state=torch.full((1, 1, 2), value))
            with self.assertRaisesRegex(ValueError, 'embedding'):
                index.encode(['page'])


if __name__ == '__main__':
    unittest.main()
