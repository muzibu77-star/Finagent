import unittest

from src.evaluation.splits import assign

ISSUERS = [{'code': f'{i:06d}', 'name': f'公司{i}', 'industry': f'行业{i % 12}'} for i in range(30)]


class SplitsTest(unittest.TestCase):
    def test_sizes_disjoint_and_deterministic(self):
        first, again = assign(ISSUERS, 7), assign(list(reversed(ISSUERS)), 7)
        self.assertEqual(first, again)
        self.assertEqual({k: len(v) for k, v in first.items()},
                         {'acceptance': 8, 'development': 6, 'training': 16})
        codes = [row['code'] for rows in first.values() for row in rows]
        self.assertEqual(len(set(codes)), 30)

    def test_held_out_splits_span_distinct_industries(self):
        splits = assign(ISSUERS, 7)
        self.assertEqual(len({row['industry'] for row in splits['acceptance']}), 8)

    def test_wrong_issuer_count_is_rejected(self):
        with self.assertRaises(ValueError):
            assign(ISSUERS[:29], 7)


if __name__ == '__main__':
    unittest.main()
