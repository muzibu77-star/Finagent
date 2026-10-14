import unittest

from src.evaluation.freeze_set import sample, split_of

BY_CODE = {'600000': 'training', '000001': 'acceptance'}


class FreezeSetTest(unittest.TestCase):
    def test_issuer_decides_the_split(self):
        self.assertEqual(split_of({'code': '000001', 'document_id': 'x'}, BY_CODE), 'acceptance')

    def test_material_without_issuer_is_spread_stably(self):
        names = [split_of({'code': None, 'document_id': f'meeting-{i}'}, BY_CODE) for i in range(300)]
        self.assertEqual(names, [split_of({'code': None, 'document_id': f'meeting-{i}'}, BY_CODE)
                                 for i in range(300)])
        self.assertEqual(set(names), {'training', 'development', 'acceptance'})

    def test_sample_caps_each_type_and_ignores_input_order(self):
        rows = ([{'id': f'a{i}', 'error': 'E1'} for i in range(10)]
                + [{'id': f'b{i}', 'error': None} for i in range(10)]
                + [{'id': f'c{i}', 'error': 'E2'} for i in range(2)])
        kept = sample(rows, 3, 5, 1)
        self.assertEqual(kept, sample(list(reversed(rows)), 3, 5, 1))
        self.assertEqual({e: sum((r['error'] or 'ok') == e for r in kept) for e in ('E1', 'E2', 'ok')},
                         {'E1': 3, 'E2': 2, 'ok': 5})


if __name__ == '__main__':
    unittest.main()
