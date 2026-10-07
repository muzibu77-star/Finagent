import unittest

from src.evaluation.numeric_score import parse_numeric, scalar, score


class NumericScoreTests(unittest.TestCase):
    def test_scale_and_percent(self):
        self.assertEqual(scalar('1,200', 'million'), 1200000000)
        self.assertEqual(scalar('12.34', 'percent'), 0.1234)
        self.assertEqual(scalar('12.34%', ''), 0.1234)
        self.assertEqual(scalar('(12.5)', 'million'), -12500000)
        self.assertTrue(score({'answer': '1200', 'scale': 'million'},
                              {'answer': ['1.2'], 'scale': 'billion'})['numeric_em'])

    def test_unscaled_percentage_alternative(self):
        self.assertTrue(score({'answer': '0.1234', 'scale': ''},
                              {'answer': 12.34, 'scale': 'percent'})['numeric_em'])

    def test_invalid_formats(self):
        for text in ('{"answer":true,"scale":""}', '{"answer":"NaN","scale":""}',
                     '{"answer":"21 December 2018","scale":""}',
                     '{"answer":12,"scale":"millions"}', 'The answer is 12'):
            with self.assertRaises(ValueError):
                parse_numeric(text)


if __name__ == '__main__':
    unittest.main()
