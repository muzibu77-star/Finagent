from decimal import Decimal
import unittest

from src.evaluation.draft_pilot import check, tolerance


class DraftPilotTest(unittest.TestCase):
    def test_tolerance_is_half_the_last_written_digit(self):
        self.assertEqual(tolerance('1741.44'), Decimal('0.005'))
        self.assertEqual(tolerance('15'), Decimal('0.5'))

    def test_amounts_and_percentages_are_checked_at_written_precision(self):
        review = '营业收入1,741.44亿元，同比增长15.7%；净利润893.35亿元，费用率下降0.3个百分点，研发投入12万元。'
        numbers = check(review, [Decimal('174144069958.25'), Decimal('-120000')],
                        [Decimal('15.66')])
        self.assertEqual([(n['text'], n['supported']) for n in numbers],
                         [('1,741.44亿元', True), ('15.7%', True), ('893.35亿元', False),
                          ('12万元', True)])

    def test_digits_inside_a_longer_number_are_not_read_separately(self):
        self.assertEqual([n['text'] for n in check('增长115.7%', [], [Decimal('15.7')])], ['115.7%'])


if __name__ == '__main__':
    unittest.main()
