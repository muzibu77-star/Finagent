"""Tests for binding an agent answer back to the facts its own run read."""

from decimal import Decimal
import unittest

from src.agent.recheck import (EXPECTED_VALUE, SAME_VALUE, SELF_CONTRADICTED, backfill,
                                in_unit, written)


def fact(key: int, text: str, unit: str = '元', scale: str = '1', page: int = 1,
         start: int = 0, end: int = 1) -> dict:
    """One fact as ``find_number`` records it: the fields the view and the match need."""
    return {'id': key, 'document_id': 'r2024', 'page': page, 'start': start, 'end': end,
            'text': text, 'unit': unit, 'scale': scale, 'row_label': '一、营业收入',
            'column_header': '2024年度', 'title': '合并利润表'}


# 174,144,069,958.25元 is 1,741.44亿元; 320,000,000.00元 is 3.20亿元.
REVENUE = fact(1, '174,144,069,958.25', start=10, end=30)
PROFIT = fact(2, '320,000,000.00', start=40, end=52)


class WrittenTest(unittest.TestCase):
    def test_the_digits_and_the_unit_of_a_written_value(self):
        self.assertEqual(written('764.47 亿元', '亿元'), ('764.47', '亿元'))
        self.assertEqual(written('1,741.44', '亿元'), ('1,741.44', '亿元'))
        self.assertEqual(written('310,000,000.00元', '亿元'), ('310,000,000.00', '元'))
        self.assertEqual(written('-1,476,991,223.18 元', '元'), ('-1,476,991,223.18', '元'))

    def test_a_text_without_a_number_yields_nothing(self):
        self.assertIsNone(written('未给应有值', '亿元'))
        self.assertIsNone(written('', '亿元'))
        self.assertIsNone(written(None, '亿元'))


class InUnitTest(unittest.TestCase):
    def test_a_money_fact_is_read_in_the_unit_of_the_draft_number(self):
        self.assertEqual(in_unit(REVENUE, '亿元'), in_unit(REVENUE, '元') / 10 ** 8)
        self.assertEqual(in_unit(PROFIT, '亿元'), Decimal('3.20'))

    def test_a_percentage_fact_stands_in_percentage_points(self):
        self.assertEqual(str(in_unit(fact(3, '15.66', unit='%'), '%')), '15.66')


class BackfillTest(unittest.TestCase):
    """The facts attached to one agent answer: the draft's figure or the expected one."""

    def test_a_consistent_answer_is_backed_by_the_fact_carrying_the_drafted_number(self):
        found = backfill([REVENUE], verdict='consistent', stated='1,741.44', unit='亿元')
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['mark'], SAME_VALUE)
        self.assertEqual((found[0]['id'], found[0]['page'], found[0]['start'],
                          found[0]['end']), (1, 1, 10, 30))
        self.assertEqual(found[0]['row_label'], '一、营业收入')

    def test_a_wrong_answer_is_backed_by_the_fact_carrying_the_expected_value(self):
        found = backfill([REVENUE, PROFIT], verdict='inconsistent', stated='3.20',
                         unit='亿元', expected='3.10亿元')
        self.assertEqual(found, [])
        found = backfill([REVENUE, PROFIT], verdict='inconsistent', stated='3.30',
                         unit='亿元', expected='3.20亿元')
        self.assertEqual([item['mark'] for item in found], [EXPECTED_VALUE])
        self.assertEqual([item['id'] for item in found], [2])

    def test_a_wrong_answer_expecting_the_drafted_number_is_marked_self_contradicted(self):
        # 茅台 8.43亿元: the agent said wrong but expected the very number drafted.
        found = backfill([PROFIT], verdict='inconsistent', stated='3.20', unit='亿元',
                         expected='3.2亿元（与点评一致）')
        self.assertEqual([item['mark'] for item in found], [SELF_CONTRADICTED])
        found = backfill([PROFIT], verdict='inconsistent', stated='3.20', unit='亿元',
                         expected='320,000,000.00元')
        self.assertEqual([item['mark'] for item in found], [SELF_CONTRADICTED])

    def test_the_expected_value_is_converted_from_its_own_unit(self):
        found = backfill([PROFIT], verdict='inconsistent', stated='3.20', unit='亿元',
                         expected='32,000.00万元')
        self.assertEqual([item['id'] for item in found], [2])

    def test_a_number_that_is_not_money_is_not_backfilled(self):
        percent = fact(4, '15.66', unit='%')
        self.assertEqual(backfill([percent], verdict='consistent', stated='15.66', unit='%'),
                         [])

    def test_nothing_is_attached_without_a_value_to_look_for(self):
        self.assertEqual(backfill([PROFIT], verdict='inconsistent', stated='3.20', unit='亿元'),
                         [])
        self.assertEqual(backfill([PROFIT], verdict='inconsistent', stated='3.20', unit='亿元',
                                  expected='未给应有值'), [])
        self.assertEqual(backfill([PROFIT], verdict='unsupported', stated='3.20',
                                  unit='亿元'), [])

    def test_a_value_that_differs_at_the_written_precision_is_not_attached(self):
        self.assertEqual(backfill([PROFIT], verdict='consistent', stated='3.21', unit='亿元'),
                         [])
        # half a unit of the last written digit, exactly as the checker compares
        rounded = fact(2, '320,040,000.00', start=40, end=52)
        self.assertEqual([item['id'] for item in
                          backfill([rounded], verdict='consistent', stated='3.20',
                                   unit='亿元')], [2])

    def test_at_most_three_facts_in_the_order_the_agent_read_them(self):
        facts = [fact(key, '320,000,000.00', start=key) for key in (9, 5, 7, 4, 6)]
        found = backfill(facts, verdict='consistent', stated='3.20', unit='亿元')
        self.assertEqual([item['id'] for item in found], [9, 5, 7])
        self.assertEqual([item['start'] for item in found], [9, 5, 7])

    def test_a_fact_the_agent_read_twice_is_attached_once(self):
        found = backfill([REVENUE, REVENUE], verdict='consistent', stated='1,741.44',
                         unit='亿元')
        self.assertEqual([item['id'] for item in found], [1])

    def test_a_fact_the_run_never_saw_is_never_attached(self):
        other = fact(5, '111,111,111,111.11')
        self.assertEqual(backfill([other], verdict='consistent', stated='1,741.44',
                                  unit='亿元'), [])


if __name__ == '__main__':
    unittest.main()
