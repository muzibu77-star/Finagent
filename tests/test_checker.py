import json
import sqlite3
import unittest

from src.data.library import SCHEMA, add_page, digits
from src.verify.checker import Checker, Claim, role


def document(db, key, kind, year, published, code='600000'):
    db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               (key, kind, '某公司', code, year, '年度报告', 'http://x', published, 'h', 1))


def fact(db, key, label, header, text, scale='1', scope='合并', title='合并利润表'):
    db.execute('INSERT INTO facts (document_id, page, start, end, box, text, digits, row_label, '
               'column_header, title, unit, scale, scope) VALUES (?, 1, 0, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               (key, json.dumps([]), text, digits(text), label, header, title, '元', scale, scope))


def build():
    db = sqlite3.connect(':memory:')
    db.executescript(SCHEMA)
    document(db, 'r2023', 'annual', 2023, '2024-03-30')
    document(db, 'r2024', 'annual', 2024, '2025-03-30')
    fact(db, 'r2024', '一、营业收入', '2024年度', '174,144,069,958.25')
    fact(db, 'r2024', '一、营业收入', '2023年度', '150,560,330,316.45')
    fact(db, 'r2024', '营业收入', '本期发生额', '90,000,000,000.00', scope='母公司', title='母公司利润表')
    fact(db, 'r2023', '营业收入', '2023年度', '150,560,330,316.45')
    fact(db, 'r2024', '营业成本', '2024年度', '13,000,000,000.00')
    fact(db, 'r2024', '营业成本', '2023年度', '12,000,000,000.00')
    fact(db, 'r2024', '营业成本', '2023年度 重述前', '11,000,000,000.00')
    document(db, 'm1', 'executive-meetings', None, '2024-06-07', code=None)
    add_page(db, 'm1', 1, '新华社北京6月7日电 国务院总理李强6月7日主持召开国务院常务会议，听取当前房地产市场形势汇报。')
    return Checker(db)


def claim(stated, **kwargs):
    return Claim(**{'code': '600000', 'metric': '营业收入', 'fiscal_year': 2024, 'stated': stated,
                    'unit': '亿元', 'as_of': '2025-04-30', **kwargs})


class CheckerTest(unittest.TestCase):
    def setUp(self):
        self.checker = build()

    def test_consistent_and_rounding(self):
        self.assertEqual(self.checker.check(claim('1741.44')).verdict, 'consistent')
        rounded = self.checker.check(claim('1741.45'))
        self.assertEqual((rounded.verdict, rounded.reason), ('consistent', 'rounding difference'))

    def test_each_mistake_is_named(self):
        cases = {'1745.44': 'digit', '1505.60': 'period', '900.00': 'scope', '1741.44万元': 'unit'}
        for stated, error in cases.items():
            number, unit = (stated[:-2], '万元') if stated.endswith('万元') else (stated, '亿元')
            verdict = self.checker.check(claim(number, unit=unit))
            self.assertEqual((verdict.verdict, verdict.error), ('inconsistent', error), stated)

    def test_nothing_published_yet_is_unsupported(self):
        self.assertEqual(self.checker.check(claim('1741.44', as_of='2025-03-01')).verdict, 'unsupported')

    def test_growth_and_wrong_denominator(self):
        self.assertEqual(self.checker.check(claim('15.7', unit='%')).verdict, 'consistent')
        wrong = self.checker.check(claim('13.5', unit='%'))
        self.assertEqual((wrong.verdict, wrong.error, wrong.expected), ('inconsistent', 'calculation', '15.7'))

    def test_source_note_naming_a_report_without_the_figure(self):
        verdict = self.checker.check(claim('1741.44', cited_year=2023))
        self.assertEqual((verdict.verdict, verdict.error), ('inconsistent', 'source'))

    def test_meeting_date(self):
        base = {'kind': 'date', 'agenda': '听取当前房地产市场形势汇报', 'as_of': '2026-10-14', 'unit': ''}
        self.assertEqual(self.checker.check(claim('2024年6月7日', **base)).verdict, 'consistent')
        wrong = self.checker.check(claim('2026年6月7日', **base))
        self.assertEqual((wrong.verdict, wrong.expected), ('inconsistent', '2024年6月7日'))

    def test_margin_from_line_items(self):
        margin = claim('92.5', unit='%', kind='ratio', numerator=['营业收入', '-营业成本'],
                       denominator=['营业收入'])
        self.assertEqual(self.checker.check(margin).verdict, 'consistent')
        self.assertEqual(self.checker.check(claim('90.0', unit='%', kind='ratio',
                         numerator=['营业成本'], denominator=['营业收入'])).verdict, 'inconsistent')
        self.assertEqual(self.checker.check(claim('9.0', unit='%', kind='ratio',
                         numerator=['研发费用'], denominator=['营业收入'])).verdict, 'unsupported')

    def test_change_in_amount_uses_the_restated_comparative(self):
        self.assertEqual(self.checker.check(claim('10.00', kind='change', metric='营业成本')).verdict,
                         'consistent')
        self.assertEqual(self.checker.check(claim('20.00', kind='change', metric='营业成本')).verdict,
                         'inconsistent')

    def test_direction_of_change(self):
        self.assertEqual(self.checker.check(claim('15.7', unit='%', direction=1)).verdict, 'consistent')
        wrong = self.checker.check(claim('15.7', unit='%', direction=-1))
        self.assertEqual((wrong.verdict, wrong.error), ('inconsistent', 'direction'))

    def test_written_year_decides_the_column(self):
        self.assertEqual(role('2024年末 调整后', 2025), 'prior')
        self.assertEqual(role('2025年末', 2025), 'current')
        self.assertIsNone(role('2024年末 调整前', 2025))
        self.assertEqual(role('期末余额', 2025), 'current')

    def test_outflows_have_no_direction_check(self):
        outflow = claim('8.3', unit='%', direction=-1, metric='营业成本')
        self.assertEqual(self.checker.check(outflow).error, 'direction')
        self.checker.db.execute("UPDATE facts SET row_label = '支付的各项税费' WHERE row_label = '营业成本'")
        self.checker._facts.clear()
        self.assertNotEqual(self.checker.check(claim('8.3', unit='%', direction=-1,
                                                     metric='支付的各项税费')).error, 'direction')


if __name__ == '__main__':
    unittest.main()
