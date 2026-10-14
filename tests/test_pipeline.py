"""Tests for the extracted-claims -> verdicts step of the review pipeline."""

import json
import sqlite3
import unittest

from src.data.library import SCHEMA, digits
from src.verify.checker import Checker
from src.verify.pipeline import judge


def document(db, key, kind, year, published, code='600000'):
    db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               (key, kind, '某公司', code, year, '年度报告', 'http://x', published, 'h', 1))


def fact(db, key, label, header, text, scale='1', scope='合并', title='合并利润表'):
    db.execute('INSERT INTO facts (document_id, page, start, end, box, text, digits, row_label, '
               'column_header, title, unit, scale, scope) VALUES (?, 1, 0, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               (key, json.dumps([]), text, digits(text), label, header, title, '元', scale, scope))


def build() -> Checker:
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
    return Checker(db)


class JudgeTest(unittest.TestCase):
    def setUp(self):
        self.checker = build()

    def verdict(self, text: str, item: dict) -> dict:
        """Run ``judge`` on one claim and return its single verdict."""
        verdicts = judge(self.checker, text, [item], '600000', '2025-04-30')
        self.assertEqual(len(verdicts), 1, verdicts)
        return next(iter(verdicts.values()))

    def test_amount_consistent(self):
        item = {'number': '1,741.44', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
                'label': '营业收入', 'phrase': '营业收入', 'year': 2024, 'scope': '合并'}
        self.assertEqual(self.verdict('2024年营业收入1,741.44亿元。', item)['verdict'], 'consistent')

    def test_amount_inconsistent_names_the_mistake(self):
        item = {'number': '1,745.44', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
                'label': '营业收入', 'phrase': '营业收入', 'year': 2024, 'scope': '合并'}
        verdict = self.verdict('2024年营业收入1,745.44亿元。', item)
        self.assertEqual((verdict['verdict'], verdict['error']), ('inconsistent', 'digit'))

    def test_growth(self):
        item = {'number': '15.7', 'occurrence': 1, 'type': 'growth', 'unit': '%',
                'label': '营业收入', 'phrase': '同比增长', 'year': 2024}
        self.assertEqual(self.verdict('营业收入同比增长15.7%。', item)['verdict'], 'consistent')

    def test_ratio(self):
        item = {'number': '92.5', 'occurrence': 1, 'type': 'ratio', 'unit': '%',
                'phrase': '毛利率', 'year': 2024, 'numerator': ['营业收入', '-营业成本'],
                'denominator': ['营业收入']}
        self.assertEqual(self.verdict('毛利率92.5%。', item)['verdict'], 'consistent')

    def test_parent_company_claim_is_unsupported(self):
        item = {'number': '900.00', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
                'label': '营业收入', 'phrase': '母公司营业收入', 'year': 2024, 'scope': '母公司'}
        verdict = self.verdict('母公司营业收入900.00亿元。', item)
        self.assertEqual((verdict['verdict'], verdict['reason']),
                         ('unsupported', 'parent-company claim'))

    def test_quarter_wording_is_unsupported(self):
        item = {'number': '3.20', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
                'label': '净利润', 'phrase': '第四季度净利润', 'year': 2024, 'scope': '合并'}
        verdict = self.verdict('第四季度净利润3.20亿元。', item)
        self.assertEqual((verdict['verdict'], verdict['reason']),
                         ('unsupported', 'quarter, segment or own-method ratio'))

    def test_label_that_does_not_fit_the_wording(self):
        item = {'number': '1.00', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
                'label': '营业收入', 'phrase': '净利润', 'year': 2024, 'scope': '合并'}
        verdict = self.verdict('净利润1.00亿元。', item)
        self.assertEqual((verdict['verdict'], verdict['reason']),
                         ('unsupported', 'line item does not fit the wording'))

    def test_number_not_written_plainly(self):
        item = {'number': '15.7%', 'occurrence': 1, 'type': 'growth', 'unit': '%',
                'label': '营业收入', 'phrase': '同比增长', 'year': 2024}
        verdict = self.verdict('营业收入同比增长15.7%。', item)
        self.assertEqual((verdict['verdict'], verdict['reason']),
                         ('unsupported', 'number not written plainly'))

    def test_checked_claims_carry_their_evidence_fact_ids(self):
        item = {'number': '1,741.44', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
                'label': '营业收入', 'phrase': '营业收入', 'year': 2024, 'scope': '合并'}
        verdict = self.verdict('2024年营业收入1,741.44亿元。', item)
        self.assertEqual((verdict['verdict'], verdict['evidence']), ('consistent', [1]))

    def test_claims_that_never_reach_the_checker_have_no_evidence(self):
        item = {'number': '3.20', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
                'label': '净利润', 'phrase': '第四季度净利润', 'year': 2024, 'scope': '合并'}
        verdict = self.verdict('第四季度净利润3.20亿元。', item)
        self.assertEqual(verdict['verdict'], 'unsupported')
        self.assertNotIn('evidence', verdict)


if __name__ == '__main__':
    unittest.main()
