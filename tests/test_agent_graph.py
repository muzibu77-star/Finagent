"""Tests for the LangGraph review graph: extract, check, consistency, revise, report."""

import json
import sqlite3
import unittest

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from src.agent.graph import build_graph, numbers_in, sentence_at, workpaper_markdown
from src.data.library import SCHEMA, digits
from src.verify.checker import Checker


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
    document(db, 'r2024', 'annual', 2024, '2025-03-30')
    fact(db, 'r2024', '一、营业收入', '2024年度', '174,144,069,958.25')
    fact(db, 'r2024', '一、营业收入', '2023年度', '150,560,330,316.45')
    return Checker(db)


TEXT = '2024年营业收入1,745.44亿元；2024年营业收入1,741.44亿元。第四季度净利润3.20亿元。'
ITEMS = [
    {'number': '1,745.44', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
     'label': '营业收入', 'phrase': '营业收入', 'year': 2024, 'scope': '合并'},
    {'number': '1,741.44', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
     'label': '营业收入', 'phrase': '营业收入', 'year': 2024, 'scope': '合并'},
    {'number': '3.20', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
     'label': '净利润', 'phrase': '第四季度净利润', 'year': 2024, 'scope': '合并'},
]
REVISION = '{"sentence": "2024年营业收入1,741.44亿元"}'


def run(*replies: str) -> dict:
    """The workpaper of one fake-model run: the extraction answer then the revisions."""
    llm = FakeMessagesListChatModel(responses=[AIMessage(content=reply) for reply in replies])
    state = build_graph(build(), llm).invoke(
        {'text': TEXT, 'code': '600000', 'company': '某公司', 'fiscal_year': 2024,
         'as_of': '2025-04-30'})
    return state['workpaper']


class ReviewGraphTest(unittest.TestCase):
    def rows(self, paper: dict) -> dict[int, dict]:
        return {row['start']: row for row in paper['numbers']}

    def test_it_reads_the_checkable_numbers_of_the_draft(self):
        self.assertEqual([number['text'] for number in numbers_in(TEXT)],
                         ['1,745.44亿元', '1,741.44亿元', '3.20亿元'])

    def test_a_sentence_carries_no_paragraph_break(self):
        self.assertEqual(sentence_at('第一段。\n第二段1.5亿元；', 8), '第二段1.5亿元')
        self.assertEqual(sentence_at('没有分隔符的句子1.5亿元', 9), '没有分隔符的句子1.5亿元')

    def test_summary_counts_every_verdict(self):
        paper = run(json.dumps(ITEMS, ensure_ascii=False), REVISION)
        self.assertEqual(paper['summary'], {'numbers': 3, 'consistent': 1, 'inconsistent': 1,
                                            'unsupported': 1, 'ambiguous': 0, 'missed': 0})
        rows = self.rows(paper)
        self.assertEqual(rows[9]['verdict'], 'inconsistent')
        self.assertEqual(rows[9]['error'], 'digit')
        self.assertEqual(rows[9]['expected'], '1741.44')

    def test_revision_is_accepted_when_only_the_number_changes(self):
        paper = run(json.dumps(ITEMS, ensure_ascii=False), REVISION)
        self.assertEqual(self.rows(paper)[9]['suggestion'],
                         {'accepted': True, 'sentence': '2024年营业收入1,741.44亿元'})

    def test_a_revision_that_changes_more_is_discarded(self):
        for reply in ('{"sentence": "营业收入为1,741.44亿元。"}', '不是 JSON'):
            paper = run(json.dumps(ITEMS, ensure_ascii=False), reply)
            row = self.rows(paper)[9]
            self.assertEqual(row['suggestion'], {'accepted': False, 'sentence': None,
                                                 'note': '需人工改写'})
            self.assertEqual(row['verdict'], 'inconsistent')

    def test_one_metric_written_twice_is_found(self):
        paper = run(json.dumps(ITEMS, ensure_ascii=False), REVISION)
        rows = self.rows(paper)
        self.assertEqual(rows[9]['consistency'],
                         {'intra_document': True, 'positions': [9, 29],
                          'numbers': ['1,745.44', '1,741.44']})
        self.assertEqual(rows[29]['consistency']['positions'], [9, 29])
        self.assertIsNone(rows[47]['consistency'])

    def test_workpaper_resolves_evidence_facts(self):
        paper = run(json.dumps(ITEMS, ensure_ascii=False), REVISION)
        rows = self.rows(paper)
        self.assertEqual(rows[29]['evidence'], [{
            'id': 1, 'document': '年度报告', 'document_id': 'r2024', 'published': '2025-03-30',
            'page': 1, 'start': 0, 'end': 1, 'row_label': '一、营业收入',
            'column_header': '2024年度', 'unit': '元'}])
        self.assertEqual(rows[9]['evidence'][0]['row_label'], '一、营业收入')
        self.assertEqual(rows[47]['evidence'], [])

    def test_markdown_reports_the_table_and_the_summary(self):
        markdown = workpaper_markdown(run(json.dumps(ITEMS, ensure_ascii=False), REVISION))
        self.assertIn('| 不一致 | 1 |', markdown)
        self.assertIn('| 抽取遗漏需人工 | 0 |', markdown)
        self.assertIn('一、营业收入／2024年度／元', markdown)
        self.assertIn('2024年营业收入1,741.44亿元', markdown)


if __name__ == '__main__':
    unittest.main()
