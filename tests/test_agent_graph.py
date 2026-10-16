"""Tests for the LangGraph review graph: extract, check, investigate, revise, report."""

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from src.agent.llm import ApiChatModel
from src.agent.graph import (build_graph, handover, numbers_in, review_draft, review_events,
                             sentence_at, tool_calls, workpaper_markdown)
from src.agent.recheck import EXPECTED_VALUE, SAME_VALUE
from src.data.library import SCHEMA, digits
from src.model.api_client import ApiClient, pieces
from src.verify.checker import Checker
from src.verify.extract import PROMPT, line_items


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


def agent(items: list[dict]):
    """A stand-in investigate node that answers with ``items`` and records its calls."""
    calls: list = []

    def investigate(draft: dict, handed: list[dict]) -> dict:
        calls.append((draft, handed))
        return {'items': items, 'answer': json.dumps(items, ensure_ascii=False)}

    investigate.calls = calls
    return investigate


def cited(key: int, text: str, start: int) -> dict:
    """One ``find_number`` result as a run records it, for a stubbed investigate node."""
    return {'id': key, 'document_id': 'r2024', 'page': 1, 'start': start, 'end': start + 1,
            'text': text, 'unit': '元', 'scale': '1', 'row_label': '一、营业收入',
            'column_header': '2024年度', 'title': '合并利润表'}


def run_investigated(investigator, *replies: str) -> dict:
    """The workpaper of one fake-model run whose investigate node is stubbed."""
    llm = FakeMessagesListChatModel(responses=[AIMessage(content=reply) for reply in replies])
    state = build_graph(build(), llm, investigator=investigator).invoke(
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
                                            'unsupported': 1, 'ambiguous': 0, 'missed': 0,
                                            'agent_flagged': 0, 'agent_hint_consistent': 0})
        self.assertEqual(paper['errors'], [])
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


class InvestigateTest(unittest.TestCase):
    """The optional node that re-opens the numbers the checker hands to a person."""

    def rows(self, paper: dict) -> dict[int, dict]:
        return {row['start']: row for row in paper['numbers']}

    def test_only_the_numbers_without_a_checker_verdict_are_handed_over(self):
        self.assertEqual(handover({}, TEXT), [
            {'number': '1,745.44', 'occurrence': 1, 'start': 9, 'text': '1,745.44亿元'},
            {'number': '1,741.44', 'occurrence': 1, 'start': 29, 'text': '1,741.44亿元'},
            {'number': '3.20', 'occurrence': 1, 'start': 47, 'text': '3.20亿元'}])

    def test_the_agent_lists_only_the_handed_numbers_and_flags_a_wrong_one(self):
        fake = agent([{'number': '3.20', 'occurrence': 1, 'verdict': '错误', 'expected': '3.10',
                       'basis': '某公司2024年年报第1页：第四季度净利润310,000,000元'}])
        paper = run_investigated(fake, json.dumps(ITEMS, ensure_ascii=False), REVISION)
        self.assertEqual(fake.calls[0][1], [{'number': '3.20', 'occurrence': 1, 'start': 47,
                                             'text': '3.20亿元'}])
        row = self.rows(paper)[47]
        self.assertEqual(row['verdict'], 'inconsistent')
        self.assertEqual(row['source'], 'agent')
        self.assertEqual(row['expected'], '3.10')
        self.assertEqual(row['agent'], {'verdict': 'inconsistent', 'expected': '3.10',
                                        'basis': '某公司2024年年报第1页：第四季度净利润310,000,000元'})
        self.assertEqual(row['suggestion'], {})  # the revision node leaves it alone
        self.assertEqual(paper['summary']['agent_flagged'], 1)
        self.assertEqual(paper['summary']['consistent'], 1)  # the agent confirms nothing
        self.assertEqual(self.rows(paper)[9]['suggestion'],
                         {'accepted': True, 'sentence': '2024年营业收入1,741.44亿元'})
        self.assertIn('Agent 认为有误：应为 3.10', workpaper_markdown(paper))

    def test_a_consistent_agent_verdict_is_only_a_hint(self):
        fake = agent([{'number': '3.20', 'occurrence': 1, 'verdict': '正确', 'expected': '',
                       'basis': ''}])
        paper = run_investigated(fake, json.dumps(ITEMS, ensure_ascii=False), REVISION)
        row = self.rows(paper)[47]
        self.assertEqual(row['verdict'], 'unsupported')
        self.assertEqual(row['source'], 'checker')
        self.assertEqual(row['agent'], {'verdict': 'consistent', 'expected': None, 'basis': None})
        self.assertEqual(row['suggestion'], {})
        self.assertEqual(paper['summary']['agent_flagged'], 0)
        self.assertEqual(paper['summary']['agent_hint_consistent'], 1)
        self.assertEqual(paper['summary']['consistent'], 1)
        self.assertIn('Agent 认为一致', workpaper_markdown(paper))

    def test_a_verdict_on_a_number_the_checker_settled_is_ignored(self):
        fake = agent([{'number': '1,741.44', 'occurrence': 1, 'verdict': '错误',
                       'expected': '1,700.00', 'basis': 'x'},
                      {'number': '3.20', 'occurrence': 1, 'verdict': '无法判断'}])
        paper = run_investigated(fake, json.dumps(ITEMS, ensure_ascii=False), REVISION)
        rows = self.rows(paper)
        self.assertEqual(rows[29]['verdict'], 'consistent')
        self.assertIsNone(rows[29]['agent'])
        self.assertEqual(rows[47]['verdict'], 'unsupported')
        self.assertIsNone(rows[47]['agent'])
        self.assertEqual(paper['summary']['agent_flagged'], 0)
        self.assertEqual(paper['summary']['agent_hint_consistent'], 0)

    def test_a_failing_agent_changes_no_verdict(self):
        def broken(draft: dict, handed: list[dict]) -> dict:
            raise RuntimeError('没有密钥')

        paper = run_investigated(broken, json.dumps(ITEMS, ensure_ascii=False), REVISION)
        self.assertEqual(self.rows(paper)[47]['verdict'], 'unsupported')
        self.assertEqual(paper['errors'], ['investigate failed: RuntimeError: 没有密钥'])

    def test_an_unparsable_answer_changes_no_verdict(self):
        def mute(draft: dict, handed: list[dict]) -> dict:
            return {'items': [], 'answer': '抱歉，我看不懂。'}

        paper = run_investigated(mute, json.dumps(ITEMS, ensure_ascii=False), REVISION)
        self.assertEqual(self.rows(paper)[47]['verdict'], 'unsupported')
        self.assertEqual(len(paper['errors']), 1)
        self.assertIn('unparsable investigate answer', paper['errors'][0])


class BackfillTest(unittest.TestCase):
    """The facts the agent's own lookups turned up, beside the verdict they never touch.

    The stubbed node answers one number the checker handed to a person (3.20亿元)
    and returns the facts its ``find_number`` calls would have read: the draft's
    figure, or the value the answer expects, as the tables write them in 元.
    """

    def rows(self, paper: dict) -> dict[int, dict]:
        return {row['start']: row for row in paper['numbers']}

    def one(self, item: dict, facts: list[dict]) -> dict:
        def fake(draft: dict, handed: list[dict]) -> dict:
            return {'items': [item], 'facts': facts, 'answer': ''}

        return run_investigated(fake, json.dumps(ITEMS, ensure_ascii=False), REVISION)

    def test_a_wrong_number_carries_the_fact_with_the_value_it_expects(self):
        paper = self.one({'number': '3.20', 'occurrence': 1, 'verdict': '错误',
                          'expected': '3.10亿元', 'basis': 'x'},
                         [cited(7, '310,000,000.00', 40)])
        row = self.rows(paper)[47]
        self.assertEqual((row['verdict'], row['source']), ('inconsistent', 'agent'))
        fact = row['agent']['evidence'][0]
        self.assertEqual((fact['id'], fact['page'], fact['start'], fact['end']),
                         (7, 1, 40, 41))
        self.assertEqual((fact['text'], fact['unit'], fact['mark']),
                         ('310,000,000.00', '元', EXPECTED_VALUE))
        self.assertIn('原文应有值 1 条', workpaper_markdown(paper))
        self.assertEqual(paper['summary']['agent_flagged'], 1)

    def test_a_hint_carries_the_fact_with_the_drafted_number(self):
        paper = self.one({'number': '3.20', 'occurrence': 1, 'verdict': '正确'},
                         [cited(7, '320,000,000.00', 40)])
        row = self.rows(paper)[47]
        self.assertEqual((row['verdict'], row['source']), ('unsupported', 'checker'))
        self.assertEqual([item['mark'] for item in row['agent']['evidence']], [SAME_VALUE])
        self.assertEqual(paper['summary']['agent_hint_consistent'], 1)
        self.assertIn('Agent 认为一致，仍需人工确认（原文中有同值，请核对科目 1 条）',
                      workpaper_markdown(paper))

    def test_the_attached_facts_change_no_verdict(self):
        item = {'number': '3.20', 'occurrence': 1, 'verdict': '错误', 'expected': '3.20亿元',
                'basis': '某公司2024年年报第1页：净利润320,000,000.00元'}
        plain = self.one(item, [])
        backed = self.one(item, [cited(7, '320,000,000.00', 40), cited(8, '1,000.00', 60)])
        self.assertEqual([(row['start'], row['verdict'], row['source'])
                          for row in plain['numbers']],
                         [(row['start'], row['verdict'], row['source'])
                          for row in backed['numbers']])
        self.assertEqual(plain['summary'], backed['summary'])
        self.assertEqual(self.rows(plain)[47]['agent'],
                         {'verdict': 'inconsistent', 'expected': '3.20亿元',
                          'basis': '某公司2024年年报第1页：净利润320,000,000.00元'})
        self.assertNotIn(SAME_VALUE, workpaper_markdown(plain))

    def test_a_number_no_fact_carries_gets_no_evidence(self):
        paper = self.one({'number': '3.20', 'occurrence': 1, 'verdict': '正确'},
                         [cited(7, '321,000,000.00', 40)])
        self.assertNotIn('evidence', self.rows(paper)[47]['agent'])

    def test_a_ratio_is_not_searched_at_all(self):
        paper = self.one({'number': '3.20', 'occurrence': 1, 'verdict': '正确'},
                         [cited(7, '3.20', 40)])
        self.assertNotIn('evidence', self.rows(paper)[47]['agent'])

    def test_an_investigator_that_reports_no_facts_keeps_the_older_shape(self):
        def fake(draft: dict, handed: list[dict]) -> dict:
            return {'items': [{'number': '3.20', 'occurrence': 1, 'verdict': '正确'}],
                    'answer': ''}

        paper = run_investigated(fake, json.dumps(ITEMS, ensure_ascii=False), REVISION)
        self.assertEqual(self.rows(paper)[47]['agent'], {'verdict': 'consistent',
                                                         'expected': None, 'basis': None})


class ReviewEventsTest(unittest.TestCase):
    """The progress events of one run: one per node, in order, with its highlight."""

    WRONG = [{'number': '3.20', 'occurrence': 1, 'verdict': '错误', 'expected': '3.10',
              'basis': '某公司2024年年报第1页：第四季度净利润310,000,000元'}]
    MESSAGES = [{'type': 'ai', 'content': '', 'tool_calls': [
        {'name': 'find_number', 'args': {'text': '3.10'}}]},
        {'type': 'tool', 'content': '[{"text": "310,000,000.00", "unit": "元"}]',
         'tool_call_id': 'a'},
        {'type': 'ai', 'content': '[]'}]

    def events(self, items: list[dict] | None = None,
               messages: list[dict] | None = None) -> list[dict]:
        def stub(draft: dict, handed: list[dict]) -> dict:
            answer = items if items is not None else self.WRONG
            return {'items': answer, 'answer': json.dumps(answer, ensure_ascii=False),
                    'messages': messages or [],
                    'facts': [cited(7, '310,000,000.00', 100)]}

        llm = FakeMessagesListChatModel(responses=[
            AIMessage(content=json.dumps(ITEMS, ensure_ascii=False)), AIMessage(content=REVISION)])
        return list(review_events(build_graph(build(), llm, investigator=stub), dict(DRAFT)))

    def test_one_event_per_node_in_the_order_the_nodes_run(self):
        events = self.events()
        self.assertEqual([event['node'] for event in events],
                         ['extract', 'check', 'investigate', 'revise', 'report'])
        self.assertEqual([event['label'] for event in events],
                         ['抽取 Agent', '核查器', '复查 Agent', '修订 Agent', '底稿'])
        for event in events:
            self.assertRegex(event['at'], r'^\d\d:\d\d:\d\d$')
            self.assertGreaterEqual(event['seconds'], 0.0)

    def test_each_event_carries_the_highlight_of_its_node(self):
        first, second, third, fourth, last = self.events()
        self.assertEqual(first['detail']['extracted'], 3)
        self.assertEqual(second['detail']['conclusions'], {
            '一致': 1, '不一致': 1, '无依据': 1, '存疑': 0, '抽取遗漏需人工': 0})
        self.assertEqual({key: third['detail'][key] for key in ('handed', 'judged', 'flagged')},
                         {'handed': 1, 'judged': 1, 'flagged': 1})
        self.assertEqual(third['detail']['tools'], [])
        self.assertEqual({key: fourth['detail'][key] for key in ('advised', 'accepted')},
                         {'advised': 1, 'accepted': 1})
        self.assertEqual(last['detail'], {'numbers': 3})

    def test_each_event_carries_the_entries_a_replay_needs(self):
        first, second, third, fourth, _ = self.events()
        self.assertEqual([entry['start'] for entry in first['detail']['items']], [9, 29, 47])
        self.assertEqual([entry['number'] for entry in first['detail']['items']],
                         ['1,745.44', '1,741.44', '3.20'])
        self.assertEqual([entry['label'] for entry in first['detail']['items']],
                         ['营业收入', '营业收入', '净利润'])
        self.assertEqual([entry['verdict'] for entry in second['detail']['numbers']],
                         ['inconsistent', 'consistent', 'unsupported'])
        self.assertEqual([entry['text'] for entry in third['detail']['numbers']], ['3.20亿元'])
        self.assertEqual(third['detail']['numbers'][0]['verdict'], 'inconsistent')
        self.assertEqual(third['detail']['numbers'][0]['evidence'][0]['mark'],
                         '原文应有值')
        self.assertEqual([entry['sentence'] for entry in fourth['detail']['numbers']],
                         ['2024年营业收入1,745.44亿元'])
        self.assertEqual([entry['accepted'] for entry in fourth['detail']['numbers']], [True])
        self.assertEqual(fourth['detail']['numbers'][0]['rewritten'],
                         '2024年营业收入1,741.44亿元')

    def test_the_agent_event_lists_the_tool_calls_of_the_run(self):
        events = self.events(messages=self.MESSAGES)
        self.assertEqual(events[2]['detail']['tools'],
                         [{'name': 'find_number', 'args': {'text': '3.10'},
                           'result': '[{"text": "310,000,000.00", "unit": "元"}]'}])

    def test_a_number_the_checker_settles_is_not_counted_as_handed_over(self):
        events = self.events(items=[])
        self.assertEqual(events[2]['detail']['handed'], 1)

    def test_only_the_last_event_carries_the_workpaper(self):
        events = self.events()
        self.assertEqual(['workpaper' in event for event in events], [False] * 4 + [True])
        self.assertEqual(len(events[4]['workpaper']['numbers']), 3)

    def test_a_tool_result_longer_than_a_trace_keeps_is_cut(self):
        calls = tool_calls([{'type': 'ai', 'content': '', 'tool_calls': [
            {'name': 'search_pages', 'args': {'query': '研发'}}]},
            {'type': 'tool', 'content': 'x' * 200, 'tool_call_id': 'a'}])
        self.assertEqual(calls, [{'name': 'search_pages', 'args': {'query': '研发'},
                                  'result': 'x' * 160 + '…'}])


def library_file(directory: str) -> Path:
    """The little ``build()`` library as a file, as ``review_draft`` opens it."""
    path = Path(directory) / 'library.sqlite'
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    document(db, 'r2024', 'annual', 2024, '2025-03-30')
    fact(db, 'r2024', '一、营业收入', '2024年度', '174,144,069,958.25')
    fact(db, 'r2024', '一、营业收入', '2023年度', '150,560,330,316.45')
    db.commit()
    db.close()
    return path


DRAFT = {'text': TEXT, 'code': '600000', 'company': '某公司', 'fiscal_year': 2024,
         'as_of': '2025-04-30'}


class NodeModelTest(unittest.TestCase):
    """The extraction model and the agent model are separate model objects."""

    def paper(self, llm, agent_llm=None) -> dict:
        state = build_graph(build(), llm, agent_llm=agent_llm).invoke(dict(DRAFT))
        return state['workpaper']

    def rows(self, paper: dict) -> dict[int, dict]:
        return {row['start']: row for row in paper['numbers']}

    def test_the_revision_node_answers_with_the_agent_model(self):
        llm, agent_llm = Model(json.dumps(ITEMS, ensure_ascii=False)), Model(REVISION)
        paper = self.paper(llm, agent_llm)
        self.assertEqual((llm.calls, agent_llm.calls), (1, 1))
        self.assertEqual(self.rows(paper)[9]['suggestion'],
                         {'accepted': True, 'sentence': '2024年营业收入1,741.44亿元'})

    def test_without_an_agent_model_every_node_uses_the_extraction_model(self):
        llm = Model(json.dumps(ITEMS, ensure_ascii=False), REVISION)
        paper = self.paper(llm)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(self.rows(paper)[9]['suggestion']['accepted'], True)

    def test_the_investigate_node_still_uses_the_stub_that_was_given(self):
        fake = agent([{'number': '3.20', 'occurrence': 1, 'verdict': '错误', 'expected': '3.10'}])
        state = build_graph(build(), Model(json.dumps(ITEMS, ensure_ascii=False), REVISION),
                            investigator=fake).invoke(dict(DRAFT))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(state['workpaper']['summary']['agent_flagged'], 1)


class ReviewDraftModelTest(unittest.TestCase):
    """One draft's model calls: which model, endpoint and cache each node uses."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.library = library_file(self.directory.name)
        self.bodies: list[dict] = []
        self.streams: list[dict] = []

    def tearDown(self):
        self.directory.cleanup()

    def send(self, body: dict) -> dict:
        self.bodies.append(body)
        answer = {'qwen3.5-9b-extract': json.dumps(ITEMS, ensure_ascii=False),
                  'qwen3.5-9b': REVISION}[body['model']]
        return {'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
                'choices': [{'message': {'content': answer}, 'finish_reason': 'stop'}]}

    def patched(self):
        """Review the draft through real clients whose transport is stubbed."""
        def factory(model, budget, cache, url=None):
            return ApiClient(model, budget, Path(cache), url=url, send=self.send,
                             stream=self.stream)
        return mock.patch('src.agent.graph.ApiClient', side_effect=factory)

    def stream(self, body: dict, on_delta) -> dict:
        """The stubbed streaming transport: one answer, reported as it is written."""
        self.streams.append(body)
        payload = self.send(body)
        for piece in pieces(payload['choices'][0]['message']['content']):
            on_delta(piece)
        return payload

    def review(self, **kwargs) -> dict:
        arguments = dict(library=self.library, model='qwen3.5-9b-extract',
                         url='http://127.0.0.1:18090/v1/chat/completions', budget_usd=0.0,
                         cache=Path(self.directory.name) / 'api_cache.jsonl', investigate=False)
        arguments.update(kwargs)
        return review_draft(dict(DRAFT), **arguments)

    def test_a_run_without_an_agent_model_keeps_the_older_request_hashes(self):
        extract = Path(self.directory.name) / 'api_cache.jsonl'
        with self.patched():
            paper = self.review()
        self.assertEqual(len(self.bodies), 2)  # the extraction, then the revision
        db = sqlite3.connect(self.library)
        labels = line_items(db, '600000', 2024)
        db.close()
        body = {'model': 'qwen3.5-9b-extract', 'max_tokens': 8000, 'temperature': 0,
                'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'user', 'content': PROMPT.format(
                    company='某公司', year=2024, labels='\n'.join(labels), review=TEXT)}]}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False)
                                .encode()).hexdigest()
        self.assertEqual(self.bodies[0], body)  # exactly the request the old code sent
        self.assertEqual(self.streams, [])  # 没有观察者时不开流式
        self.assertEqual(sorted(path.name for path in Path(self.directory.name).iterdir()
                                if path.name.startswith('api_cache')), ['api_cache.jsonl'])
        self.assertIn(digest, [json.loads(line)['request_sha256']
                               for line in extract.read_text().splitlines()])
        self.assertEqual({json.loads(line)['model'] for line in extract.read_text().splitlines()},
                         {'qwen3.5-9b-extract'})
        self.assertEqual(paper['summary']['inconsistent'], 1)

    @staticmethod
    def investigate(draft: dict, numbers: list[dict], *, library, model, url, budget_usd,
                    cache, on_call=None) -> dict:
        """A stubbed review agent: one lookup, one conclusion about ``3.20亿元``."""
        if on_call is not None:
            on_call('find_number', {'text': '3.10'}, '[{"text": "310,000,000.00"}]')
        items = [{'number': '3.20', 'occurrence': 1, 'verdict': '错误', 'expected': '3.10',
                  'basis': '某公司2024年年报第1页'}]
        return {'items': items, 'answer': json.dumps(items, ensure_ascii=False), 'facts': [],
                'tool_calls': 1, 'usage': {}, 'messages': []}

    def test_the_agent_reports_its_tool_calls_and_then_its_conclusions(self):
        seen: list[dict] = []
        with mock.patch('src.agent.graph.investigate_draft', side_effect=self.investigate), \
                self.patched():
            paper = self.review(on_event=seen.append, investigate=True)
        self.assertEqual([event for event in seen if event.get('event') == 'tool'],
                         [{'event': 'tool', 'name': 'find_number', 'args': {'text': '3.10'},
                           'result': '[{"text": "310,000,000.00"}]'}])
        agents = [event for event in seen if event.get('event') == 'agent']
        self.assertEqual([[event[key] for key in ('start', 'text', 'verdict', 'expected')]
                          for event in agents], [[47, '3.20亿元', 'inconsistent', '3.10']])
        self.assertIn('Agent 认为有误', agents[0]['note'])
        self.assertEqual(paper['summary']['agent_flagged'], 1)
        self.assertEqual(paper['numbers'][2]['source'], 'agent')
        # 底稿的 trace 明细也带着这次结论，回放不依赖过程事件
        detail = [event for event in paper['trace'] if event['node'] == 'investigate'][0]['detail']
        self.assertEqual([entry['start'] for entry in detail['numbers']], [47])

    def test_a_review_returns_the_workpaper_plus_its_trace(self):
        with self.patched():
            paper = self.review()
        trace = paper.pop('trace')
        self.assertEqual([event['node'] for event in trace],
                         ['extract', 'check', 'revise', 'report'])
        self.assertEqual(paper.pop('models'), {'extract': 'qwen3.5-9b-extract',
                                               'agent': 'qwen3.5-9b-extract'})
        # apart from the trace and the models, the workpaper of the same run without events
        self.assertEqual(set(paper), {'review', 'text', 'numbers', 'errors', 'summary'})
        self.assertEqual(paper['summary'], {'numbers': 3, 'consistent': 1, 'inconsistent': 1,
                                            'unsupported': 1, 'ambiguous': 0, 'missed': 0,
                                            'agent_flagged': 0, 'agent_hint_consistent': 0})
        self.assertEqual([row['verdict'] for row in paper['numbers']],
                         ['inconsistent', 'consistent', 'unsupported'])

    def test_the_streamed_events_are_the_trace_of_the_workpaper(self):
        seen: list[dict] = []
        with self.patched():
            paper = self.review(on_event=seen.append)
        # 边跑边推的过程事件也在流里，但底稿的 trace 只留每个节点的汇总事件。
        self.assertEqual([event for event in seen if event.get('node')], paper['trace'])
        self.assertEqual(''.join(event['text'] for event in seen
                                 if event.get('event') == 'delta'),
                         json.dumps(ITEMS, ensure_ascii=False))
        self.assertEqual([event['event'] for event in seen
                          if not event.get('node') and event['event'] != 'delta'],
                         ['item', 'item', 'item', 'check', 'check', 'check', 'revise'])

    def test_the_agent_model_keeps_its_own_cache(self):
        directory = Path(self.directory.name)
        agent_cache = directory / 'api_cache_qwen3.5-9b.jsonl'
        with self.patched():
            paper = self.review(agent_model='qwen3.5-9b',
                                agent_url='http://127.0.0.1:18091/v1/chat/completions',
                                agent_cache=agent_cache)
        extract, agent = directory / 'api_cache.jsonl', agent_cache
        self.assertEqual({json.loads(line)['model'] for line in extract.read_text().splitlines()},
                         {'qwen3.5-9b-extract'})
        self.assertEqual({json.loads(line)['model'] for line in agent.read_text().splitlines()},
                         {'qwen3.5-9b'})
        self.assertEqual(self.bodies[1]['model'], 'qwen3.5-9b')
        self.assertEqual(paper['summary']['inconsistent'], 1)

    def test_the_tool_agent_runs_on_the_agent_model_url_and_cache(self):
        agent_cache = Path(self.directory.name) / 'api_cache_qwen3.5-9b.jsonl'
        seen = {}

        def fake_investigate(draft: dict, handed: list[dict], **kwargs) -> dict:
            seen.update(kwargs)
            return {'items': [], 'answer': ''}

        with mock.patch('src.agent.graph.investigate_draft', fake_investigate), self.patched():
            self.review(investigate=True, agent_model='qwen3.5-9b',
                        agent_url='http://127.0.0.1:18091/v1/chat/completions',
                        agent_cache=agent_cache)
        self.assertEqual(seen['model'], 'qwen3.5-9b')
        self.assertEqual(seen['url'], 'http://127.0.0.1:18091/v1/chat/completions')
        self.assertEqual(seen['cache'], agent_cache)


class Model:
    """A fake chat model: it answers from a script or raises what the script holds."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply)


TWICE = '2024年营业收入1,745.44亿元；2024年营业收入1,744.44亿元。'
TWICE_ITEMS = [
    {'number': '1,745.44', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
     'label': '营业收入', 'phrase': '营业收入', 'year': 2024, 'scope': '合并'},
    {'number': '1,744.44', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
     'label': '营业收入', 'phrase': '营业收入', 'year': 2024, 'scope': '合并'},
]


class FailureTest(unittest.TestCase):
    """A model node that fails leaves numbers to a person and never settles one."""

    def paper(self, llm, text=TEXT) -> dict:
        state = build_graph(build(), llm).invoke(
            {'text': text, 'code': '600000', 'company': '某公司', 'fiscal_year': 2024,
             'as_of': '2025-04-30'})
        return state['workpaper']

    def verdicts(self, paper: dict) -> list[str]:
        return [row['verdict'] for row in paper['numbers']]

    def test_a_failed_extraction_leaves_every_number_to_a_person(self):
        paper = self.paper(Model(RuntimeError('网络不通')))
        self.assertEqual(paper['errors'], ['extract failed: RuntimeError: 网络不通'])
        self.assertEqual(self.verdicts(paper), ['missed', 'missed', 'missed'])
        self.assertEqual([paper['summary'][key] for key in ('consistent', 'inconsistent')],
                         [0, 0])

    def test_an_unparsable_extraction_answer_leaves_every_number_to_a_person(self):
        paper = self.paper(Model('抱歉，我读不懂。'))
        self.assertEqual(len(paper['errors']), 1)
        self.assertTrue(paper['errors'][0].startswith('unparsable extract answer'),
                        paper['errors'][0])
        self.assertEqual(self.verdicts(paper), ['missed', 'missed', 'missed'])

    def test_a_spent_budget_leaves_every_number_to_a_person(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ApiClient('deepseek-flash', 0.0, Path(directory) / 'cache.jsonl',
                               send=lambda body: None)
            paper = self.paper(ApiChatModel(client=client))
        self.assertEqual(paper['errors'],
                         ['extract failed: BudgetExceeded: spent $0.00 of $0.00'])
        self.assertEqual(self.verdicts(paper), ['missed', 'missed', 'missed'])

    def test_a_failed_revision_leaves_that_number_to_a_person_and_the_others_alone(self):
        llm = Model(json.dumps(TWICE_ITEMS, ensure_ascii=False), RuntimeError('超时'),
                    '{"sentence": "2024年营业收入1741.44亿元"}')
        paper = self.paper(llm, TWICE)
        rows = {row['start']: row for row in paper['numbers']}
        self.assertEqual(rows[9]['suggestion'],
                         {'accepted': False, 'sentence': None, 'note': '需人工改写'})
        self.assertEqual(rows[29]['suggestion'],
                         {'accepted': True, 'sentence': '2024年营业收入1741.44亿元'})
        self.assertEqual(paper['errors'], ['revise failed: RuntimeError: 超时'])
        self.assertEqual(rows[9]['verdict'], 'inconsistent')

    def test_every_number_appears_once_and_a_failure_settles_none(self):
        for llm in (Model(RuntimeError('x')), Model('乱码')):
            paper = self.paper(llm)
            self.assertEqual([row['start'] for row in paper['numbers']],
                             [number['start'] for number in numbers_in(TEXT)])
            self.assertNotIn('consistent', self.verdicts(paper))
            self.assertNotIn('inconsistent', self.verdicts(paper))


if __name__ == '__main__':
    unittest.main()
