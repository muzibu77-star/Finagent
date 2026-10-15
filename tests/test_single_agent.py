"""Tests for the single-agent control group: tools, tool round trip, verdicts."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.llm import ApiChatModel, message_payload, answer_message
from src.agent.single import build_tools, review_draft_single
from src.data.library import SCHEMA, Library, digits, terms
from src.model.api_client import ApiClient

# A second page of ``r2024``: a small table with a unit line above it, so the
# keyword excerpt has a line above and below the hit to show.
TABLE_PAGE = ('2.1 主要会计数据和财务指标\r\n（人民币百万元）\r\n营业收入 1,745.44 1,600.00\r\n'
              '营业成本 900.00 850.00\r\n管理费用 300.00 280.00\r\n净利润 200.00 180.00')


def build_library(path: Path) -> None:
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               ('r2024', 'annual', '某公司', '600000', 2024, '年度报告', 'http://x',
                '2025-03-30', 'h', 1))
    db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               ('r2025', 'annual', '某公司', '600000', 2025, '年度报告', 'http://y',
                '2026-03-30', 'h', 1))
    for key in ('r2024', 'r2025'):
        db.execute('INSERT INTO pages VALUES (?, 1, ?)', (key, f'{key}的第1页原文'))
    db.execute('INSERT INTO pages VALUES (?, 2, ?)', ('r2024', TABLE_PAGE))
    db.execute('INSERT INTO page_terms VALUES (?, ?, ?)',
               (' '.join(terms(TABLE_PAGE)), 'r2024', 2))
    db.execute('INSERT INTO facts (document_id, page, start, end, box, text, digits, row_label, '
               'column_header, title, unit, scale, scope) VALUES (?, 2, 0, 1, ?, ?, ?, ?, ?, ?, '
               '?, ?, ?)', ('r2024', json.dumps([]), '900.00', digits('900.00'), '营业成本',
                            '2024年度', '2.1 主要会计数据和财务指标', '百万元', '1000000', '合并'))
    db.execute('INSERT INTO facts (document_id, page, start, end, box, text, digits, row_label, '
               'column_header, title, unit, scale, scope) VALUES (?, 1, 0, 1, ?, ?, ?, ?, ?, ?, '
               '?, ?, ?)', ('r2024', json.dumps([]), '174,144,069,958.25',
                            digits('174,144,069,958.25'), '一、营业收入', '2024年度',
                            '合并利润表', '元', '1', '合并'))
    db.commit()
    db.close()


def payload(content='', tool_calls=None, prompt_tokens=10, completion_tokens=5):
    message = {'content': content}
    if tool_calls:
        message['tool_calls'] = tool_calls
    return {'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens},
            'choices': [{'message': message, 'finish_reason': 'stop'}]}


def row(content='', tool_calls=None):
    return {'content': content, 'tool_calls': tool_calls, 'usage': {'prompt_tokens': 0,
                                                                   'completion_tokens': 0}}


def call(name, arguments):
    return {'id': 'call_1', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}


class MessageTest(unittest.TestCase):
    def test_an_answer_becomes_a_message_with_decoded_tool_calls(self):
        message = answer_message(row(tool_calls=[call('find_number', {'text': '1'})]))
        self.assertEqual(message.tool_calls,
                         [{'name': 'find_number', 'args': {'text': '1'}, 'id': 'call_1',
                           'type': 'tool_call'}])
        self.assertEqual(answer_message(row()).tool_calls, [])

    def test_tool_calls_and_tool_results_travel_back_in_the_request(self):
        assistant = AIMessage(content='', tool_calls=[{'name': 'find_number',
                                                       'args': {'text': '1'},
                                                       'id': 'call_1', 'type': 'tool_call'}])
        self.assertEqual(message_payload(assistant)['tool_calls'],
                         [{'id': 'call_1', 'type': 'function',
                           'function': {'name': 'find_number', 'arguments': '{"text": "1"}'}}])
        result = ToolMessage(content='[]', tool_call_id='call_1')
        self.assertEqual(message_payload(result),
                         {'role': 'tool', 'tool_call_id': 'call_1', 'content': '[]'})
        self.assertEqual(message_payload(HumanMessage(content='a')),
                         {'role': 'user', 'content': 'a'})

    def test_bound_tools_are_sent_in_the_openai_format(self):
        sent = []
        client = ApiClient('deepseek-flash', 1.0, Path(tempfile.mkdtemp()) / 'c.jsonl',
                           send=lambda body: sent.append(body) or payload('好'))
        model = ApiChatModel(client=client)
        tools = build_tools(self.library(), '600000', '2025-04-30')
        model.bind_tools(tools).invoke([HumanMessage(content='a')])
        self.assertEqual(sent[0]['tools'], [convert_to_openai_tool(tool) for tool in tools])
        self.assertEqual([tool['function']['name'] for tool in sent[0]['tools']],
                         ['find_number', 'search_pages', 'page_text', 'calculate'])
        model.invoke([HumanMessage(content='a')])
        self.assertNotIn('tools', sent[1])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'library.sqlite'
        build_library(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def library(self) -> Path:
        return self.path

    def tools(self, as_of='2025-04-30'):
        return {tool.name: tool for tool in build_tools(self.path, '600000', as_of)}

    def test_the_tools_only_see_files_published_by_the_cutoff(self):
        tools = self.tools()
        facts = json.loads(tools['find_number'].invoke({'text': '1,741.44'}))
        self.assertEqual(facts, [])  # written as 174,144,069,958.25, not 1,741.44
        self.assertEqual(json.loads(tools['find_number'].invoke({'text': '174,144,069,958.25'}))[0]
                         ['row_label'], '一、营业收入')
        self.assertEqual(tools['page_text'].invoke({'document_id': 'r2024', 'page': 1}),
                         'r2024的第1页原文')
        self.assertEqual(tools['page_text'].invoke({'document_id': 'r2025', 'page': 1}),
                         '该文件在写稿截止日之后才披露，不能作为依据。')
        self.assertEqual(tools['page_text'].invoke({'document_id': 'nope', 'page': 1}),
                         '没有这份文件。')

    def test_a_keyword_brings_back_its_lines_with_the_page_header(self):
        tools = self.tools()
        text = tools['page_text'].invoke({'document_id': 'r2024', 'page': 2,
                                          'keyword': '营业成本'})
        self.assertIn('表题：2.1 主要会计数据和财务指标', text)
        self.assertIn('单位声明：（人民币百万元）', text)
        self.assertIn('营业收入 1,745.44 1,600.00', text)  # the line above the hit
        self.assertIn('营业成本 900.00 850.00', text)
        self.assertIn('管理费用 300.00 280.00', text)  # the line below the hit
        self.assertNotIn('净利润 200.00', text)
        missed = tools['page_text'].invoke({'document_id': 'r2024', 'page': 2,
                                           'keyword': '研发费用'})
        self.assertIn('未找到“研发费用”', missed)
        self.assertIn('2.1 主要会计数据和财务指标', missed)  # the first lines of the page

    def test_search_returns_the_file_page_table_and_a_short_snippet(self):
        hits = json.loads(self.tools()['search_pages'].invoke({'query': '营业成本'}))
        self.assertEqual([(hit['document_id'], hit['page'], hit['table']) for hit in hits],
                         [('r2024', 2, '2.1 主要会计数据和财务指标')])
        self.assertEqual(hits[0]['document'], '年度报告')
        self.assertIn('营业成本 900.00', hits[0]['snippet'])
        self.assertLessEqual(len(hits[0]['snippet']), 80)

    def test_calculate_uses_decimal_and_refuses_anything_else(self):
        tools = self.tools()
        self.assertEqual(tools['calculate'].invoke(
            {'expression': '(1741.44-1556.03)/1556.03*100'}), '11.91558003380397550175767819')
        self.assertIn('算不了', tools['calculate'].invoke({'expression': '1/0'}))
        self.assertIn('算不了', tools['calculate'].invoke({'expression': "__import__('os')"}))

    def test_the_agent_checks_a_number_with_a_tool_and_reports_its_verdict(self):
        sent = []

        def send(body):
            sent.append(body)
            if any(message['role'] == 'tool' for message in body['messages']):
                return payload(json.dumps([{'number': '1,745.44', 'occurrence': 1,
                                            'verdict': '正确'}], ensure_ascii=False))
            return payload(tool_calls=[call('find_number', {'text': '1,745.44'})])

        draft = {'text': '2024年营业收入1,745.44亿元。', 'code': '600000', 'company': '某公司',
                 'fiscal_year': 2024, 'as_of': '2025-04-30'}
        with mock.patch('src.agent.single.ApiClient',
                        lambda model, budget, cache, url=None: ApiClient(
                            model, budget, cache, send=send)):
            run = review_draft_single(draft, library=self.path, model='deepseek-flash',
                                      url='https://api.deepseek.com/chat/completions',
                                      budget_usd=1.0, cache=Path(tempfile.mkdtemp()) / 'c.jsonl')
        self.assertEqual(run['verdicts'], {9: {'verdict': 'consistent'}})
        self.assertEqual(run['tool_calls'], 1)
        self.assertEqual(run['usage'], {'prompt_tokens': 20, 'completion_tokens': 10})
        self.assertEqual([message['type'] for message in run['messages']],
                         ['human', 'ai', 'tool', 'ai'])
        self.assertEqual(run['messages'][2]['content'], '[]')
        self.assertEqual([message['role'] for message in sent[1]['messages']],
                         ['system', 'user', 'assistant', 'tool'])
        self.assertEqual(sent[1]['messages'][2]['tool_calls'][0]['function']['name'],
                         'find_number')
        self.assertEqual(json.loads(sent[1]['messages'][2]['tool_calls'][0]['function']
                                    ['arguments']), {'text': '1,745.44'})


if __name__ == '__main__':
    unittest.main()
