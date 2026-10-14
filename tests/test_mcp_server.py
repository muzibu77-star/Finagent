"""Tests for the MCP server: stdio, the four tools, the cutoff refusal."""

import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.data.library import SCHEMA, add_page, digits

ROOT = Path(__file__).resolve().parent.parent
TEXT = '2024年营业收入1,745.44亿元；2024年营业收入1,741.44亿元。'
ANSWER = json.dumps([
    {'number': '1,745.44', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
     'label': '营业收入', 'phrase': '营业收入', 'year': 2024, 'scope': '合并'},
    {'number': '1,741.44', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
     'label': '营业收入', 'phrase': '营业收入', 'year': 2024, 'scope': '合并'},
], ensure_ascii=False)


class StubModel(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint that always answers with the same claims."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get('Content-Length') or 0))
        body = json.dumps({
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
            'choices': [{'message': {'content': ANSWER}, 'finish_reason': 'stop'}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


def build_library(path: Path) -> None:
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    for key, year, published in (('r2024', 2024, '2025-03-30'), ('r2025', 2025, '2026-03-30')):
        db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                   (key, 'annual', '某公司', '600000', year, '年度报告', 'http://x', published,
                    'h', 1))
        add_page(db, key, 1, f'{key}的第1页原文 营业收入')
    db.execute('INSERT INTO facts (document_id, page, start, end, box, text, digits, row_label, '
               'column_header, title, unit, scale, scope) VALUES (?, 1, 0, 1, ?, ?, ?, ?, ?, ?, '
               '?, ?, ?)', ('r2024', json.dumps([]), '174,144,069,958.25',
                            digits('174,144,069,958.25'), '一、营业收入', '2024年度',
                            '合并利润表', '元', '1', '合并'))
    db.commit()
    db.close()


class McpServerTest(unittest.IsolatedAsyncioTestCase):
    """The server started by a real MCP client over stdio."""

    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.library = root / 'library.sqlite'
        build_library(self.library)
        self.data_dir = root / 'data'
        self.model = ThreadingHTTPServer(('127.0.0.1', 0), StubModel)
        threading.Thread(target=self.model.serve_forever, kwargs={'poll_interval': 0.02},
                         daemon=True).start()

    async def asyncTearDown(self):
        self.model.shutdown()
        self.model.server_close()
        self.directory.cleanup()

    def parameters(self) -> StdioServerParameters:
        url = f'http://127.0.0.1:{self.model.server_address[1]}/v1/chat/completions'
        env = dict(os.environ, NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
        return StdioServerParameters(
            command=sys.executable,
            args=['-m', 'src.service.mcp_server', '--library', str(self.library),
                  '--data-dir', str(self.data_dir), '--model', 'qwen3.5-9b-extract',
                  '--url', url, '--budget-usd', '0'],
            env=env, cwd=str(ROOT))

    async def test_the_server_lists_and_answers_the_tools(self):
        async with stdio_client(self.parameters()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                self.assertEqual([tool.name for tool in tools.tools],
                                 ['check_draft', 'find_number', 'search_pages', 'page_text'])
                found = json.loads((await session.call_tool(
                    'find_number', {'text': '174,144,069,958.25', 'code': '600000',
                                    'as_of': '2025-04-30'})).content[0].text)
                self.assertEqual(found[0]['row_label'], '一、营业收入')
                self.assertEqual(json.loads((await session.call_tool(
                    'search_pages', {'query': '营业收入', 'code': '600000',
                                     'as_of': '2025-04-30'})).content[0].text),
                    [{'document_id': 'r2024', 'page': 1, 'publisher': '某公司',
                      'title': '年度报告', 'published': '2025-03-30', 'kind': 'annual'}])
                page = json.loads((await session.call_tool(
                    'page_text', {'document_id': 'r2024', 'page': 1,
                                  'as_of': '2025-04-30'})).content[0].text)
                self.assertTrue(page['text'].startswith('r2024的第1页原文'))
                refused = json.loads((await session.call_tool(
                    'page_text', {'document_id': 'r2025', 'page': 1,
                                  'as_of': '2025-04-30'})).content[0].text)
                self.assertIn('之后才披露', refused['error'])

    async def test_check_draft_runs_the_flow_through_the_model(self):
        async with stdio_client(self.parameters()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                paper = json.loads((await session.call_tool(
                    'check_draft', {'code': '600000', 'fiscal_year': 2024,
                                    'as_of': '2025-04-30', 'text': TEXT})).content[0].text)
                self.assertEqual(paper['review']['company'], '某公司')
                self.assertEqual(paper['summary'], {'numbers': 2, 'consistent': 1,
                                                    'inconsistent': 1, 'unsupported': 0,
                                                    'ambiguous': 0, 'missed': 0,
                                                    'agent_flagged': 0,
                                                    'agent_hint_consistent': 0})
                self.assertEqual(paper['errors'], [])

    async def test_check_draft_refuses_a_year_without_a_report(self):
        async with stdio_client(self.parameters()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                answer = json.loads((await session.call_tool(
                    'check_draft', {'code': '600000', 'fiscal_year': 2030,
                                    'as_of': '2025-04-30', 'text': TEXT})).content[0].text)
                self.assertIn('出处库中没有发行人 600000 的 2030 年年度报告', answer['error'])


if __name__ == '__main__':
    unittest.main()
