"""Tests for the review page: routing, validation, downloads, the issuer list."""

import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from src.agent.graph import build_graph
from src.data.library import SCHEMA, digits
from src.service.app import ReviewServer
from src.verify.checker import Checker

# The page talks to a loopback server; the proxy of this machine must not see it.
os.environ.setdefault('NO_PROXY', '127.0.0.1,localhost')
os.environ.setdefault('no_proxy', '127.0.0.1,localhost')

TEXT = '2024年营业收入1,745.44亿元；2024年营业收入1,741.44亿元。'
ITEMS = [
    {'number': '1,745.44', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
     'label': '营业收入', 'phrase': '营业收入', 'year': 2024, 'scope': '合并'},
    {'number': '1,741.44', 'occurrence': 1, 'type': 'amount', 'unit': '亿元',
     'label': '营业收入', 'phrase': '营业收入', 'year': 2024, 'scope': '合并'},
]
REVISION = '{"sentence": "2024年营业收入1741.44亿元"}'


def build_library(path: Path) -> None:
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               ('r2024', 'annual', '某公司', '600000', 2024, '年度报告', 'http://x',
                '2025-03-30', 'h', 1))
    db.execute('INSERT INTO facts (document_id, page, start, end, box, text, digits, row_label, '
               'column_header, title, unit, scale, scope) VALUES (?, 1, 0, 1, ?, ?, ?, ?, ?, ?, '
               '?, ?, ?)', ('r2024', json.dumps([]), '174,144,069,958.25',
                            digits('174,144,069,958.25'), '一、营业收入', '2024年度',
                            '合并利润表', '元', '1', '合并'))
    db.commit()
    db.close()


def fake_review_draft(draft, *, library, model, url, budget_usd, cache, investigate=True) -> dict:
    """The real graph over a fake model, so an endpoint test needs no API key."""
    llm = FakeMessagesListChatModel(responses=[AIMessage(content=json.dumps(ITEMS,
                                                                           ensure_ascii=False)),
                                               AIMessage(content=REVISION)])
    checker = Checker(sqlite3.connect(f'file:{library}?mode=ro', uri=True))
    state = build_graph(checker, llm).invoke(
        {key: draft[key] for key in ('text', 'code', 'company', 'fiscal_year', 'as_of')})
    return state['workpaper']


def post(base: str, path: str, body: dict) -> tuple[int, dict]:
    request = Request(base + path, data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def get(base: str, path: str) -> tuple[int, bytes]:
    try:
        with urlopen(base + path, timeout=10) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


class ServiceTest(unittest.TestCase):
    """The endpoints of the page, served on a random loopback port."""

    def setUp(self):
        self.patch = mock.patch('src.service.app.review_draft', fake_review_draft)
        self.patch.start()
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.library = root / 'library.sqlite'
        build_library(self.library)
        self.data_dir = root / 'data'
        self.data_dir.mkdir()
        self.server = ReviewServer(('127.0.0.1', 0), library=self.library,
                                   data_dir=self.data_dir, budget=0.0, local_url=None)
        threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.02},
                         daemon=True).start()
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.patch.stop()
        self.directory.cleanup()

    def body(self, **changes) -> dict:
        request = {'code': '600000', 'fiscal_year': 2024, 'as_of': '2025-04-30', 'text': TEXT,
                   'model': 'deepseek-flash'}
        request.update(changes)
        return request

    def test_the_page_is_served(self):
        status, page = get(self.base, '/')
        self.assertEqual(status, 200)
        self.assertIn('研报点评预审', page.decode())

    def test_the_issuers_come_from_the_library(self):
        status, found = get(self.base, '/api/issuers')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(found), [{'code': '600000', 'publisher': '某公司',
                                              'reports': [{'fiscal_year': 2024,
                                                           'published': '2025-03-30'}]}])

    def test_a_review_returns_a_workpaper_and_downloads_it(self):
        status, body = post(self.base, '/api/review', self.body())
        self.assertEqual(status, 200)
        paper = body['workpaper']
        self.assertEqual(paper['summary'], {'numbers': 2, 'consistent': 1, 'inconsistent': 1,
                                            'unsupported': 0, 'ambiguous': 0, 'missed': 0,
                                            'agent_flagged': 0, 'agent_hint_consistent': 0})
        self.assertEqual([row['verdict'] for row in paper['numbers']],
                         ['inconsistent', 'consistent'])
        self.assertIn('2024年营业收入1741.44亿元', paper['numbers'][0]['suggestion']['sentence'])
        status, saved = get(self.base, f"/api/workpapers/{body['id']}.json")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(saved), paper)
        status, markdown = get(self.base, f"/api/workpapers/{body['id']}.md")
        self.assertEqual(status, 200)
        self.assertIn('# 工作底稿', markdown.decode())
        self.assertTrue((self.data_dir / f"{body['id']}.json").is_file())

    def test_an_unknown_issuer_is_refused(self):
        status, body = post(self.base, '/api/review', self.body(code='999999'))
        self.assertEqual(status, 400)
        self.assertIn('出处库中没有发行人 999999', body['error'])

    def test_a_year_without_a_report_is_refused(self):
        status, body = post(self.base, '/api/review', self.body(fiscal_year=2030))
        self.assertEqual(status, 400)
        self.assertIn('出处库中没有发行人 600000 的 2030 年年度报告', body['error'])

    def test_an_empty_draft_is_refused(self):
        status, body = post(self.base, '/api/review', self.body(text='   '))
        self.assertEqual(status, 400)
        self.assertEqual(body['error'], '稿件正文为空')

    def test_a_bad_cutoff_date_is_refused(self):
        for as_of in ('2025/04/30', '2025-4-30', '2025-02-30'):
            status, body = post(self.base, '/api/review', self.body(as_of=as_of))
            self.assertEqual(status, 400)
            self.assertIn('写稿截止日格式错', body['error'])

    def test_a_local_model_without_an_endpoint_is_refused(self):
        status, body = post(self.base, '/api/review', self.body(model='qwen3.5-9b-extract'))
        self.assertEqual(status, 400)
        self.assertIn('--local-url', body['error'])

    def test_an_unknown_workpaper_is_not_found(self):
        status, _ = get(self.base, '/api/workpapers/nope.json')
        self.assertEqual(status, 404)


if __name__ == '__main__':
    unittest.main()
