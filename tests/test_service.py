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

from src.agent import graph as graph_module
from src.agent.graph import build_graph
from src.data.library import SCHEMA, digits
from src.model.api_client import ApiClient, pieces
from src.service.app import DEEPSEEK_URL, ReviewServer
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
LOCAL_URL = 'http://127.0.0.1:18090/v1/chat/completions'
# A draft with a number the extraction answer leaves out, so an agent re-opens it.
HANDED_TEXT = '2024年营业收入1,745.44亿元；第四季度净利润3.20亿元。'
AGENT_ANSWER = json.dumps([{'number': '3.20', 'occurrence': 1, 'verdict': '正确',
                            'expected': '', 'basis': '某公司2024年年报'}], ensure_ascii=False)
DEMO_DRAFTS = [{'title': '某公司2024年报点评（示例）', 'code': '600000', 'fiscal_year': 2024,
                'as_of': '2025-04-30', 'source': '来自验收集', 'text': TEXT}]


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


def sse(base: str, path: str, body: dict) -> list[dict]:
    """The server-sent events of one streaming review, in the order they arrive."""
    request = Request(base + path, data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=30) as response:
        stream = response.read().decode()
    return [json.loads(line[len('data: '):]) for line in stream.splitlines()
            if line.startswith('data: ')]


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
        self.demo_drafts = root / 'demo_drafts.json'
        self.demo_drafts.write_text(json.dumps(DEMO_DRAFTS, ensure_ascii=False))
        self.servers = []
        self.server = self.start()
        self.base = self.url(self.server)

    def start(self, local_url: str | None = None, budget: float = 0.0) -> ReviewServer:
        """One more server, so a test can switch ``--local-url`` on and off."""
        server = ReviewServer(('127.0.0.1', 0), library=self.library, data_dir=self.data_dir,
                              budget=budget, local_url=local_url,
                              demo_drafts=self.demo_drafts)
        threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.02},
                         daemon=True).start()
        self.servers.append(server)
        return server

    @staticmethod
    def url(server: ReviewServer) -> str:
        return f'http://127.0.0.1:{server.server_address[1]}'

    def tearDown(self):
        for server in self.servers:
            server.shutdown()
            server.server_close()
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

    def test_the_page_offers_the_local_extractor_with_a_flash_review(self):
        status, page = get(self.base, '/')
        self.assertEqual(status, 200)
        self.assertIn('本机抽取（qwen3.5-9b-extract）＋ flash 复查', page.decode())

    def test_the_page_learns_whether_a_local_model_is_configured(self):
        status, body = get(self.base, '/api/config')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {'local_url': False})
        server = self.start(local_url=LOCAL_URL)
        status, body = get(self.url(server), '/api/config')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {'local_url': True})

    def test_the_local_extractor_hands_the_review_to_flash_on_its_own_cache(self):
        seen = {}

        def capture(draft, **kwargs):
            seen.update(kwargs)
            core = {key: value for key, value in kwargs.items()
                    if not key.startswith('agent_')}
            return fake_review_draft(draft, **core)

        server = self.start(local_url=LOCAL_URL)
        with mock.patch('src.service.app.review_draft', capture):
            status, _ = post(self.url(server), '/api/review',
                             self.body(model='qwen3.5-9b-extract'))
        self.assertEqual(status, 200)
        self.assertEqual(seen['model'], 'qwen3.5-9b-extract')
        self.assertEqual(seen['url'], LOCAL_URL)
        self.assertEqual(seen['cache'], self.data_dir / 'api_cache.jsonl')
        self.assertEqual(seen['agent_model'], 'deepseek-flash')
        self.assertEqual(seen['agent_url'], DEEPSEEK_URL)
        self.assertEqual(seen['agent_cache'],
                         self.data_dir / 'api_cache_deepseek-flash.jsonl')

    def test_the_demo_drafts_are_served_from_the_configured_file(self):
        status, body = get(self.base, '/api/demo-drafts')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), DEMO_DRAFTS)

    def answer(self, body: dict) -> dict:
        """A transport that scripts the extraction, the agent and the revision."""
        last = body['messages'][-1]
        content = last['content'] if isinstance(last['content'], str) else ''
        if '请只把这个数字改成正确的值' in content:
            answer = REVISION
        elif body['model'] == 'deepseek-flash':
            answer = AGENT_ANSWER
        else:
            answer = json.dumps(ITEMS, ensure_ascii=False)
        return {'usage': {'prompt_tokens': 100, 'completion_tokens': 20},
                'choices': [{'message': {'content': answer}, 'finish_reason': 'stop'}]}

    def client(self, model, budget, cache, url=None) -> ApiClient:
        return ApiClient(model, budget, Path(cache), url=url, send=self.answer,
                         stream=self.stream)

    def stream(self, body: dict, on_delta) -> dict:
        """The streaming transport: the same answer, reported as it is written."""
        payload = self.answer(body)
        for piece in pieces(payload['choices'][0]['message']['content']):
            on_delta(piece)
        return payload

    def test_the_stream_endpoint_pushes_one_event_per_node_and_then_the_id(self):
        server = self.start(local_url=LOCAL_URL, budget=0.5)
        with mock.patch('src.service.app.review_draft', graph_module.review_draft), \
                mock.patch('src.agent.graph.ApiClient', side_effect=self.client), \
                mock.patch('src.agent.single.ApiClient', side_effect=self.client):
            events = sse(self.url(server), '/api/review/stream',
                         self.body(model='qwen3.5-9b-extract', text=HANDED_TEXT))
        nodes = [event for event in events if event.get('node')]
        self.assertEqual([event['node'] for event in nodes],
                         ['extract', 'check', 'investigate', 'revise', 'report', 'done'])
        self.assertEqual(nodes[1]['detail']['conclusions'],
                         {'一致': 0, '不一致': 1, '无依据': 0, '存疑': 0, '抽取遗漏需人工': 1})
        self.assertEqual(nodes[2]['detail']['handed'], 1)  # the number the checker left over
        self.assertEqual(nodes[2]['detail']['judged'], 1)
        # 过程事件：抽取的文字与条目、核查的逐个结论、复查的工具调用与结论、修订。
        live = [event['event'] for event in events if not event.get('node')]
        self.assertEqual(sorted(set(live)), ['agent', 'check', 'delta', 'item', 'revise'])
        self.assertGreater(len([kind for kind in live if kind == 'delta']), 1)
        self.assertEqual(''.join(event['text'] for event in events
                                 if event.get('event') == 'delta'),
                         json.dumps(ITEMS, ensure_ascii=False))
        self.assertEqual([kind for kind in live if kind != 'delta'],
                         ['item', 'item', 'check', 'check', 'agent', 'revise'])
        self.assertEqual([event['number'] for event in events if event.get('event') == 'item'],
                         ['1,745.44', '1,741.44'])
        self.assertEqual([event['verdict'] for event in events if event.get('event') == 'check'],
                         ['inconsistent', 'missed'])
        missed = next(event for event in events
                      if event.get('event') == 'check' and event['verdict'] == 'missed')
        self.assertEqual([event['start'] for event in events if event.get('event') == 'agent'],
                         [missed['start']])
        self.assertEqual([event['accepted'] for event in events
                          if event.get('event') == 'revise'], [True])
        review_id = nodes[-1]['id']
        paper = json.loads((self.data_dir / f'{review_id}.json').read_text(encoding='utf-8'))
        self.assertEqual([event['node'] for event in paper['trace']],
                         [event['node'] for event in nodes[:-1]])
        self.assertEqual(paper['summary']['numbers'], 2)
        self.assertEqual(paper['numbers'][1]['agent']['verdict'], 'consistent')
        self.assertEqual(paper['numbers'][1]['verdict'], 'missed')  # a hint, not a verdict
        self.assertTrue((self.data_dir / f'{review_id}.md').is_file())

    def test_the_stream_endpoint_refuses_a_bad_draft_like_the_plain_one(self):
        status, body = post(self.base, '/api/review/stream', self.body(code='999999'))
        self.assertEqual(status, 400)
        self.assertIn('出处库中没有发行人 999999', body['error'])

    def test_a_flash_run_calls_the_graph_exactly_as_before(self):
        seen = {}

        def capture(draft, **kwargs):
            seen.update(kwargs)
            return fake_review_draft(draft, **kwargs)

        with mock.patch('src.service.app.review_draft', capture):
            status, _ = post(self.base, '/api/review', self.body())
        self.assertEqual(status, 200)
        self.assertEqual(seen, {'library': self.library, 'model': 'deepseek-flash',
                                'url': DEEPSEEK_URL, 'budget_usd': 0.0,
                                'cache': self.data_dir / 'api_cache.jsonl'})


if __name__ == '__main__':
    unittest.main()
