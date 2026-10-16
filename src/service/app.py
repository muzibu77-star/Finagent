"""The local review page: one form in front of the LangGraph review flow.

The server binds to the loopback address only, runs one draft at a time and
keeps what it returns on disk: ``<data-dir>/<id>.json`` is the workpaper, its
``.md`` the Markdown version and ``api_cache.jsonl`` the metered client's cache.
Only the issuer list is read from the source library; the draft itself comes
from the page.  ``index.html`` beside this file is the whole front end.

``POST /api/review/stream`` reviews one draft the same way but answers with
server-sent events: one per node, as ``graph.review_events`` yields them, and a
last one carrying the id of the workpaper it wrote, which is what the page's flow
band follows.  ``GET /api/demo-drafts`` serves the demonstration drafts of
``--demo-drafts``.

The pages of a number of a workpaper are served from the same source library
and the download receipts of ``--sources`` (see ``src/service/evidence.py``):
``GET /api/evidence`` describes the fact behind one character range and
``GET /api/evidence.png`` renders its page, or the crop around its box.  The
buttons of the page append one line per click to ``<data-dir>/feedback.jsonl``
through ``POST /api/feedback``; that file is the only thing feedback writes and
it never changes a workpaper, a verdict or a rendering.  ``GET /api/feedback``
reads back the last click on each number of a workpaper and ``GET
/api/workpapers`` lists the workpapers kept in the data directory.  ``GET
/api/config`` tells the page whether a local model is configured, so it can
fall back to flash when it is not.  A draft reviewed by the local extractor
hands investigate and revise to flash on ``<data-dir>/api_cache_deepseek-flash.jsonl``;
a flash draft keeps the single ``api_cache.jsonl`` of ``--model``.

    python -m src.service.app --library <library.sqlite> --data-dir <dir> \
        [--sources configs/cninfo_sources.json] \
        [--demo-drafts configs/demo_drafts.json] \
        [--local-url http://127.0.0.1:18090/v1/chat/completions] [--budget-usd 0.5]
"""

import argparse
from datetime import date
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlencode, urlparse
import uuid

from pypdfium2 import PdfiumError

from src.agent.graph import review_draft, workpaper_markdown
from src.service import evidence

log = logging.getLogger(__name__)
DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions'
LOCAL_MODEL = 'qwen3.5-9b-extract'
AGENT_MODEL = 'deepseek-flash'
AGENT_CACHE = 'api_cache_deepseek-flash.jsonl'
MODELS = ('deepseek-flash', LOCAL_MODEL)
PAGE = Path(__file__).with_name('index.html')
DATE = re.compile(r'\d{4}-\d{2}-\d{2}')
WORKPAPER = re.compile(r'[A-Za-z0-9_-]+\.(json|md)')
ANSWERS = ('一致', '确有错误', '跳过')


def route(target: str) -> tuple[str, dict[str, str]]:
    """The path and the first value of each query parameter of a request target."""
    parsed = urlparse(target)
    return parsed.path, {key: values[0] for key, values in parse_qs(parsed.query).items()}


def issuers(db: sqlite3.Connection) -> list[dict]:
    """The issuers that have an annual report in the library, with their years."""
    rows = db.execute(
        "SELECT code, publisher, fiscal_year, published FROM documents"
        " WHERE kind = 'annual' AND code IS NOT NULL"
        ' ORDER BY publisher, fiscal_year').fetchall()
    found: dict[str, dict] = {}
    for code, publisher, year, published in rows:
        entry = found.setdefault(code, {'code': code, 'publisher': publisher,
                                        'reports': []})
        entry['reports'].append({'fiscal_year': year, 'published': published})
    return sorted(found.values(), key=lambda entry: (entry['publisher'], entry['code']))


def valid_date(text: str) -> bool:
    """Whether ``YYYY-MM-DD`` names a real day."""
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


class ReviewHandler(BaseHTTPRequestHandler):
    """The page, its JSON API and the workpaper downloads."""

    def do_GET(self) -> None:
        path, params = route(self.path)
        if path in ('/', '/index.html'):
            return self.reply(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/issuers':
            return self.reply_json(issuers(self.library()))
        if path == '/api/config':
            return self.reply_json({'local_url': bool(self.server.local_url)})
        if path == '/api/workpapers':
            return self.reply_json(self.workpapers())
        if path == '/api/feedback':
            return self.feedback_of(params.get('review_id') or '')
        if path == '/api/demo-drafts':
            return self.demo_drafts()
        if path == '/api/evidence':
            return self.evidence_json(params)
        if path == '/api/evidence.png':
            return self.evidence_image(params)
        name = path.removeprefix('/api/workpapers/')
        if name != path and WORKPAPER.fullmatch(name):
            return self.download(name)
        self.fail(404, '没有这个地址')

    def do_POST(self) -> None:
        path, _ = route(self.path)
        if path == '/api/review':
            return self.review()
        if path == '/api/review/stream':
            return self.review_stream()
        if path == '/api/feedback':
            return self.feedback()
        self.fail(404, '没有这个地址')

    def review(self) -> None:
        """Review one draft and keep its workpaper in the data directory."""
        draft = self.draft_of_request()
        if draft is None:
            return
        try:
            with self.server.lock:  # one draft at a time
                paper = review_draft(draft, library=self.server.library,
                                     model=draft['model'], url=draft['url'],
                                     budget_usd=self.server.budget, cache=self.server.cache,
                                     **self.agent_calls(draft))
        except Exception as error:  # the graph itself already survives model failures
            return self.fail(500, f'核查失败：{type(error).__name__}: {error}')
        self.reply_json({'id': self.save(paper, draft), 'workpaper': paper})

    def review_stream(self) -> None:
        """Review one draft, pushing one event per node and then the workpaper id.

        The response is a server-sent event stream, so its headers go out before
        the graph starts and a failure after that travels as an ``error`` event.
        The lock is the same one ``/api/review`` takes, so only one draft runs.
        """
        draft = self.draft_of_request()
        if draft is None:
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        try:
            with self.server.lock:  # one draft at a time
                paper = review_draft(draft, library=self.server.library,
                                     model=draft['model'], url=draft['url'],
                                     budget_usd=self.server.budget, cache=self.server.cache,
                                     on_event=self.push, **self.agent_calls(draft))
        except Exception as error:  # the graph itself already survives model failures
            return self.push({'node': 'error',
                              'message': f'核查失败：{type(error).__name__}: {error}'})
        self.push({'node': 'done', 'id': self.save(paper, draft)})

    def push(self, event: dict) -> None:
        """One server-sent event of a streaming review."""
        self.wfile.write(f'data: {json.dumps(event, ensure_ascii=False)}\n\n'.encode())
        self.wfile.flush()

    def save(self, paper: dict, draft: dict) -> str:
        """Write one workpaper and its Markdown, and return the id it is kept under."""
        review_id = (f"{draft['code']}-{draft['fiscal_year']}-{time.strftime('%Y%m%d%H%M%S')}"
                     f"-{uuid.uuid4().hex[:4]}")
        (self.server.data_dir / f'{review_id}.json').write_text(
            json.dumps(paper, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        (self.server.data_dir / f'{review_id}.md').write_text(workpaper_markdown(paper),
                                                              encoding='utf-8')
        return review_id

    def agent_calls(self, draft: dict) -> dict:
        """The investigate and revise model of one draft.

        The local extractor hands those two nodes to flash, which is billed and
        cached apart from the extractor; a flash draft uses flash throughout, so
        it names no agent model and its requests are the ones of before.
        """
        if draft['model'] != LOCAL_MODEL:
            return {}
        return {'agent_model': AGENT_MODEL, 'agent_url': DEEPSEEK_URL,
                'agent_cache': self.server.data_dir / AGENT_CACHE}

    def draft_of_request(self) -> dict | None:
        """The validated draft of a request; a refusal is answered and yields ``None``."""
        try:
            body = json.loads(self.read_body())
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.fail(400, '请求不是合法的 JSON')
            return None
        if not isinstance(body, dict):
            self.fail(400, '请求不是合法的 JSON 对象')
            return None
        draft, reason = self.validated(body)
        if reason is not None:
            self.fail(400, reason)
            return None
        return draft

    def demo_drafts(self) -> None:
        """The demonstration drafts of ``--demo-drafts``, as the page loads them."""
        if not self.server.demo_drafts.is_file():
            return self.reply_json([])
        self.reply_json(json.loads(self.server.demo_drafts.read_text(encoding='utf-8')))

    def evidence_json(self, params: dict[str, str]) -> None:
        """The fact behind one character range: its page, box and renderings."""
        located, reason = self.located(params)
        if located is None:
            return self.fail(*reason)
        found, box = located['fact'], located['box']
        body = {'id': found['id'], 'document_id': located['document_id'],
                'title': found['title'],
                'page': located['page'], 'start': located['start'], 'end': located['end'],
                'box': box, 'text': found['text'], 'unit': found['unit'],
                'row_label': found['row_label'], 'column_header': found['column_header'],
                'page_url': self.render_url(params), 'crop_url': None}
        if box:
            body['crop_url'] = self.render_url(params, crop=True)
        else:
            body['note'] = '这一页没有文本层位置框，无法定位到数字，只显示整页'
        self.reply_json(body)

    def evidence_image(self, params: dict[str, str]) -> None:
        """The page of a fact, or with ``crop=1`` the crop around its box."""
        located, reason = self.located(params)
        if located is None:
            return self.fail(*reason)
        fact, crop = located['fact'], params.get('crop') == '1'
        if crop and not located['box']:
            return self.fail(400, '这条事实没有位置框，只有整页可看')
        try:
            png = evidence.page_png(located['file'], fact['page'], located['box'] or None,
                                    scale=evidence.ZOOM if crop else evidence.SCALE,
                                    margin=evidence.MARGIN if crop else None)
        except (PdfiumError, ValueError, IndexError) as error:
            return self.fail(409, f'这一页渲染不了：{error}')
        self.reply(200, png, 'image/png')

    def located(self, params: dict[str, str]) -> tuple[dict | None, tuple[int, str] | None]:
        """The fact, its box and its verified file of one evidence request."""
        document_id = str(params.get('document_id') or '')
        numbers = [str(params.get(name) or '') for name in ('page', 'start', 'end')]
        if not document_id or not all(number.isdigit() for number in numbers):
            return None, (400, 'document_id、page、start、end 必须齐全且为数字')
        page, start, end = (int(number) for number in numbers)
        found = evidence.fact(self.library(), document_id, page, start, end)
        if found is None:
            return None, (404, f'出处库中没有 {document_id} 第{page}页的 [{start},{end}) '
                               '这条事实')
        try:
            entry = evidence.receipt(self.server.sources, document_id)
        except LookupError as error:
            return None, (404, str(error))
        except ValueError as error:
            return None, (409, str(error))
        return {'document_id': document_id, 'page': page, 'start': start, 'end': end,
                'fact': found, 'box': evidence.box_of(found), 'file': entry['file']}, None

    def render_url(self, params: dict[str, str], crop: bool = False) -> str:
        """The rendering URL of one fact."""
        query = {name: params[name] for name in ('document_id', 'page', 'start', 'end')}
        if crop:
            query['crop'] = '1'
        return '/api/evidence.png?' + urlencode(query)

    def feedback(self) -> None:
        """Append one click of a person to ``<data-dir>/feedback.jsonl``."""
        try:
            body = json.loads(self.read_body())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self.fail(400, '请求不是合法的 JSON')
        if not isinstance(body, dict):
            return self.fail(400, '请求不是合法的 JSON 对象')
        review_id = str(body.get('review_id') or '')
        if not WORKPAPER.fullmatch(f'{review_id}.json'):
            return self.fail(400, '底稿编号不合法')
        start = body.get('start')
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            return self.fail(400, '数字起点不合法')
        answer = body.get('answer')
        if answer not in ANSWERS:
            return self.fail(400, f'按钮只能是 {"、".join(ANSWERS)}')
        note = body.get('note') or ''
        if not isinstance(note, str) or len(note) > 500:
            return self.fail(400, '备注必须是不超过 500 字的文字')
        paper = self.server.data_dir / f'{review_id}.json'
        record = {'review_id': review_id, 'start': start, 'answer': answer, 'note': note,
                  'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                  'workpaper_version': (hashlib.sha256(paper.read_bytes()).hexdigest()
                                        if paper.is_file() else None)}
        with self.server.feedback_lock, open(self.server.feedback, 'a',
                                             encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.reply_json(record)

    def workpapers(self) -> list[dict]:
        """The workpapers kept in the data directory, newest first, with their counts."""
        papers = []
        for path in sorted(self.server.data_dir.glob('*.json'),
                           key=lambda item: item.stat().st_mtime, reverse=True):
            if not WORKPAPER.fullmatch(path.name):
                continue
            try:
                summary = json.loads(path.read_text(encoding='utf-8'))['summary']
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            papers.append({'id': path.stem, 'summary': summary})
        return papers

    def feedback_of(self, review_id: str) -> None:
        """The last click on each number of one workpaper, keyed by the number's start."""
        if not WORKPAPER.fullmatch(f'{review_id}.json'):
            return self.fail(400, '底稿编号不合法')
        latest = {}
        if self.server.feedback.is_file():
            for line in self.server.feedback.read_text(encoding='utf-8').splitlines():
                record = json.loads(line)
                if record['review_id'] == review_id:
                    latest[str(record['start'])] = record
        self.reply_json(latest)

    def validated(self, body: dict) -> tuple[dict, str | None]:
        """The draft of a request, or the reason its parameters are refused."""
        model = str(body.get('model') or '')
        if model not in MODELS:
            return {}, f'模型只能是 {"、".join(MODELS)}'
        text = str(body.get('text') or '')
        if not text.strip():
            return {}, '稿件正文为空'
        as_of = str(body.get('as_of') or '').strip()
        if not DATE.fullmatch(as_of) or not valid_date(as_of):
            return {}, '写稿截止日格式错，应为 YYYY-MM-DD'
        year = body.get('fiscal_year')
        code = str(body.get('code') or '').strip()
        row = None
        if isinstance(year, int):
            row = self.library().execute(
                "SELECT publisher FROM documents WHERE kind = 'annual' AND code = ?"
                ' AND fiscal_year = ? ORDER BY published DESC LIMIT 1',
                (code, year)).fetchone()
        if row is None:
            return {}, f'出处库中没有发行人 {code} 的 {year} 年年度报告'
        if model == LOCAL_MODEL and not self.server.local_url:
            return {}, '本机模型未配置服务地址（启动时加 --local-url）'
        url = self.server.local_url if model == LOCAL_MODEL else DEEPSEEK_URL
        return {'text': text, 'code': code, 'company': row[0], 'fiscal_year': year,
                'as_of': as_of, 'model': model, 'url': url}, None

    def library(self) -> sqlite3.Connection:
        return sqlite3.connect(f'file:{self.server.library}?mode=ro', uri=True)

    def read_body(self) -> str:
        return self.rfile.read(int(self.headers.get('Content-Length') or 0)).decode()

    def download(self, name: str) -> None:
        path = self.server.data_dir / name
        if not path.is_file():
            return self.fail(404, '底稿不存在')
        kind = ('application/json; charset=utf-8' if name.endswith('.json')
                else 'text/markdown; charset=utf-8')
        self.reply(200, path.read_bytes(), kind)

    def fail(self, status: int, reason: str) -> None:
        self.reply_json({'error': reason}, status)

    def reply_json(self, body, status: int = 200) -> None:
        self.reply(status, json.dumps(body, ensure_ascii=False).encode(),
                   'application/json; charset=utf-8')

    def reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        log.info('%s - %s', self.address_string(), format % args)


class ReviewServer(ThreadingHTTPServer):
    """The review server: the library, the source receipts, the budget and the locks."""

    daemon_threads = True

    def __init__(self, address, *, library: Path, data_dir: Path, budget: float,
                 local_url: str | None,
                 sources: Path = Path('configs/cninfo_sources.json'),
                 demo_drafts: Path = Path('configs/demo_drafts.json')) -> None:
        super().__init__(address, ReviewHandler)
        self.library = Path(library)
        self.data_dir = Path(data_dir)
        self.budget = budget
        self.local_url = local_url
        self.sources = Path(sources)
        self.demo_drafts = Path(demo_drafts)
        self.cache = self.data_dir / 'api_cache.jsonl'
        self.feedback = self.data_dir / 'feedback.jsonl'
        self.lock = threading.Lock()
        self.feedback_lock = threading.Lock()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='workpapers, their Markdown and the API cache')
    parser.add_argument('--port', type=int, default=18100, help='loopback port')
    parser.add_argument('--sources', type=Path, default=Path('configs/cninfo_sources.json'),
                        help='download receipts of the original files, for the evidence view')
    parser.add_argument('--demo-drafts', type=Path, default=Path('configs/demo_drafts.json'),
                        help='the demonstration drafts the page can load')
    parser.add_argument('--local-url', default=None,
                        help='OpenAI-compatible endpoint of the local model')
    parser.add_argument('--budget-usd', type=float, default=0.5,
                        help='model budget of one run; the local model is free')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    server = ReviewServer(('127.0.0.1', args.port), library=args.library,
                          data_dir=args.data_dir, budget=args.budget_usd,
                          local_url=args.local_url, sources=args.sources,
                          demo_drafts=args.demo_drafts)
    print(f'http://127.0.0.1:{server.server_address[1]}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
