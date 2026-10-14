"""The local review page: one form in front of the LangGraph review flow.

The server binds to the loopback address only, runs one draft at a time and
keeps what it returns on disk: ``<data-dir>/<id>.json`` is the workpaper, its
``.md`` the Markdown version and ``api_cache.jsonl`` the metered client's cache.
Only the issuer list is read from the source library; the draft itself comes
from the page.  ``index.html`` beside this file is the whole front end.

    python -m src.service.app --library <library.sqlite> --data-dir <dir> \
        [--local-url http://127.0.0.1:18090/v1/chat/completions] [--budget-usd 0.5]
"""

import argparse
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

from src.agent.graph import review_draft, workpaper_markdown

log = logging.getLogger(__name__)
DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions'
LOCAL_MODEL = 'qwen3.5-9b-extract'
MODELS = ('deepseek-flash', LOCAL_MODEL)
PAGE = Path(__file__).with_name('index.html')
DATE = re.compile(r'\d{4}-\d{2}-\d{2}')
WORKPAPER = re.compile(r'[A-Za-z0-9_-]+\.(json|md)')


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
        if self.path in ('/', '/index.html'):
            return self.reply(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
        if self.path == '/api/issuers':
            return self.reply_json(issuers(self.library()))
        name = self.path.removeprefix('/api/workpapers/')
        if name != self.path and WORKPAPER.fullmatch(name):
            return self.download(name)
        self.fail(404, '没有这个地址')

    def do_POST(self) -> None:
        if self.path != '/api/review':
            return self.fail(404, '没有这个地址')
        try:
            body = json.loads(self.read_body())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self.fail(400, '请求不是合法的 JSON')
        if not isinstance(body, dict):
            return self.fail(400, '请求不是合法的 JSON 对象')
        draft, reason = self.validated(body)
        if reason is not None:
            return self.fail(400, reason)
        try:
            with self.server.lock:  # one draft at a time
                paper = review_draft(draft, library=self.server.library,
                                     model=draft['model'], url=draft['url'],
                                     budget_usd=self.server.budget, cache=self.server.cache)
        except Exception as error:  # the graph itself already survives model failures
            return self.fail(500, f'核查失败：{type(error).__name__}: {error}')
        review_id = (f"{draft['code']}-{draft['fiscal_year']}-{time.strftime('%Y%m%d%H%M%S')}"
                     f"-{uuid.uuid4().hex[:4]}")
        (self.server.data_dir / f'{review_id}.json').write_text(
            json.dumps(paper, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        (self.server.data_dir / f'{review_id}.md').write_text(workpaper_markdown(paper),
                                                              encoding='utf-8')
        self.reply_json({'id': review_id, 'workpaper': paper})

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
    """The review server: the library, the model budget and the one-at-a-time lock."""

    daemon_threads = True

    def __init__(self, address, *, library: Path, data_dir: Path, budget: float,
                 local_url: str | None) -> None:
        super().__init__(address, ReviewHandler)
        self.library = Path(library)
        self.data_dir = Path(data_dir)
        self.budget = budget
        self.local_url = local_url
        self.cache = self.data_dir / 'api_cache.jsonl'
        self.lock = threading.Lock()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='workpapers, their Markdown and the API cache')
    parser.add_argument('--port', type=int, default=18100, help='loopback port')
    parser.add_argument('--local-url', default=None,
                        help='OpenAI-compatible endpoint of the local model')
    parser.add_argument('--budget-usd', type=float, default=0.5,
                        help='model budget of one run; the local model is free')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    server = ReviewServer(('127.0.0.1', args.port), library=args.library,
                          data_dir=args.data_dir, budget=args.budget_usd,
                          local_url=args.local_url)
    print(f'http://127.0.0.1:{server.server_address[1]}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
