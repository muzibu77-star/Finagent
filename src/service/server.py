"""Loopback-only UI with one persistent worker and durable polling cursors."""

import argparse
import hashlib
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, unquote, urlparse

from src.harness.store import Conflict, Store


class Service:
    def __init__(self, database: str, corpus_path: str, snapshot: str, adapter: str | None,
                 reports: bool = False, neural_index: str | None = None):
        self.store = Store(database)
        self.corpus_path, self.snapshot, self.adapter = corpus_path, snapshot, adapter
        self.reports = reports
        self.neural_index = neural_index
        if neural_index and not reports:
            raise ValueError('neural document retrieval requires report metadata')
        raw = Path(corpus_path).read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.documents = json.loads(raw)
        identity = {'snapshot': snapshot, 'corpus_sha256': self.fingerprint,
                    'workflow': 'fixed-v1'}
        if adapter:
            identity['adapter_sha256'] = {name: hashlib.sha256(
                (Path(adapter) / name).read_bytes()).hexdigest()
                for name in ('adapter_config.json', 'adapter_model.safetensors')}
        if reports:
            identity['prompt_serialization'] = 'unicode-v1'
            identity['document_tool_contract'] = 'lookup-v1'
        if neural_index:
            identity['retrieval_config_sha256'] = hashlib.sha256(
                Path('configs/jd_retrieval.json').read_bytes()).hexdigest()
        binding = json.dumps(identity, sort_keys=True)
        with self.store.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS service_config (id INTEGER PRIMARY KEY, body TEXT NOT NULL)')
            previous = db.execute('SELECT body FROM service_config WHERE id=1').fetchone()
            if previous and previous['body'] != binding:
                raise ValueError('database belongs to a different source snapshot, model or workflow')
            db.execute('INSERT OR IGNORE INTO service_config VALUES(1,?)', (binding,))
        self.ready, self.error = False, None
        self.shutdown = threading.Event()
        self.wakeup = threading.Event()
        self.thread = None
        self.current = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self.work, name='model-worker', daemon=False)
        self.thread.start()

    def work(self) -> None:
        # Model imports/loading and the SQLite retrieval index stay on their owning thread.
        from src.agent.fixed_runner import FixedRunner
        from src.data.corpus import Corpus
        from src.data.report_corpus import ReportCorpus
        from src.model.agent import ModelDriver

        try:
            with ModelDriver(self.adapter) as model:
                if self.neural_index:
                    from src.data.rag_corpus import RagCorpus

                    corpus = RagCorpus(self.corpus_path, self.snapshot, Path(self.neural_index))
                elif self.reports:
                    choice = json.loads(Path('configs/report_retrieval_choice.json').read_text())
                    corpus = ReportCorpus(self.corpus_path, self.snapshot, choice['mode'])
                    if corpus.fingerprint != choice['corpus_sha256']:
                        raise ValueError('retrieval choice belongs to a different snapshot')
                else:
                    corpus = Corpus(self.corpus_path, self.snapshot)
                try:
                    self.store.recover()
                    runner = FixedRunner(self.store, corpus, model,
                                         unicode_context=self.reports)
                    self.ready = True
                    while not self.shutdown.is_set():
                        with self.store.transaction() as db:
                            row = db.execute("SELECT id FROM tasks WHERE execution='queued' ORDER BY rowid LIMIT 1").fetchone()
                        if row is None:
                            self.wakeup.wait(0.5)
                            self.wakeup.clear()
                            continue
                        self.current = row['id']
                        runner.run(self.current)
                        self.current = None
                        if not model.healthy:
                            raise RuntimeError('model health check failed; worker stopped')
                finally:
                    corpus.close()
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'[:500]
            logging.exception('Worker stopped')
        finally:
            self.ready = False

    def close(self) -> None:
        self.shutdown.set()
        self.wakeup.set()
        if self.current:
            self.store.cancel(self.current)
        if self.thread:
            self.thread.join(timeout=60)
            if self.thread.is_alive():
                raise RuntimeError('worker has not released resources')

    def catalogue(self) -> dict:
        companies = {}
        for row in self.documents.values():
            companies.setdefault(row['company'], set()).add(row['period'])
        return {'snapshot_id': self.snapshot, 'ready': self.ready, 'error': self.error,
                'companies': {key: sorted(values) for key, values in sorted(companies.items())},
                'strict_pit': self.reports,
                'notice': '研究原型 · 单题取证计算 · 结果需逐项核对原文。'}

    def source(self, task_id: str, evidence_id: str) -> dict:
        task = self.store.get(task_id)
        cited = set(task['state'].get('read', {}))
        for calculation in (task['report'] or {}).get('calculations', []):
            cited.update(f['evidence_id'] for f in calculation['facts'].values())
        if evidence_id not in cited or evidence_id not in self.documents:
            raise ValueError('source not read or cited by this task revision')
        document = self.documents[evidence_id]
        return {key: document[key] for key in ('evidence', 'source_report_id', 'url',
                'pdf_page', 'printed_page', 'published_at', 'version') if key in document}


def handler_for(service: Service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logging.info('%s %s', self.address_string(), fmt % args)

        def send(self, status: int, body, content_type='application/json; charset=utf-8'):
            data = json.dumps(body, ensure_ascii=False).encode() if not isinstance(body, bytes) else body
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def dispatch(self, mutation=False):
            host = self.headers.get('Host', '')
            expected = {'127.0.0.1', 'localhost'}
            if host.split(':')[0] not in expected:
                return self.send(403, {'error': 'loopback host required'})
            origin = self.headers.get('Origin')
            if mutation and origin and origin != 'http://' + host:
                return self.send(403, {'error': 'origin mismatch'})
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                if mutation:
                    if self.headers.get_content_type() != 'application/json':
                        raise ValueError('JSON request required')
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 65536:
                        raise ValueError('request body size invalid')
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError('JSON object required')
                    if path == '/api/tasks':
                        if not service.ready:
                            return self.send(503, {'error': service.error or '模型正在加载'})
                        if not isinstance(body.get('request_id'), str) or not 1 <= len(body['request_id']) <= 128:
                            raise ValueError('request ID required')
                        payload = body.get('payload')
                        if not isinstance(payload, dict):
                            raise ValueError('payload required')
                        if payload.get('snapshot_id') != service.snapshot:
                            raise ValueError('unknown snapshot')
                        if payload.get('company') is not None and not isinstance(payload['company'], str):
                            raise ValueError('company must be a string')
                        periods = payload.get('period')
                        if periods is not None and not (isinstance(periods, str) or
                                isinstance(periods, list) and all(isinstance(p, str) for p in periods)):
                            raise ValueError('invalid report period')
                        if payload.get('as_of') is not None and not isinstance(payload['as_of'], str):
                            raise ValueError('invalid cutoff')
                        if body.get('task_id') is not None and not isinstance(body['task_id'], str):
                            raise ValueError('invalid task ID')
                        result = service.store.submit(body['request_id'], payload, body.get('task_id'))
                        service.wakeup.set()
                        return self.send(202, result)
                    match = re.fullmatch(r'/api/tasks/([^/]+)/cancel', path)
                    if match:
                        service.store.cancel(match[1])
                        return self.send(200, {'cancellation_requested': True})
                else:
                    if path in ('/', '/app.js', '/style.css'):
                        filename = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[path]
                        mime = {'/': 'text/html', '/app.js': 'text/javascript', '/style.css': 'text/css'}[path]
                        return self.send(200, Path(__file__).with_name(filename).read_bytes(), mime + '; charset=utf-8')
                    if path == '/api/catalogue':
                        return self.send(200, service.catalogue())
                    match = re.fullmatch(r'/api/tasks/([^/]+)(?:/(events|report|source))?', path)
                    if match:
                        task_id, operation = match.groups()
                        task = service.store.get(task_id)
                        query = parse_qs(parsed.query)
                        if operation == 'events':
                            after = int(query.get('after', ['0'])[0])
                            if after < 0:
                                raise ValueError('negative event cursor')
                            return self.send(200, service.store.events(task_id, after))
                        if operation == 'report':
                            if task['report'] is None:
                                return self.send(409, {'error': '本轮尚无报告'})
                            return self.send(200, task['report'])
                        if operation == 'source':
                            return self.send(200, service.source(task_id, query.get('id', [''])[0]))
                        return self.send(200, {key: task[key] for key in
                            ('id', 'revision', 'payload', 'execution', 'business', 'report', 'cancel')} |
                            {'detail': task['state'].get('stop_detail'), 'attempts': task['state'].get('attempts', 0)})
                return self.send(404, {'error': 'not found'})
            except Conflict as exc:
                self.send(409, {'error': str(exc)})
            except KeyError:
                self.send(404, {'error': 'task or source not found'})
            except (ValueError, TypeError) as exc:
                self.send(400, {'error': str(exc)})

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch(mutation=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--database', default='artifacts/service/tasks.sqlite')
    parser.add_argument('--corpus', default='data/staged/m3_acceptance_v1/corpus.json')
    parser.add_argument('--snapshot', default='m3-frozen-v1')
    parser.add_argument('--reports', action='store_true')
    parser.add_argument('--neural-index', help='Persistent Qdrant index for the explicit document RAG mode')
    args = parser.parse_args()
    if args.neural_index and not args.reports:
        parser.error('--neural-index requires --reports')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    choice = json.loads(Path('configs/model_choice.json').read_text())
    adapter = (choice['adapter'] if args.reports else
               choice.get('calculator_adapter', choice['adapter']))
    service = Service(args.database, args.corpus, args.snapshot, adapter, args.reports, args.neural_index)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(service))
    service.start()
    logging.info('UI http://127.0.0.1:%d ; database=%s', args.port, args.database)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
