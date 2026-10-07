"""Real HTTP contract tests; model behavior is covered by separate GPU acceptance."""

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from src.service.server import Service, handler_for


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        corpus = {'e1': {'company': 'A', 'period': '2023', 'source_report_id': 'A/2023',
                  'evidence': {'evidence_id': 'e1', 'table': [['Revenue', '10']]}}}
        (root / 'corpus.json').write_text(json.dumps(corpus))
        self.service = Service(str(root / 'tasks.sqlite'), str(root / 'corpus.json'), 's1', None)
        self.service.ready = True
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.opener = build_opener(ProxyHandler({}))
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.body = {'request_id': 'r1', 'payload': {'question': 'Revenue?', 'company': 'A',
                     'period': '2023', 'snapshot_id': 's1'}}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        request = Request(self.url + path, data=data,
                          headers={'Content-Type': 'application/json', **(headers or {})})
        try:
            with self.opener.open(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_duplicate_and_conflicting_request(self):
        status, first = self.request('/api/tasks', self.body)
        self.assertEqual(status, 202)
        self.assertEqual(self.request('/api/tasks', self.body), (202, first))
        changed = {**self.body, 'payload': {**self.body['payload'], 'question': 'Other?'}}
        self.assertEqual(self.request('/api/tasks', changed)[0], 409)

    def test_events_reconnect_and_cancel(self):
        self.request('/api/tasks', self.body)
        status, events = self.request('/api/tasks/r1/events?after=0')
        self.assertEqual(status, 200)
        self.assertEqual(len(events), 1)
        self.assertEqual(self.request('/api/tasks/r1/events?after=' + str(events[0]['seq']))[1], [])
        self.assertEqual(self.request('/api/tasks/r1/cancel', {})[0], 200)
        task = self.request('/api/tasks/r1')[1]
        self.assertEqual(task['execution'], 'cancelled')
        self.assertEqual(self.request('/api/tasks/r1/report')[0], 409)

    def test_export_matches_saved_version_and_citations(self):
        self.request('/api/tasks', self.body)
        store = self.service.store
        store.claim('r1')
        store.save('r1', 1, {'read': {'e1': ['f0']}}, 'test', {})
        report = store.finish('r1', 1, 'answered', {'calculations': []})
        self.assertEqual(self.request('/api/tasks/r1/report')[1], report)
        self.assertEqual(self.request('/api/tasks/r1/source?id=e1')[0], 200)
        self.assertEqual(self.request('/api/tasks/r1/source?id=outside')[0], 400)
        self.request('/api/tasks', {**self.body, 'request_id': 'r2', 'task_id': 'r1'})
        self.assertEqual(self.request('/api/tasks/r1/report')[0], 409)
        self.assertEqual(self.request('/api/tasks/r1/source?id=e1')[0], 400)

    def test_origin_and_snapshot_guards(self):
        self.assertEqual(self.request('/api/tasks', self.body, {'Origin': 'https://external.example'})[0], 403)
        wrong = {**self.body, 'payload': {**self.body['payload'], 'snapshot_id': 'other'}}
        self.assertEqual(self.request('/api/tasks', wrong)[0], 400)
        self.assertEqual(self.request('/api/catalogue', headers={'Host': 'external.example'})[0], 403)

    def test_initializing_worker_does_not_accept_tasks(self):
        self.service.ready = False
        self.assertEqual(self.request('/api/tasks', self.body)[0], 503)
        self.assertFalse(self.request('/api/catalogue')[1]['ready'])

    def test_database_cannot_switch_source_snapshots(self):
        path = Path(self.service.corpus_path)
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'different source snapshot'):
            Service(self.service.store.path, str(path), 's1', None)

    def test_static_assets_and_no_path_traversal(self):
        for path in ('/', '/app.js', '/style.css'):
            with self.opener.open(self.url + path) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.headers['Content-Security-Policy'])
                self.assertGreater(len(response.read()), 100)
        self.assertEqual(self.request('/../configs/model_choice.json')[0], 404)


if __name__ == '__main__':
    unittest.main()
