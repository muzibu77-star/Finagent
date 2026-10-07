"""Real browser UI checks against a deterministic synthetic task store, without GPU."""

import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

from src.service.server import Service, handler_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    corpus = {'fixture': {'company': 'A', 'period': '2023', 'source_report_id': 'synthetic/A/2023',
              'evidence': {'evidence_id': 'fixture', 'table': [['Revenue', '10']],
                           'pre_text': ['<img src=x onerror="window.injected=true">']}}}
    corpus_path = args.output_dir / 'corpus.json'
    corpus_path.write_text(json.dumps(corpus))
    service = Service(str(args.output_dir / 'tasks.sqlite'), str(corpus_path), 'synthetic', None)
    service.ready = True
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    checks = {}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1280, 'height': 1000})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.wait_for_function("() => document.getElementById('connection').textContent==='模型已就绪'")
            page.select_option('#company', 'A')
            page.fill('#period', '2023')
            page.fill('#question', 'Synthetic revenue?')
            lost = [False]
            def lose_response(route):
                if route.request.method == 'POST' and not lost[0]:
                    lost[0] = True
                    route.fetch()
                    route.abort('failed')
                else:
                    route.continue_()
            page.route('**/api/tasks', lose_response)
            page.click('#submit')
            page.wait_for_function("() => localStorage.getItem('finagent.task')!==null && localStorage.getItem('finagent.pending')===null")
            task_id = page.evaluate("localStorage.getItem('finagent.task')")
            with service.store.transaction() as db:
                checks['lost_response_retry_idempotent'] = db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
            page.reload()
            page.wait_for_function("() => document.getElementById('events').children.length>0")
            checks['reload_reconnects_events'] = service.store.get(task_id)['execution'] == 'queued'
            page.click('#cancel')
            page.wait_for_function("() => document.getElementById('status').textContent==='已取消'")
            checks['queued_cancellation'] = service.store.get(task_id)['execution'] == 'cancelled'
            page.click('#new')
            page.fill('#question', 'Revenue after clarification?')
            page.select_option('#company', '')
            page.click('#submit')
            page.wait_for_function("() => localStorage.getItem('finagent.task')!==null")
            task_id = page.evaluate("localStorage.getItem('finagent.task')")
            service.store.claim(task_id)
            service.store.stop(task_id, 1, 'succeeded', 'awaiting_input', {'question': '请选择公司'})
            page.wait_for_function("() => document.getElementById('clarification').textContent==='请选择公司'")
            page.select_option('#company', 'A')
            page.fill('#question', '公司 A')
            page.click('#submit')
            page.wait_for_function("() => document.getElementById('status').textContent==='已排队'")
            task = service.store.get(task_id)
            checks['clarification_advances_revision'] = task['revision'] == 2
            service.store.claim(task_id)
            service.store.save(task_id, 2, {'read': {'fixture': ['f0']}}, 'tool_observation', {'tool': 'calculate'})
            report = service.store.finish(task_id, 2, 'answered', {'calculations': [
                {'value': 10, 'facts': {'f0': {'evidence_id': 'fixture'}}}]})
            page.wait_for_function("() => document.getElementById('status').textContent==='已回答'")
            page.click('.citations button')
            page.wait_for_selector('dialog[open]')
            checks['source_text_not_executable'] = page.evaluate('window.injected===undefined')
            page.click('#closeSource')
            with page.expect_download() as event:
                page.click('#export')
            export_path = args.output_dir / 'download.json'
            event.value.save_as(export_path)
            checks['export_exact_committed_report'] = json.loads(export_path.read_text()) == report
            page.screenshot(path=str(args.output_dir / 'desktop.png'), full_page=True)
            page.set_viewport_size({'width': 390, 'height': 844})
            checks['mobile_no_horizontal_overflow'] = page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            page.screenshot(path=str(args.output_dir / 'mobile.png'), full_page=True)
            checks['no_javascript_errors'] = not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    result = {'passed': all(checks.values()), 'checks': checks, 'limits': 'Synthetic store fixture; UI behavior only, real GPU service acceptance is separate.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
