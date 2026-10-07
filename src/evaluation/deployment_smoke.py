"""Real-model HTTP deployment demo, including interruption and persisted report recovery."""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    demo = json.loads(Path('configs/m6_demo.json').read_text())
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    opener = build_opener(ProxyHandler({}))
    database = args.output_dir / 'tasks.sqlite'
    python = str(Path('.venv/bin/python').absolute())
    command = [python, '-m', 'src.service.server', '--port', str(port), '--database', str(database)]
    checks, records = {}, {}

    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        with opener.open(Request(url + path, data=data,
                         headers={'Content-Type': 'application/json'}), timeout=10) as response:
            return json.loads(response.read())

    def wait_ready(process):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('service exited before ready; inspect server log')
            try:
                state = request('/api/catalogue')
                if state['error']:
                    raise RuntimeError(state['error'])
                if state['ready']:
                    return
            except URLError:
                pass
            time.sleep(0.25)
        raise TimeoutError('model load timeout')

    def submit(request_id, question, *, task_id=None, period=None, company='ANSS'):
        body = {'request_id': request_id, 'payload': {'question': question,
                'company': company, 'period': period or demo['period'], 'snapshot_id': demo['snapshot_id']}}
        if task_id:
            body['task_id'] = task_id
        return request('/api/tasks', body)

    def wait_task(task_id):
        deadline = time.monotonic() + 220
        while time.monotonic() < deadline:
            row = request('/api/tasks/' + task_id)
            if row['execution'] not in ('queued', 'running'):
                return row
            time.sleep(0.1)
        raise TimeoutError('task did not terminate')

    log = (args.output_dir / 'server.log').open('a')
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(process)
        submit('demo-calc', demo['question'])
        first = wait_task('demo-calc')
        records['calculation'] = first
        checks['real_calculation'] = (first['business'] == 'answered' and
            first['report']['calculations'][0]['value'] == demo['expected'])
        if first['report']:
            export = request('/api/tasks/demo-calc/report')
            checks['export_matches_report'] = export == first['report']
            cited = next(iter(export['calculations'][0]['facts'].values()))['evidence_id']
            source = request('/api/tasks/demo-calc/source?id=' + cited)
            checks['citation_resolves'] = source['evidence']['evidence_id'] == cited
        submit('demo-followup', demo['followup'], task_id='demo-calc')
        followup = wait_task('demo-calc')
        records['followup'] = followup
        checks['continuous_followup'] = (followup['revision'] == 2 and followup['business'] == 'answered'
            and followup['report']['calculations'][0]['value'] == demo['followup_expected'])
        submit('demo-no-data', demo['question'], period='2099')
        insufficient = wait_task('demo-no-data')
        records['insufficient'] = insufficient
        checks['insufficient_evidence'] = insufficient['business'] == 'insufficient_evidence'
        submit('demo-clarify', demo['question'], company=None)
        clarify = wait_task('demo-clarify')
        records['clarification'] = clarify
        checks['clarification_pauses_without_report'] = clarify['business'] == 'awaiting_input' and clarify['report'] is None
        submit('demo-cancel', demo['question'])
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            events = request('/api/tasks/demo-cancel/events')
            if any(e['kind'] == 'generation_started' for e in events):
                break
            time.sleep(0.02)
        else:
            raise TimeoutError('generation did not start for cancellation')
        started = time.monotonic()
        request('/api/tasks/demo-cancel/cancel', {})
        cancelled = wait_task('demo-cancel')
        records['cancellation'] = cancelled
        records['cancellation_seconds'] = time.monotonic() - started
        checks['real_running_cancellation'] = cancelled['execution'] == 'cancelled' and cancelled['report'] is None
        before = request('/api/tasks/demo-calc/events')
        process.kill()  # Only the subprocess started above; deliberate crash after commits.
        process.wait(timeout=30)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        wait_ready(process)
        after = request('/api/tasks/demo-calc/events')
        checks['restart_preserves_committed_report'] = request('/api/tasks/demo-calc')['report'] == followup['report']
        checks['restart_does_not_repeat_events'] = before == after
        records['recovered_task'] = request('/api/tasks/demo-calc')
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
                checks['graceful_resource_release'] = False
            else:
                checks['graceful_resource_release'] = process.returncode == 0
        log.close()
        (args.output_dir / 'records.json').write_text(json.dumps(records, indent=2))
        required = {'real_calculation', 'export_matches_report', 'citation_resolves',
                    'continuous_followup', 'insufficient_evidence', 'clarification_pauses_without_report',
                    'real_running_cancellation', 'restart_preserves_committed_report',
                    'restart_does_not_repeat_events', 'graceful_resource_release'}
        result = {'passed': required <= checks.keys() and all(checks.values()), 'checks': checks,
                  'cwd': str(Path.cwd()), 'command': command, 'demo': demo,
                  'requirements_sha256': hashlib.sha256(Path('requirements.txt').read_bytes()).hexdigest()}
        (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
