"""Independent completeness, citation and transactional audit of frozen M3 runs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from src.evaluation.vendor.finqa_execution import eval_program
from src.evidence.identity import evidence_identity
from src.model.financial import decode_prediction
from src.tools.benchmark_tools import extract_numeric_facts


def audit(run: Path) -> dict:
    settings = json.loads((run / 'settings.json').read_text())
    root = Path(settings.get('tasks_root', 'data/staged/m3_acceptance_v1'))
    lock_path = (root / 'lock.json' if settings.get('tasks_root') else
                 Path('configs/m3_acceptance_lock.json'))
    lock = json.loads(lock_path.read_text())
    if settings['lock'] != lock:
        raise ValueError('agent run does not match its frozen input lock')
    for name, digest in lock['sha256'].items():
        assert hashlib.sha256((root / f'{name}.json').read_bytes()).hexdigest() == digest
    tasks = json.loads((root / 'tasks.json').read_text())
    gold = json.loads((root / 'gold.json').read_text())
    corpus = json.loads((root / 'corpus.json').read_text())
    records = [json.loads(line) for line in (run / 'records.jsonl').read_text().splitlines()]
    turns = [json.loads(line) for line in (run / 'turns.jsonl').read_text().splitlines()]
    expected = {(t['task_id'], i) for t in tasks for i in range(len(t['turns']))}
    assert len(turns) == len(expected)
    assert {(t['task_id'], t['turn']) for t in turns} == expected
    assert len(records) == len(tasks)
    assert {r['task_id'] for r in records} == {t['task_id'] for t in tasks}
    task_map = {t['task_id']: t for t in tasks}
    db = sqlite3.connect(run / 'tasks.sqlite')
    report_rows = db.execute('SELECT task_id,revision,body FROM reports').fetchall()
    reports = {(tid, revision): json.loads(body) for tid, revision, body in report_rows}
    committed = db.execute("SELECT task_id,revision FROM events WHERE kind='report_committed'").fetchall()
    assert len(committed) == len(set(committed)) == len(reports)
    assert set(committed) == set(reports)
    assert not db.execute("SELECT id FROM tasks WHERE execution IN ('running','queued')").fetchall()
    facts_checked = 0
    state_counts = Counter()
    for entry in turns:
        task_id, i, row = entry['task_id'], entry['turn'], entry['result']
        task = task_map[task_id]
        state_counts[f"{row['execution']}/{row['business']}"] += 1
        saved = reports.get((task_id + ':turn0', row['revision']))
        assert saved == row['report']
        assert 0 <= row['attempts'] <= 10
        if row['business'] == 'awaiting_input':
            assert row['execution'] == 'succeeded' and saved is None
        if row['execution'] != 'succeeded':
            assert saved is None
        expected_answer = gold[task_id]['expected'][i]
        calculations = (saved or {}).get('calculations', [])
        for calc in calculations:
            invalid, value = eval_program(calc['program'], [])
            assert not invalid and value == calc['value']
            for fact in calc['facts'].values():
                document = corpus[fact['evidence_id']]
                periods = task['period'] if isinstance(task['period'], list) else [task['period']]
                assert document['company'] == task['company'] and document['period'] in periods
                assert fact in extract_numeric_facts(document['evidence']).values()
                facts_checked += 1
        if isinstance(expected_answer, str):
            passed = row['execution'] == 'succeeded' and row['business'] == expected_answer
            if expected_answer == 'awaiting_input':
                passed = passed and saved is None
        else:
            numeric = (row['execution'] == 'succeeded' and row['business'] == 'answered'
                       and [c['value'] for c in calculations] == expected_answer)
            ids = gold[task_id]['evidence_ids'][i]
            supported = len(calculations) == len(ids)
            for calc, target in zip(calculations, ids):
                target_doc = corpus[target]
                identity = evidence_identity(target_doc['evidence'], target_doc['source_report_id'])
                cited = {evidence_identity(corpus[f['evidence_id']]['evidence'],
                         corpus[f['evidence_id']]['source_report_id']) for f in calc['facts'].values()}
                supported = supported and cited == {identity}
            assert row['numeric_correct'] == numeric and row['source_supported'] == supported
            passed = numeric and supported
        assert row['passed'] == passed
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['tasks'] == len(records) and summary['turns'] == len(turns)
    for record in records:
        rows = [t['result'] for t in turns if t['task_id'] == record['task_id']]
        assert record['turns'] == rows
        assert record['all_turns_passed'] == all(row['passed'] for row in rows)
        if settings.get('tasks_root'):
            fixed = record.get('fixed')
            if task_map[record['task_id']]['category'] != 'calculation':
                assert fixed is None
                continue
            assert fixed is not None
            selected = fixed.get('selected_evidence')
            if selected is None:
                assert fixed.get('error') and not fixed['correct'] and not fixed['source_supported']
                continue
            document = corpus[selected]
            task = task_map[record['task_id']]
            periods = task['period'] if isinstance(task['period'], list) else [task['period']]
            assert document['company'] == task['company'] and document['period'] in periods
            target = corpus[gold[record['task_id']]['evidence_ids'][0][0]]
            assert fixed['source_supported'] == (
                evidence_identity(document['evidence'], document['source_report_id']) ==
                evidence_identity(target['evidence'], target['source_report_id']))
            raw = {key: fixed[key] for key in ('text', 'stopped_on_eos') if key in fixed}
            if 'text' not in raw and fixed.get('error'):
                raw['error'] = fixed['error']
            decoded = decode_prediction(raw, extract_numeric_facts(document['evidence']), 'calculator')
            correct = ('error' not in decoded and decoded.get('prediction') ==
                       gold[record['task_id']]['expected'][0][0])
            assert fixed['correct'] == correct
            if 'error' not in decoded:
                assert decoded['calculation'] == fixed['calculation']
    if settings.get('tasks_root'):
        controls = [r['fixed'] for r in records if r.get('fixed') is not None]
        assert summary['fixed_30_calculation_tasks'] == {
            'total': len(controls), 'correct_with_source': sum(
                row['correct'] and row['source_supported'] for row in controls)}
    categories = {task['category'] for task in tasks}
    assert set(summary['categories']) == categories
    for category in categories:
        selected = [r for r in records if task_map[r['task_id']]['category'] == category]
        assert summary['categories'][category] == {
            'passed': sum(r['all_turns_passed'] for r in selected), 'total': len(selected)}
    failures = Counter(json.loads(body).get('error', kind) for kind, body in db.execute(
        "SELECT kind,body FROM events WHERE kind IN ('failed','cancelled','interrupted')"))
    started_calls = db.execute("SELECT count(*) FROM events WHERE kind='generation_started'").fetchone()[0]
    generated = [json.loads(body) for (body,) in db.execute(
        "SELECT body FROM events WHERE kind='generation_finished'")]
    cost = {'started_calls': started_calls, 'returned_calls': len(generated),
            'missing_results_after_interruption': started_calls - len(generated),
            'reserved_output_tokens': started_calls * 512,
            'observed_output_tokens': sum(r.get('new_tokens', 0) for r in generated),
            'observed_generation_seconds': sum(r.get('latency_s', 0) for r in generated)}
    prompt_tokens = sorted(r['prompt_tokens'] for r in generated if 'prompt_tokens' in r)
    observed, repeated = Counter(), Counter()
    seen = set()
    for task_id, revision, body in db.execute(
            "SELECT task_id,revision,body FROM events WHERE kind='tool_observation' ORDER BY seq"):
        event = json.loads(body)
        name, arguments = event['tool'], event['arguments']
        if name in ('search_documents', 'get_evidence'):
            observed[name] += 1
            value = (' '.join(arguments['query'].casefold().split()) if name == 'search_documents'
                     else arguments['evidence_id'])
            key = (task_id, revision, name, value)
            repeated[name] += key in seen
            seen.add(key)
        if name == 'submit_report' and 'error' in event['observation']:
            observed['invalid_submission'] += 1
    context = {'input_over_budget': sum(r.get('error') == 'input_over_budget' for r in generated),
               'prompt_tokens_count': len(prompt_tokens),
               'prompt_tokens_max': max(prompt_tokens, default=0),
               'prompt_tokens_p50': prompt_tokens[len(prompt_tokens) // 2] if prompt_tokens else None,
               'prompt_tokens_p95': prompt_tokens[int(.95 * (len(prompt_tokens) - 1))] if prompt_tokens else None,
               'tool_calls': dict(observed), 'repeated_calls_within_revision': dict(repeated),
               'rejected_generations': db.execute(
                   "SELECT count(*) FROM events WHERE kind='tool_rejected'").fetchone()[0]}
    db.close()
    result = {'passed': True, 'tasks': len(records), 'turns': len(turns),
              'records_sha256': hashlib.sha256((run / 'records.jsonl').read_bytes()).hexdigest(),
              'turns_sha256': hashlib.sha256((run / 'turns.jsonl').read_bytes()).hexdigest(),
              'reports': len(reports), 'scalar_references_checked': facts_checked,
              'states': dict(state_counts), 'failures': dict(failures), 'cost': cost,
              'context': context,
              'meaning': 'Artifact and runtime invariants only; business failures remain failures.'}
    (run / 'validation.json').write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run), indent=2))


if __name__ == '__main__':
    main()
