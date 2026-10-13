"""Replay an actual dated restatement, preserving scanned source provenance."""

import argparse
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pypdfium2 as pdfium

from src.agent.runner import Runner
from src.data.report_corpus import ReportCorpus
from src.harness.store import Store


def replay_state(config: dict, corpus: ReportCorpus, output: Path) -> list[dict]:
    """Exercise real sources with scripted actions; this is not model quality."""
    store = Store(str(output / 'state.sqlite'))
    correction_day = date.fromisoformat(config['correction']['published_at'])
    before_day = (correction_day - timedelta(days=1)).isoformat()
    records = []
    for metric_index, (metric, before, _, after) in enumerate(config['rows']):
        task_id = None
        for revision, (cutoff, source, value) in enumerate([
            (before_day, 'original', before),
            (correction_day.isoformat(), 'correction', after),
            (before_day, 'original', before),
        ], 1):
            payload = {'question': metric, 'company': config['company'],
                       'period': config['period'], 'snapshot_id': config['snapshot_id'],
                       'as_of': cutoff}
            receipt = store.submit(f'row-{metric_index}-revision-{revision}', payload, task_id)
            task_id = receipt['task_id']
            assert store.get(task_id)['state'] == {'history': []}
            prompts = []

            def scripted_model(messages, tools, cancelled, deadline):
                body = json.loads(messages[1]['content'])
                prompts.append(body)
                assert not body['history']
                observations = [json.loads(m['content']) for m in messages
                                if m['role'] == 'tool' and m['name'] == 'get_evidence']
                documents = [o['document'] for o in observations if 'document' in o]
                assert {d['evidence']['evidence_id'] for d in documents} <= {source}
                assert set(body['read_evidence_ids']) <= {source}
                call_number = len(prompts)
                if call_number <= 2:
                    name, arguments = 'search_documents', {'query': metric}
                elif call_number == 3:
                    latest = json.loads(messages[-1]['content'])
                    assert latest['duplicate_search']
                    hits = latest['hits']
                    assert hits and {h['evidence_id'] for h in hits} == {source}
                    name, arguments = 'get_evidence', {'evidence_id': hits[0]['evidence_id']}
                elif call_number == 4:
                    # Scripted reference selection validates the runtime's source guards.
                    matches = [key for key, fact in observations[-1]['facts'].items()
                               if fact.split(' @ ', 1)[0].replace(',', '') == value.replace(',', '')]
                    if not matches:
                        raise ValueError(f'reference value absent from context: {metric}')
                    name, arguments = 'calculate', {'steps': [
                        {'operation': 'multiply', 'args': [matches[0], 'const_1']}]}
                else:
                    name, arguments = 'submit_report', {
                        'calculation_ids': list(body['calculations']), 'outcome': 'answered'}
                # Native Qwen tool syntax, parsed by the actual Runner.
                text = (f'<tool_call>\n<function={name}>\n' + ''.join(
                    f'<parameter={key}>\n'
                    f'{item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)}'
                    '\n</parameter>\n'
                    for key, item in arguments.items()) + '</function>\n</tool_call>')
                return {'text': text, 'stopped_on_eos': True, 'new_tokens': 0}

            row = Runner(store, corpus, scripted_model, context_management=True).run(task_id)
            assert row['execution'] == 'succeeded', row['state'].get('stop_detail')
            assert row['business'] == 'answered'
            report = row['report']
            assert report['request_scope']['as_of'] == cutoff
            calculation = report['calculations'][0]
            assert Decimal(str(calculation['value'])) == Decimal(value.replace(',', ''))
            assert {f['evidence_id'] for f in calculation['facts'].values()} == {source}
            observations = [e['body']['observation'] for e in store.events(task_id)
                            if e['revision'] == revision and e['kind'] == 'tool_observation'
                            and e['body']['tool'] == 'search_documents']
            if revision == 2:
                assert not observations[0].get('cache_replay')
            if revision == 3:
                assert observations[0].get('cache_replay')
            records.append({'metric': metric, 'revision': revision, 'as_of': cutoff,
                            'source': source, 'value': calculation['value'],
                            'history_cleared': True, 'calls': len(prompts),
                            'cache_replay': bool(observations[0].get('cache_replay'))})
            with store.transaction() as db:
                saved = db.execute('SELECT revision, body FROM reports WHERE task_id=?',
                                   (task_id,)).fetchall()
            assert len(saved) == revision
            assert json.loads(saved[0]['body'])['request_scope']['as_of'] == before_day
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/jd_restatement.json'))
    parser.add_argument('--state-replay', action='store_true')
    args = parser.parse_args()
    config_path = args.config
    config = json.loads(config_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    documents = {}
    for name in ('original', 'correction'):
        source = config[name]
        raw = Path(source['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise ValueError('original source changed')
        with pdfium.PdfDocument(raw) as pdf:
            page = pdf[source['page']-1]
            tp = page.get_textpage()
            text = tp.get_text_bounded()
            tp.close()
            bitmap = page.render(scale=2)
            bitmap.to_pil().save(args.output_dir/f'{name}.png')
            bitmap.close()
            page.close()
        if name == 'original':
            if any(row[1] not in text for row in config['rows']):
                raise ValueError('restatement original values not in original report')
        else:
            for _, before, delta, after in config['rows']:
                a, b, c = [Decimal(v.replace(',', '')) for v in (before, delta, after)]
                if a+b != c:
                    raise ValueError('transcribed correction arithmetic mismatch')
            if source.get('extraction') == 'PDFium':
                if any(value not in text for row in config['rows'] for value in row[1:]):
                    raise ValueError('correction values not in original PDF text')
            else:
                text = (config['period']+'年12月31日，合并财务报表，金额：人民币元。'
                        '列为更正前金额、差错更正累计影响金额、更正后金额。\n'
                        + '\n'.join(' | '.join(row) for row in config['rows']))
        key = name
        documents[key] = {'company': config['company'], 'report_period': config['period'],
            'source_report_id': name, 'version': source['sha256'],
            'published_at': source['published_at'], 'ingested_at': datetime.now(timezone.utc).isoformat(),
            'supersedes': 'original' if name == 'correction' else None,
            'url': source['url'], 'page': source['page'], 'source_sha256': source['sha256'],
            'extraction_method': (source.get('extraction', 'reviewed transcription')
                                  if name == 'correction' else 'PDFium'),
            'evidence': {'evidence_id': key, 'table': [], 'pre_text': [text], 'post_text': []}}
    path = args.output_dir/'corpus.json'
    path.write_text(json.dumps(documents, ensure_ascii=False, indent=2))
    receipt = {'snapshot_id': config['snapshot_id'],
               'corpus_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output_dir/'receipt.json').write_text(json.dumps(receipt, indent=2))
    corpus = ReportCorpus(str(path), config['snapshot_id'])
    records = []
    try:
        first = date.fromisoformat(config['original']['published_at'])
        corrected = date.fromisoformat(config['correction']['published_at'])
        for cutoff, expected in [((first-timedelta(days=1)).isoformat(), set()),
                                 (first.isoformat(), {'original'}),
                                 ((corrected-timedelta(days=1)).isoformat(), {'original'}),
                                 (corrected.isoformat(), {'correction'})]:
            payload = {'company': config['company'], 'period': config['period'],
                       'snapshot_id': config['snapshot_id'], 'as_of': cutoff}
            allowed = corpus.allowed(payload)
            assert allowed == expected
            for metric, before, _, after in config['rows']:
                hits = corpus.search(metric, allowed)
                assert {hit['evidence_id'] for hit in hits} <= allowed
                if allowed:
                    assert hits
                    value = before if 'original' in allowed else after
                    assert value in documents[hits[0]['evidence_id']]['evidence']['pre_text'][0]
                records.append({'as_of': cutoff, 'metric': metric, 'allowed': sorted(allowed), 'hits': hits})
            for key in set(documents)-allowed:
                try:
                    corpus.read(key, allowed)
                except ValueError:
                    continue
                raise AssertionError('future or superseded source could be read')
        state_records = replay_state(config, corpus, args.output_dir) if args.state_replay else []
    finally:
        corpus.close()
    (args.output_dir/'summary.json').write_text(json.dumps({'passed': True,
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(), 'receipt': receipt,
        'records': records, 'state_records': state_records,
        'state_protocol': 'scripted actions over real sources; not model quality',
        'limits': config['review']}, ensure_ascii=False, indent=2))
    print(json.dumps({'passed': True, 'cases': len(records), 'issuers': 1}))


if __name__ == '__main__':
    main()
