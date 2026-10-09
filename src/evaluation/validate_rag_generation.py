"""Independently rescore frozen RAG outputs, citations and stage diagnostics."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from src.data.leakage import content_text
from src.evaluation.full_benchmark import tatqa_prediction, tatqa_score
from src.evaluation.retrieval_run import metrics
from src.evaluation.statistics import cluster_interval

MODES = ('bm25', 'dense', 'hybrid', 'cross_encoder')


def validate_retrieval(retrieval: dict, tasks: dict, documents: dict) -> None:
    texts = {key: content_text(row['evidence']) for key, row in documents.items()}
    for key, row in retrieval.items():
        if set(row['results']) != set(MODES):
            raise ValueError('retrieval modes differ')
        for result in row['results'].values():
            ranking = result['ranking']
            if len(set(ranking)) != len(ranking) or not set(ranking) <= texts.keys():
                raise ValueError('invalid retrieval page ranking')
            for chunk in result['chunks']:
                text = texts[chunk['page_id']]
                if (chunk['page_id'] not in ranking
                        or not 0 <= chunk['start'] < chunk['end'] <= len(text)
                        or text[chunk['start']:chunk['end']] != chunk['text']):
                    raise ValueError('retrieved excerpt differs from original page span')
            expected = metrics(ranking, tasks[key]['relevant'])
            if any(result[name] != value for name, value in expected.items()):
                raise ValueError('retrieval metrics differ from page labels')


def validate_rows(rows: list[dict], tasks: dict, gold: dict,
                  retrieval: dict, parsed: set) -> dict:
    expected = {(key, mode) for key in tasks for mode in MODES}
    pairs = [(row['id'], row['mode']) for row in rows]
    if len(pairs) != len(expected) or set(pairs) != expected:
        raise ValueError('missing, duplicate or unexpected RAG observations')
    diagnostics = {}
    for row in rows:
        key, mode = row['id'], row['mode']
        task = tasks[key]
        results = retrieval[key]['results']
        supplied = {chunk['page_id'] for chunk in results[mode]['chunks']}
        candidates = set(results['hybrid' if mode == 'cross_encoder' else mode]['ranking'])
        relevant = set(task['relevant'])
        if (row['supplied'] != sorted(supplied)
                or row['labels_available'] != (key in gold)
                or any(row[field] != task[field] for field in
                       ('split', 'company', 'source_report_id'))):
            raise ValueError('RAG observation differs from frozen inputs')
        score = {'em': 0.0, 'f1': 0.0, 'scale': 0.0}
        supported = False
        if not row.get('error') and row.get('stopped_on_eos'):
            try:
                prediction = json.loads(row['text'])
                if not isinstance(prediction, dict) or set(prediction) != {
                        'answer', 'scale', 'citations'}:
                    raise ValueError('invalid output fields')
                citations = prediction.pop('citations')
                if not isinstance(citations, list) or any(
                        not isinstance(c, str) for c in citations):
                    raise ValueError('invalid citations')
                prediction = tatqa_prediction(json.dumps(prediction))
                supported = (bool(citations) and set(citations) <= supplied
                             and bool(set(citations) & relevant))
                if key in gold:
                    score = tatqa_score(prediction, gold[key])
            except (ValueError, TypeError, KeyError):
                pass  # Invalid generations remain failures in the denominator.
        if row['citation_supported'] != supported or any(
                row[metric] != value for metric, value in score.items()):
            raise ValueError('RAG score differs from raw generation')
        missing = {'parsing': sorted(relevant - parsed),
                   'recall': sorted(relevant - candidates),
                   'reranking': sorted(relevant - supplied)}
        stage = next((name for name, pages in missing.items() if pages), None)
        if stage is None and not (supported and (score['em'] == 1 if key in gold else True)):
            stage = 'generation'
        if row['stage'] != stage:
            raise ValueError('RAG stage differs from saved pipeline evidence')
        diagnostics[f'{key}/{mode}'] = {
            'missing_pages': missing, 'citation_supported': supported,
            'answer_failed': score['em'] != 1 if key in gold else None,
            'first_failed_stage': stage}
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tasks-dir', type=Path, required=True)
    parser.add_argument('--retrieval-dir', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads((args.run / 'settings.json').read_text())
    lock = json.loads((args.tasks_dir / 'lock.json').read_text())
    if settings['lock'] != lock or hashlib.sha256(args.corpus.read_bytes()).hexdigest() != lock['corpus_sha256']:
        raise ValueError('RAG corpus or lock changed')
    for name, digest in lock['sha256'].items():
        if hashlib.sha256((args.tasks_dir / f'{name}.json').read_bytes()).hexdigest() != digest:
            raise ValueError('frozen RAG inputs changed')
    retrieval_path = args.retrieval_dir / 'records.jsonl'
    if hashlib.sha256(retrieval_path.read_bytes()).hexdigest() != settings['retrieval_sha256']:
        raise ValueError('retrieval records changed')
    tasks = {r['id']: r for r in json.loads((args.tasks_dir / 'tasks.json').read_text())}
    retrieved = [json.loads(s) for s in retrieval_path.read_text().splitlines()]
    retrieval = {r['id']: r for r in retrieved}
    if len(retrieved) != len(tasks) or retrieval.keys() != tasks.keys():
        raise ValueError('retrieval membership differs')
    split = settings.get('split', 'all')
    if split not in ('all', 'dev', 'test'):
        raise ValueError('unknown RAG evaluation split')
    if split != 'all':
        tasks = {key: task for key, task in tasks.items() if task['split'] == split}
        retrieval = {key: row for key, row in retrieval.items() if key in tasks}
    gold = json.loads((args.tasks_dir / 'gold.json').read_text())
    docs = json.loads(args.corpus.read_text())
    validate_retrieval(retrieval, tasks, docs)
    parsed = {key for key, row in docs.items() if row.get('parse_status') != 'empty'}
    records_path = args.run / 'records.jsonl'
    rows = [json.loads(s) for s in records_path.read_text().splitlines()]
    diagnostics = validate_rows(rows, tasks, gold, retrieval, parsed)
    summary = {'calls': len(rows), 'results': {}}
    for split in ('dev', 'test'):
        for mode in MODES:
            selected = [r for r in rows if r['split'] == split and r['mode'] == mode]
            if not selected:
                continue
            labeled = [r for r in selected if r['labels_available']]
            summary['results'][f'{split}/{mode}'] = {
                'inputs': len(selected), 'scored': len(labeled),
                'em': cluster_interval(labeled, 'em'), 'f1': cluster_interval(labeled, 'f1'),
                'stages': dict(Counter(r['stage'] or 'passed' for r in selected)),
                'supported_citations': sum(r['citation_supported'] for r in selected)}
    if summary != json.loads((args.run / 'summary.json').read_text()):
        raise ValueError('RAG summary differs from independent rescoring')
    (args.run / 'stage_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    result = {'passed': True, 'calls': len(rows),
              'records_sha256': hashlib.sha256(records_path.read_bytes()).hexdigest(),
              'limits': 'Page coverage and citation overlap are proxies, not semantic entailment. '
                        'Missing pages at several stages may share an upstream cause; '
                        'diagnostics do not prove independent causal failures.'}
    (args.run / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
