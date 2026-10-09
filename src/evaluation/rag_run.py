"""Resumable frozen document retrieval ablation; no generation-quality claims."""

import argparse
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import statistics
import time

from src.evaluation.retrieval_run import metrics
from src.retrieval.neural import NeuralIndex, page_ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--tasks-dir', type=Path, required=True)
    parser.add_argument('--index-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = json.loads(Path('configs/jd_retrieval.json').read_text())
    lock = json.loads((args.tasks_dir / 'lock.json').read_text())
    if hashlib.sha256(args.corpus.read_bytes()).hexdigest() != lock['corpus_sha256']:
        raise ValueError('corpus differs from frozen query protocol')
    task_path = args.tasks_dir / 'tasks.json'
    if hashlib.sha256(task_path.read_bytes()).hexdigest() != lock['sha256']['tasks']:
        raise ValueError('query hash mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    settings = {'config': config, 'lock': lock, 'source_sha256': {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
        ('src/retrieval/neural.py', 'src/retrieval/report_index.py', __file__)}}
    settings_path = args.output_dir / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('resume settings changed')
    settings_path.write_text(json.dumps(settings, indent=2))
    records_path = args.output_dir / 'records.jsonl'
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    done = {row['id'] for row in records}
    tasks = json.loads(task_path.read_text())
    documents = json.loads(args.corpus.read_text())
    gpu_lock = Path('/tmp/finagent-cuda0.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    index = NeuralIndex(documents, args.index_dir, config)
    started = time.monotonic()
    try:
        for task in tasks:
            if task['id'] in done:
                continue
            start = time.monotonic()
            rankings = index.compare(task['query'], set(documents))
            results = {}
            for mode, chunks in rankings.items():
                pages = page_ranking(index.chunks, chunks)
                results[mode] = {'ranking': pages, 'chunks': [
                    {'chunk_id': key, **index.chunks[key]} for key in chunks[:5]],
                    **metrics(pages, task['relevant'])}
            record = {'id': task['id'], 'split': task['split'], 'company': task['company'],
                      'latency_s_all_modes': time.monotonic() - start,
                      'stage_latency_s': index.last_timings, 'results': results}
            with records_path.open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
            records.append(record)
            logging.info('%d/%d elapsed=%.1fs records=%s', len(records), len(tasks),
                         time.monotonic() - started, records_path)
    finally:
        index.close()
        gpu_lock.close()
    summary = {'tasks': len(records), 'metrics': {}}
    for split in ('dev', 'test'):
        rows = [row for row in records if row['split'] == split]
        summary['metrics'][split] = {mode: {metric: statistics.mean(
            row['results'][mode][metric] for row in rows) for metric in
            ('recall_at_5', 'mrr', 'ndcg_at_5', 'top1_source_support')}
            for mode in ('bm25', 'dense', 'hybrid', 'cross_encoder')} if rows else {}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
