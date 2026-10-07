"""Fixed page relevance comparison; retrieval metrics do not imply answer quality."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from src.data.report_corpus import ReportCorpus


def metrics(ranking: list[str], relevant: list[str]) -> dict:
    flags = [key in relevant for key in ranking]
    first = next((i + 1 for i, matched in enumerate(flags) if matched), None)
    dcg = sum(1 / math.log2(i + 2) for i, matched in enumerate(flags[:5]) if matched)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(5, len(relevant))))
    return {'recall_at_5': sum(flags[:5]) / len(relevant),
            'mrr': 1 / first if first else 0,
            'ndcg_at_5': dcg / ideal,
            'top1_source_support': int(bool(flags) and flags[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    task_path = Path('configs/m4_retrieval_tasks.json')
    config = json.loads(task_path.read_text())
    corpus = ReportCorpus('data/staged/reports_v2/corpus.json', 'reports-v1')
    records = []
    for task in config['tasks']:
        if not set(task['relevant']) <= corpus.documents.keys():
            raise ValueError('relevance label outside frozen corpus')
        for mode in ('bm25', 'fusion', 'rerank'):
            start = time.perf_counter()
            ranking = corpus.index.search(task['question'], set(corpus.documents), mode, len(corpus.documents))
            elapsed = (time.perf_counter() - start) * 1000
            records.append({'task_id': task['id'], 'mode': mode, 'ranking': ranking,
                            'latency_ms': elapsed, **metrics(ranking, task['relevant'])})
    summaries = {}
    for mode in ('bm25', 'fusion', 'rerank'):
        rows = [r for r in records if r['mode'] == mode]
        summaries[mode] = {metric: statistics.mean(r[metric] for r in rows)
                          for metric in ('recall_at_5', 'mrr', 'ndcg_at_5', 'top1_source_support')}
        summaries[mode]['median_latency_ms'] = statistics.median(r['latency_ms'] for r in rows)
    baseline = summaries['bm25']
    eligible = [mode for mode, score in summaries.items() if mode != 'bm25'
                and score['mrr'] > baseline['mrr'] and score['ndcg_at_5'] > baseline['ndcg_at_5']
                and score['recall_at_5'] >= baseline['recall_at_5'] and score['median_latency_ms'] < 50]
    selected = max(eligible, key=lambda m: summaries[m]['mrr']) if eligible else 'bm25'
    result = {'metrics': summaries, 'selected': selected, 'corpus_sha256': corpus.fingerprint,
              'task_sha256': hashlib.sha256(task_path.read_bytes()).hexdigest(),
              'protocol': config['protocol'], 'retention': config['retention'],
              'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                  for name in ('src/evaluation/retrieval_run.py', 'src/retrieval/report_index.py',
                               'src/data/report_corpus.py')},
              'limits': 'Small development set, one issuer. Top-1 source support is relevance, not generated answer accuracy. MRR uses full ranking; reranker retains top 10 candidates.'}
    (args.output_dir / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2))
    corpus.close()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
