"""Hold generation fixed while changing retrieval; audit page-level citations only."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

from src.data.report_corpus import ReportCorpus
from src.model.agent import ModelDriver
from src.model.financial import TOOLS, decode_prediction, messages_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path('configs/m4_retrieval_tasks.json').read_text())
    adapter = json.loads(Path('configs/model_choice.json').read_text())['adapter']
    corpus = ReportCorpus('data/staged/reports_v2/corpus.json', 'reports-v1')
    settings = {'adapter': adapter, 'tasks': config, 'corpus_sha256': corpus.fingerprint,
                'max_input_tokens': 8192, 'max_output_tokens': 512,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / 'settings.json').write_text(json.dumps(settings, indent=2))
    records = []
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    with ModelDriver(adapter) as model:
        for task in config['tasks']:
            for mode in ('bm25', 'fusion', 'rerank'):
                key = corpus.index.search(task['question'], set(corpus.documents), mode)[0]
                messages, facts = messages_for(corpus.documents[key]['evidence'], task['question'], 'calculator')
                row = decode_prediction(model(messages, TOOLS, lambda: False, time.monotonic() + 180), facts, 'calculator')
                cited = list(row.get('calculation', {}).get('facts', {}).values())
                row.update(task_id=task['id'], mode=mode, retrieved=key,
                           valid_calculation=bool(cited) and 'error' not in row,
                           relevant_citation=bool(cited) and 'error' not in row
                           and all(f['evidence_id'] in task['relevant'] for f in cited))
                records.append(row)
                with (args.output_dir / 'records.jsonl').open('a') as output:
                    output.write(json.dumps(row) + '\n')
                logging.info('%d/36 mode=%s relevant_citation=%s', len(records), mode, row['relevant_citation'])
    result = {'completed': len(records) == 36, 'settings': settings,
              'modes': {mode: {'calls': len(rows),
                       'valid_calculations': sum(r['valid_calculation'] for r in rows),
                       'relevant_citations': sum(r['relevant_citation'] for r in rows),
                       'generation_seconds': sum(r.get('latency_s', 0) for r in rows)}
                       for mode in ('bm25', 'fusion', 'rerank')
                       if (rows := [r for r in records if r['mode'] == mode])},
              'limits': 'Citation audit checks exact numeric source spans and relevant pages only. No answer-completeness or financial-semantic correctness claim; multi-value questions exceed the fixed single-calculation contract.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2))
    corpus.close()


if __name__ == '__main__':
    main()
