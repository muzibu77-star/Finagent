"""Map released TAT-DQA page annotations to full reports without using answers."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import zipfile

from src.data.prepare_visual import ARCHIVE_SHA


def shingles(text: str) -> set[tuple]:
    tokens = re.findall(r'\w+', text.casefold())
    return {tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    documents = json.loads(args.corpus.read_text())
    by_report = defaultdict(list)
    page_shingles = {}
    for key, row in documents.items():
        by_report[row['source_report_id']].append(key)
        page_shingles[key] = shingles(row['evidence']['paragraphs'][0]['text'])
    archive_path = Path('data/raw/tatdqa/tatdqa_docs_test.zip')
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError('archive hash mismatch')
    label_path = Path('data/raw/tatdqa/tatdqa_dataset_test_gold.json')
    expected = json.loads(Path('configs/tatqa_sources.json').read_text())['tatdqa']['sha256'][label_path.name]
    if hashlib.sha256(label_path.read_bytes()).hexdigest() != expected:
        raise ValueError('annotation hash mismatch')
    tasks, gold, mappings, excluded = [], {}, [], []
    with zipfile.ZipFile(archive_path) as archive:
        for context in json.loads(label_path.read_text()):
            doc = context['doc']
            report = doc['source'].removesuffix('.pdf')
            if report not in by_report:
                continue
            ocr = json.loads(archive.read(f"test/{doc['uid']}.json"))
            block_pages = {}
            for page in ocr['pages']:
                visible = ' '.join(b['text'] for b in sorted(page['blocks'], key=lambda b: b['order']))
                original = shingles(visible)
                scores = sorted(((len(original & page_shingles[key]) / max(1, len(original)), key)
                                 for key in by_report[report]), reverse=True)
                best, key = scores[0]
                margin = best - (scores[1][0] if len(scores) > 1 else 0)
                mapping = {'doc_uid': doc['uid'], 'page_id': key,
                           'shingle_coverage': best, 'margin': margin}
                mappings.append(mapping)
                if best < 0.65 or margin < 0.15:
                    excluded.append({**mapping, 'reason': 'ambiguous or weak full-page alignment'})
                    continue
                for block in page['blocks']:
                    block_pages[block['uuid']] = key
            for question in context['questions']:
                blocks = {key for mapping in question['block_mapping'] for key in mapping}
                if not blocks or not blocks <= block_pages.keys():
                    excluded.append({'id': question['uid'], 'reason': 'unmapped relevant block'})
                    continue
                relevant = sorted({block_pages[key] for key in blocks})
                company = documents[relevant[0]]['company']
                split = 'dev' if int(hashlib.sha256(company.encode()).hexdigest()[:8], 16) % 3 == 0 else 'test'
                tasks.append({'id': question['uid'], 'question': question['question'],
                    'query': company.replace('-', ' ') + ' 2019 annual report: ' + question['question'],
                    'company': company, 'period': '2019', 'language': 'en', 'split': split,
                    'relevant': relevant, 'annotation': 'TAT-DQA block-to-PDF text alignment',
                    'source_report_id': report})
                gold[question['uid']] = question
    chinese = json.loads(Path('configs/m4_retrieval_tasks.json').read_text())['tasks']
    for row in chinese:
        page = documents[row['relevant'][0]]
        tasks.append({**row, 'query': row['question'], 'company': page['company'],
                      'period': page['report_period'], 'language': 'zh', 'split': 'dev',
                      'source_report_id': page['source_report_id'],
                      'annotation': 'Existing M4 development relevance; not independent test'})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in [('tasks', tasks), ('gold', gold), ('mappings', mappings), ('excluded', excluded)]:
        (args.output_dir / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2))
    lock = {'tasks': len(tasks), 'counts': dict(Counter(t['split'] for t in tasks)),
            'issuers': len({t['company'] for t in tasks}),
            'corpus_sha256': hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
            'sha256': {name: hashlib.sha256((args.output_dir / f'{name}.json').read_bytes()).hexdigest()
                       for name in ('tasks', 'gold', 'mappings', 'excluded')},
            'protocol': 'Full-corpus retrieval; issuer/year in query; page labels aligned by visible OCR only. '
                        'Historical test sources already exposed. No tuning on test partition.',
            'alignment': {'min_coverage': 0.65, 'min_margin': 0.15}}
    (args.output_dir / 'lock.json').write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock))


if __name__ == '__main__':
    main()
