"""Freeze a small paired TAT-DQA test study without training/development sources."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import re
import zipfile

from PIL import Image
import pypdfium2 as pdfium


ARCHIVE_SHA = '83c3baf38102fe5b17041efe41ea770912e6afa32eb603fece5337318259ffc8'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = Path('data/raw/tatdqa')
    archive_path = raw / 'tatdqa_docs_test.zip'
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError('unverified visual archive')
    sources = json.loads(Path('configs/tatqa_sources.json').read_text())['tatdqa']['sha256']
    label_path = raw / 'tatdqa_dataset_test_gold.json'
    if hashlib.sha256(label_path.read_bytes()).hexdigest() != sources[label_path.name]:
        raise ValueError('label hash mismatch')
    annotations = json.loads(label_path.read_text())
    manifest_path = Path('data/staged/m0_frozen_v1/manifest.jsonl')
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    eligible = {r['source_report_id'] for r in manifest if r['dataset'] == 'tatqa'
                and r['official_eligible'] and r['original_split'] == 'test'}
    excluded = {r['source_report_id'] for r in manifest if r['group_split'] in {'train', 'dev'}}
    if eligible & excluded:
        raise ValueError('source split overlap')
    candidates, reasons, coverage = [], Counter(), Counter()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for context in annotations:
            uid = context['doc']['uid']
            ocr_name = f'test/{uid}.json'
            if ocr_name not in names or f'test/{uid}.pdf' not in names:
                coverage['missing_ocr'] += 1
                continue
            ocr = json.loads(archive.read(ocr_name))
            page_count = len(ocr['pages'])
            if any(f'test/{uid}_{i+1}.png' not in names for i in range(page_count)):
                coverage['missing_image'] += 1
                continue
            coverage['paired_contexts'] += 1
            source = 'tat:' + context['doc']['source']
            if source not in eligible:
                reasons['source_not_eligible_test'] += 1
                continue
            if page_count != 1:
                reasons['multi_page_budget_excluded'] += 1
                continue
            for q in context['questions']:
                scalar = q['answer_type'] == 'arithmetic' or (q['answer_type'] == 'span'
                         and len(q['answer']) == 1 and re.fullmatch(
                             r'[$€£]?\(?-?\d[\d,]*(?:\.\d+)?%?\)?', q['answer'][0].strip()))
                if not scalar:
                    continue
                candidates.append((context['doc'], q, ocr))
        random.Random(97).shuffle(candidates)
        selected, used, types = [], set(), Counter()
        for doc, question, ocr in candidates:
            kind = question['answer_type']
            if doc['source'] in used or types[kind] >= 6:
                continue
            selected.append((doc, question, ocr))
            used.add(doc['source'])
            types[kind] += 1
            if len(selected) == 12:
                break
        if types != {'span': 6, 'arithmetic': 6}:
            raise ValueError('insufficient source-isolated paired candidates')
        args.output_dir.mkdir(parents=True, exist_ok=False)
        visible, labels = [], {}
        for doc, q, ocr in selected:
            uid = doc['uid']
            page = ocr['pages'][0]
            thumbnail = Image.open(io.BytesIO(archive.read(f'test/{uid}_1.png')))
            pdf_bytes = archive.read(f'test/{uid}.pdf')
            with pdfium.PdfDocument(pdf_bytes) as document:
                if len(document) != 1:
                    raise ValueError('PDF and OCR page counts differ')
                pdf_page = document[0]
                width, height = pdf_page.get_size()
                if abs(width / height - page['bbox'][2] / page['bbox'][3]) > 0.002:
                    raise ValueError('PDF and OCR aspect ratios differ')
                bitmap = pdf_page.render(scale=page['bbox'][2] / width)
                image = bitmap.to_pil().copy()
                bitmap.close()
                pdf_page.close()
            image_path = args.output_dir / f'{uid}_1.png'
            image.save(image_path)
            image_bytes = image_path.read_bytes()
            blocks = [{'id': b['uuid'], 'text': b['text'], 'bbox': b['bbox']}
                      for b in sorted(page['blocks'], key=lambda b: b['order'])]
            for mapping in q['block_mapping']:
                for key, span in mapping.items():
                    block = next(b for b in blocks if b['id'] == key)
                    if not 0 <= span[0] < span[1] <= len(block['text']):
                        raise ValueError('invalid released block span')
            visible.append({'id': q['uid'], 'question': q['question'],
                'source_report_id': 'tat:' + doc['source'], 'doc_uid': uid,
                'dataset_relevant_start_page': doc['page'], 'original_report_page': None,
                'image_path': str(image_path), 'image_sha256': hashlib.sha256(image_bytes).hexdigest(),
                'width': image.width, 'height': image.height, 'pages': 1, 'blocks': blocks,
                'image_origin': 'PDFium render of archived original PDF; bundled PNG is only a thumbnail',
                'pdf_sha256': hashlib.sha256(pdf_bytes).hexdigest(),
                'bundled_thumbnail_size': list(thumbnail.size), 'ocr_coordinate_bbox': page['bbox']})
            labels[q['uid']] = q
        (args.output_dir / 'tasks.json').write_text(json.dumps(visible, indent=2))
        (args.output_dir / 'gold.json').write_text(json.dumps(labels, indent=2))
    lock = {'seed': 97, 'tasks': 12, 'types': dict(types), 'coverage': dict(coverage),
            'context_exclusions': dict(reasons), 'archive_sha256': ARCHIVE_SHA,
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'hashes': {name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
                       for name in ('tasks.json', 'gold.json')},
            'protocol': 'One-page, one-scalar, 12-source held-out diagnostic subset; no full benchmark claim. No visual tuning. Region labels validated but region localization is not evaluated.'}
    (args.output_dir / 'lock.json').write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2))


if __name__ == '__main__':
    main()
