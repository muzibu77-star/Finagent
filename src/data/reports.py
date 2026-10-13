"""Immutable PDF page ingestion with source hashes and disclosure metadata."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pypdfium2 as pdfium


def build(config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    documents = {}
    for source in config['sources']:
        path = Path(source['path'])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(source['url'], timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != source['sha256']:
                raise ValueError('download does not match frozen PDF')
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('source PDF changed')
        with pdfium.PdfDocument(path) as document:
            for i in range(len(document)):
                page = document[i]
                textpage = page.get_textpage()
                text = textpage.get_text_bounded()
                textpage.close()
                page.close()
                if not text.strip():
                    raise ValueError(f'page requires OCR, not silently skipped: {path}:{i+1}')
                key = f"{source['id']}:p{i+1}"
                documents[key] = {
                    **source, 'company': config['company'], 'period': source['report_period'],
                    'source_report_id': source['id'], 'pdf_page': i + 1,
                    'printed_page': source['first_printed_page'] + i,
                    'evidence': {'evidence_id': key, 'table': [], 'paragraphs': [{'text': text}]},
                }
    corpus_path = output / 'corpus.json'
    corpus_path.write_text(json.dumps(documents, ensure_ascii=False, indent=2))
    receipt = {'snapshot_id': config['snapshot_id'], 'documents': len(config['sources']),
               'pages': len(documents), 'parser': str(pdfium.PYPDFIUM_INFO),
               'corpus_sha256': hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
               'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/report_sources.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.output_dir)))


if __name__ == '__main__':
    main()
