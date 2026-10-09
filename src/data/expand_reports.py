"""Acquire linked annual reports and preserve every page and acquisition failure."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin
from urllib.request import urlopen

import pypdfium2 as pdfium


ISSUERS = (
    'plexus-corp', 'cisco-systems-inc', 'jabil-circuit-inc',
    'oracle-corporation', 'international-business-machines-corp',
    'microsoft-corporation', 'adobe-systems-inc', 'cts-corporation',
    'american-tower-corporation', 'bce-inc', 'netapp-inc',
    'micron-technology-inc', 'coherent-inc', 'microchip-technology-inc',
    'maxlinear-inc', 'teradyne-inc', 'advanced-energy',
    'jack-in-the-box-inc', 'conagra-brands-inc', 'rogers-communications-inc',
)


def acquire(issuer: str, root: Path) -> tuple[list, list]:
    """Discover links on the source directory; never infer a PDF URL."""
    directory = f'https://www.annualreports.com/Company/{issuer}'
    root.mkdir(parents=True, exist_ok=True)
    html_path = root / f'{issuer}.html'
    sources, failures = [], []
    try:
        if not html_path.exists():
            with urlopen(directory, timeout=45) as response:
                html_path.write_bytes(response.read())
        html = html_path.read_text()
        links = sorted(set(re.findall(r'href="([^"]+_201[89]\.pdf)"', html)))
        for year in ('2018', '2019'):
            matching = [urljoin(directory, link) for link in links
                        if link.endswith(f'_{year}.pdf')]
            if len(matching) != 1:
                failures.append({'issuer': issuer, 'year': year,
                                 'error': 'missing or ambiguous linked annual report'})
                continue
            path = root / f'{issuer}_{year}.pdf'
            try:
                if not path.exists():
                    with urlopen(matching[0], timeout=90) as response:
                        data = response.read()
                    if not data.startswith(b'%PDF'):
                        raise ValueError('response is not PDF')
                    path.write_bytes(data)
                sources.append({
                    'id': f'{issuer}_{year}', 'company': issuer,
                    'report_period': year, 'language': 'en', 'path': str(path),
                    'url': matching[0], 'directory_url': directory,
                    'directory_sha256': hashlib.sha256(html_path.read_bytes()).hexdigest(),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'published_at': None, 'version': 'original_archive',
                    'supersedes': None,
                    'ingested_at': datetime.now(timezone.utc).isoformat(),
                    'use': 'Local research; third-party report rights retained; no redistribution',
                })
            except (OSError, ValueError) as exc:
                failures.append({'issuer': issuer, 'year': year, 'error': str(exc)})
    except (OSError, ValueError) as exc:
        failures.append({'issuer': issuer, 'error': str(exc)})
    return sources, failures


def extract_pages(sources: list[dict]) -> tuple[dict, list]:
    """Keep physical PDF page numbers; unknown printed pages remain unknown."""
    documents, failures = {}, []
    for source in sources:
        raw = Path(source['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise ValueError('source hash mismatch')
        with pdfium.PdfDocument(raw) as pdf:
            for i in range(len(pdf)):
                page = pdf[i]
                text_page = page.get_textpage()
                text = text_page.get_text_bounded()
                text_page.close()
                page.close()
                key = f"{source['id']}:p{i + 1}"
                if not text.strip():
                    failures.append({'page_id': key, 'error': 'empty text; needs page/OCR review'})
                documents[key] = {
                    **source, 'source_report_id': source['id'],
                    'period': source['report_period'], 'pdf_page': i + 1,
                    'printed_page': None, 'parse_status': 'text' if text.strip() else 'empty',
                    'evidence': {'evidence_id': key, 'table': [],
                                 'paragraphs': [{'text': text}]},
                }
    return documents, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path('data/raw/expanded_reports')
    with ThreadPoolExecutor(max_workers=4) as pool:
        acquired = list(pool.map(lambda issuer: acquire(issuer, root), ISSUERS))
    sources = [s for rows, _ in acquired for s in rows]
    failures = [f for _, rows in acquired for f in rows]
    old = json.loads(Path('configs/report_sources.json').read_text())
    sources.extend({**s, 'company': old['company'], 'language': 'zh'}
                   for s in old['sources'])
    (args.output_dir / 'sources.json').write_text(json.dumps(sources, indent=2))
    documents, parse_failures = extract_pages(sources)
    path = args.output_dir / 'corpus.json'
    path.write_text(json.dumps(documents, ensure_ascii=False))
    receipt = {
        'snapshot_id': 'jd-reports-v1', 'reports': len(sources), 'pages': len(documents),
        'issuers': len({s['company'] for s in sources}),
        'years': sorted({s['report_period'] for s in sources}),
        'corpus_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'sources_sha256': hashlib.sha256((args.output_dir / 'sources.json').read_bytes()).hexdigest(),
        'parser': str(pdfium.PYPDFIUM_INFO),
        'acquisition_failures': failures, 'parse_failures': parse_failures,
    }
    (args.output_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
