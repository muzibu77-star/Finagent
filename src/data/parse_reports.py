"""Parse every downloaded report with MinerU, one resumable document at a time.

Layout parsing is the expensive step and is independent of how facts are bound
afterwards, so its output is kept per document.  A finished document is never
parsed again; a failed one is recorded and the run continues.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

import pypdfium2 as pdfium

log = logging.getLogger(__name__)
REQUIRED_ENV = ('MINERU_HOME', 'MINERU_MODEL_VLM_SERVER_URL', 'MINERU_MODEL_VLM_MODEL')


def finished(path: Path, pages: int) -> bool:
    """A complete parse of the whole document already exists."""
    if not path.exists():
        return False
    middle = json.loads(path.read_text())
    return middle.get('schema') == 'docvortex.middle' and len(middle['pages']) == pages


def parse(source: dict, output: Path, mineru: Path, timeout: int) -> None:
    """Parse to a scratch file and publish the result with one rename."""
    # MinerU writes a single input to the given path when it has a suffix.
    scratch = output.with_name(output.stem + '.partial.json')
    scratch.unlink(missing_ok=True)
    done = subprocess.run(
        [str(mineru), 'parse', source['path'], '-o', str(scratch), '-f', 'middle_json',
         '--tier', 'standard'], capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0 or not scratch.is_file():
        raise RuntimeError((done.stderr or done.stdout)[-2000:])
    scratch.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=Path('configs/cninfo_sources.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mineru', type=Path, default=Path('.venv-mineru/bin/mineru-kit'))
    parser.add_argument('--timeout', type=int, default=7200, help='seconds per document')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        # Without the served model MinerU silently falls back to a CPU engine
        # that is roughly fifty times slower.
        raise SystemExit(f'set {", ".join(missing)} before parsing')
    with urlopen(os.environ['MINERU_MODEL_VLM_SERVER_URL'].rstrip('/') + '/v1/models',
                 timeout=10) as response:
        if os.environ['MINERU_MODEL_VLM_MODEL'] not in response.read().decode():
            raise SystemExit('the document model server does not serve the configured model')
    sources = json.loads(args.sources.read_text())['sources']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = {}
    for source in sources:
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
            raise SystemExit(f"source changed since download: {source['path']}")
        with pdfium.PdfDocument(source['path']) as document:
            pages[source['id']] = len(document)
    total = sum(pages.values())
    failures = args.output_dir / 'failures.jsonl'
    started, parsed_pages, parsed_seconds, covered = time.time(), 0, 0.0, 0
    for position, source in enumerate(sources, 1):
        output = args.output_dir / f"{source['id']}.json"
        count = pages[source['id']]
        if finished(output, count):
            covered += count
            continue
        begun = time.time()
        try:
            parse(source, output, args.mineru, args.timeout)
            if not finished(output, count):
                raise RuntimeError('parse result does not cover every page')
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            with failures.open('a') as handle:
                handle.write(json.dumps({'id': source['id'], 'path': source['path'],
                                         'error': str(error)[-2000:]}, ensure_ascii=False) + '\n')
            log.error('%d/%d %s failed: %s', position, len(sources), source['id'],
                      str(error)[-200:])
            continue
        parsed_pages += count
        parsed_seconds += time.time() - begun
        covered += count
        rate = parsed_pages / parsed_seconds
        log.info('%d/%d documents, %d/%d pages, elapsed %.0f min, %.2f pages/s, ETA %.0f min (%s %s)',
                 position, len(sources), covered, total, (time.time() - started) / 60, rate,
                 (total - covered) / rate / 60, source['company'], source['fiscal_year'])
    complete = sum(finished(args.output_dir / f"{s['id']}.json", pages[s['id']]) for s in sources)
    log.info('finished: %d/%d documents parsed; failures are in %s', complete, len(sources), failures)


if __name__ == '__main__':
    main()
