"""Collect dated policy releases and official statistics for the source library.

Each source publishes its own list of titles, links and publication dates; the
list is taken as given and every article is saved unmodified next to its
extracted text.  An article whose body cannot be located is recorded as a
failure, not approximated.
"""

import argparse
from datetime import date
import hashlib
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
import time
from urllib.parse import urljoin
from urllib.request import Request, urlopen

AGENT = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) Chrome/120'}
VOID = {'br', 'img', 'input', 'meta', 'link', 'hr', 'source', 'wbr', 'area', 'col'}
log = logging.getLogger(__name__)


def fetch(url: str, attempts: int = 4) -> bytes:
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(Request(url, headers=AGENT), timeout=60) as response:
                return response.read()
        except OSError as error:
            if attempt == attempts:
                raise
            log.warning('retry %d for %s after %r', attempt, url, error)
            time.sleep(20 * attempt)
    raise AssertionError('unreachable')


class _Body(HTMLParser):
    """Text of the first element whose id or class contains a marker."""

    def __init__(self, markers: tuple[str, ...]) -> None:
        super().__init__()
        self.markers, self.depth, self.done = markers, 0, False
        self.skip, self.parts = 0, []

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        if self.depth:
            if tag not in VOID:
                self.depth += 1
            if tag in ('script', 'style'):
                self.skip += 1
            elif tag in ('p', 'tr', 'br', 'div', 'li'):
                self.parts.append('\n')
            elif tag in ('td', 'th'):
                self.parts.append('\t')
            return
        names = ' '.join(value or '' for key, value in attrs if key in ('id', 'class'))
        if any(marker in names.split() for marker in self.markers) and tag not in VOID:
            self.depth = 1

    def handle_endtag(self, tag):
        if self.depth and not self.done and tag not in VOID:
            if tag in ('script', 'style') and self.skip:
                self.skip -= 1
            self.depth -= 1
            self.done = self.depth == 0

    def handle_data(self, data):
        if self.depth and not self.done and not self.skip:
            self.parts.append(data)


def body_text(html: str, markers: tuple[str, ...]) -> str:
    """Article text with one line per paragraph or table row; empty if not found."""
    parser = _Body(markers)
    parser.feed(html)
    lines = (re.sub(r'[ 　\xa0]+', ' ', line).strip(' ')
             for line in ''.join(parser.parts).splitlines())
    return '\n'.join(line for line in lines if line.strip())


def executive_meetings(source: dict) -> list[dict]:
    """State Council executive meeting releases from the Xinhua activity feed."""
    rows = json.loads(fetch(source['list']))['datasource']
    return [{'title': re.sub(r'<[^>]+>', '', row['title']),
             'url': urljoin(source['list'], row['publishUrl']),
             'published': row['publishTime'][:10]}
            for row in rows if row.get('publishUrl') and '国务院常务会议' in row['title']]


def state_council_policies(source: dict) -> list[dict]:
    """Documents in the State Council's latest-policy feed."""
    return [{'title': row['TITLE'], 'url': row['URL'], 'published': row['DOCRELPUBTIME']}
            for row in json.loads(fetch(source['list']))
            if row['URL'].startswith('https://www.gov.cn/')]


def listed_pages(source: dict, start: str, pause: float) -> list[dict]:
    """Items of a paginated HTML list, newest first, back to ``start``.

    ``source['item']`` is a regular expression with ``link``, ``title`` and
    ``published`` groups; ``source['next']`` is the URL of later pages with a
    ``{page}`` placeholder counted from ``source['first_next']``.
    """
    item = re.compile(source['item'], re.S)
    found, url, page = [], source['list'], source['first_next']
    while True:
        try:
            html = fetch(url).decode('utf-8', 'replace')
        except OSError:
            # These sites answer 404 past their last page.
            return found
        rows = [match.groupdict() for match in item.finditer(html)]
        if not rows:
            return found
        for row in rows:
            row['published'] = re.sub(r'[/.]', '-', row['published'])
            found.append({'title': re.sub(r'<[^>]+>|\s+', ' ', row['title']).strip(),
                          'url': urljoin(url, row['link']), 'published': row['published']})
        if rows[-1]['published'] < start:
            return found
        url = urljoin(source['list'], source['next'].format(page=page))
        page += 1
        time.sleep(pause)


def csrc_channel(source: dict, start: str, pause: float) -> list[dict]:
    """Items of a securities regulator channel from its own paged list interface."""
    found, page = [], 1
    while True:
        results = json.loads(fetch(source['list'].format(page=page)))['data']['results']
        if not results:
            return found
        found.extend({'title': row['title'], 'url': urljoin(source['list'], row['url']),
                      'published': row['publishedTimeStr'][:10]} for row in results)
        if results[-1]['publishedTimeStr'][:10] < start:
            return found
        page += 1
        time.sleep(pause)


def build(config: dict, root: Path, pause: float) -> dict:
    documents, failures, listed = [], [], {}
    for source in config['sources']:
        # A source may state its own window, e.g. to reach an older known case.
        start, end = source.get('published', config['published'])
        if source['kind'] == 'executive_meetings':
            rows = executive_meetings(source)
        elif source['kind'] == 'state_council_policies':
            rows = state_council_policies(source)
        elif source['kind'] == 'listed_pages':
            rows = listed_pages(source, start, pause)
        elif source['kind'] == 'csrc_channel':
            rows = csrc_channel(source, start, pause)
        else:
            raise ValueError(f"unknown source kind {source['kind']}")
        rows = [row for row in rows if start <= row['published'] <= end]
        rows = list({row['url']: row for row in rows}.values())
        listed[source['id']] = len(rows)
        log.info('%s: %d items listed between %s and %s', source['id'], len(rows), start, end)
        for position, row in enumerate(rows, 1):
            name = hashlib.sha256(row['url'].encode()).hexdigest()[:16]
            path = root / source['id'] / f'{name}.html'
            try:
                if not path.exists():
                    data = fetch(row['url'])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    time.sleep(pause)
                raw = path.read_bytes()
                text = body_text(raw.decode('utf-8', 'replace'), tuple(source['body']))
                if len(text) < 50:
                    raise ValueError('article body not found')
            except (OSError, ValueError) as error:
                failures.append({'source': source['id'], **row, 'error': repr(error)})
                continue
            path.with_suffix('.txt').write_text(text)
            documents.append({
                'id': f"{source['id']}-{name}", 'source': source['id'],
                'publisher': source['publisher'], **row, 'path': str(path),
                'sha256': hashlib.sha256(raw).hexdigest(), 'characters': len(text)})
            if position % 50 == 0:
                log.info('%s: %d/%d fetched', source['id'], position, len(rows))
    return {'snapshot_id': config['snapshot_id'], 'published': config['published'],
            'listed': listed, 'documents': documents, 'failures': failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/policy_feeds.json'))
    parser.add_argument('--root', type=Path, default=Path('data/raw/policy'))
    parser.add_argument('--receipt', type=Path, default=Path('configs/policy_sources.json'))
    parser.add_argument('--pause', type=float, default=1.5, help='seconds between requests')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = json.loads(args.config.read_text())
    if not date.fromisoformat(config['published'][0]) <= date.fromisoformat(config['published'][1]):
        raise SystemExit('publication window is empty')
    receipt = build(config, args.root, args.pause)
    receipt['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + '\n')
    log.info('documents=%d failures=%d', len(receipt['documents']), len(receipt['failures']))


if __name__ == '__main__':
    main()
