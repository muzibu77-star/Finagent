"""The source library: documents, page text and table facts in one SQLite file.

``build`` writes annual reports (page text exactly as PDFium returns it, so fact
spans index into it) and policy or statistics releases (one page per article).
``Library`` answers the two questions verification needs: which pages mention
something, and where a stated number occurs.  Every query can be limited to an
issuer and to documents published on or before a cutoff date.
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
import sqlite3
import time

import pypdfium2 as pdfium

from src.data.table_facts import extract
from src.data.table_map import key

log = logging.getLogger(__name__)
SCHEMA = '''
CREATE TABLE documents (id TEXT PRIMARY KEY, kind TEXT NOT NULL, publisher TEXT NOT NULL,
    code TEXT, fiscal_year INTEGER, title TEXT NOT NULL, url TEXT NOT NULL,
    published TEXT NOT NULL, sha256 TEXT NOT NULL, pages INTEGER NOT NULL);
CREATE TABLE pages (document_id TEXT NOT NULL, page INTEGER NOT NULL, text TEXT NOT NULL,
    PRIMARY KEY (document_id, page));
CREATE VIRTUAL TABLE page_terms USING fts5(terms, document_id UNINDEXED, page UNINDEXED);
CREATE TABLE facts (id INTEGER PRIMARY KEY, document_id TEXT NOT NULL, page INTEGER NOT NULL,
    start INTEGER NOT NULL, end INTEGER NOT NULL, box TEXT NOT NULL, text TEXT NOT NULL,
    digits TEXT NOT NULL, row_label TEXT, section TEXT, column_header TEXT, title TEXT,
    statement_date TEXT, unit TEXT, scale TEXT, currency TEXT, unit_source TEXT, scope TEXT);
CREATE INDEX facts_digits ON facts (digits);
CREATE INDEX facts_page ON facts (document_id, page);
CREATE TABLE problems (document_id TEXT NOT NULL, page INTEGER NOT NULL, problem TEXT NOT NULL,
    detail TEXT NOT NULL);
CREATE TABLE meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
'''
FACT_FIELDS = ('row_label', 'section', 'column_header', 'title', 'statement_date', 'unit',
               'scale', 'currency', 'unit_source', 'scope')


def terms(text: str) -> list[str]:
    pieces = re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]+', text.lower())
    return [token for p in pieces for token in
            ([p] if not '\u4e00' <= p[0] <= '\u9fff' or len(p) == 1
             else [p[i:i + 2] for i in range(len(p) - 1)])]


def digits(text: str) -> str:
    """A written number without sign marks or grouping, for exact lookup."""
    return key(text).strip('()%-')


def add_page(db: sqlite3.Connection, document_id: str, page: int, text: str) -> None:
    db.execute('INSERT INTO pages VALUES (?, ?, ?)', (document_id, page, text))
    db.execute('INSERT INTO page_terms VALUES (?, ?, ?)',
               (' '.join(terms(text)), document_id, page))


def add_report(db: sqlite3.Connection, source: dict, parsed: Path) -> dict:
    """One annual report or correction: pages, facts and binding problems."""
    middle = json.loads((parsed / f"{source['id']}.json").read_text())
    with pdfium.PdfDocument(source['path']) as document:
        count = len(document)
        for index in range(count):
            textpage = document[index].get_textpage()
            add_page(db, source['id'], index + 1, textpage.get_text_bounded())
            textpage.close()
        result = extract(document, middle)
    db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               (source['id'], source['kind'], source['company'], source['code'],
                source['fiscal_year'], source['title'], source['url'],
                source['disclosure_date'], source['sha256'], count))
    db.executemany(
        'INSERT INTO facts (document_id, page, start, end, box, text, digits, '
        + ', '.join(FACT_FIELDS) + ') VALUES (' + ', '.join('?' * (7 + len(FACT_FIELDS))) + ')',
        [(source['id'], fact['pdf_page'], fact['start'], fact['end'], json.dumps(fact['box']),
          fact['text'], digits(fact['text']), *(fact[name] for name in FACT_FIELDS))
         for fact in result['facts']])
    db.executemany('INSERT INTO problems VALUES (?, ?, ?, ?)',
                   [(source['id'], problem['pdf_page'], problem['problem'],
                     json.dumps({k: v for k, v in problem.items()
                                 if k not in ('pdf_page', 'problem')}, ensure_ascii=False))
                    for problem in result['problems']])
    return {'pages': count, 'facts': len(result['facts']), 'problems': len(result['problems'])}


RESTATED_TITLE = '合并资产负债表'


def add_restatement(db: sqlite3.Connection, config: dict) -> int:
    """A historical correction: the original report page and the correction notice.

    Only the rows a reviewer transcribed are stored.  The original values are
    located in the original page text; the correction notice is a scanned page,
    so its values carry the transcription as their page text and an empty box.
    """
    code, year = config['company'], int(config['period'])
    rows = config['rows']
    sources = []
    for kind, part in (('annual', config['original']), ('correction', config['correction'])):
        path = Path(part['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != part['sha256']:
            raise ValueError(f'source changed since download: {path}')
        document_id = f"{code}-{config['snapshot_id']}-{kind}"
        db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                   (document_id, kind, config.get('name', code), code, year,
                    f'{year}年年度报告' if kind == 'annual' else f'{year}年年度报告更正公告',
                    part['url'], part['published_at'], part['sha256'], 1))
        sources.append((document_id, part))
    (original_id, original), (correction_id, correction) = sources
    with pdfium.PdfDocument(original['path']) as document:
        textpage = document[original['page'] - 1].get_textpage()
        page_text = textpage.get_text_bounded()
        textpage.close()
    add_page(db, original_id, original['page'], page_text)
    transcript = '\n'.join(' '.join(row) for row in rows)
    add_page(db, correction_id, correction['page'], transcript)
    count = 0
    for label, before, _, after in rows:
        start = page_text.find(before)
        if start < 0:
            raise ValueError(f'original value {before} not on page {original["page"]}')
        for document_id, page, text, offset, source in (
                (original_id, original['page'], before, start, 'line before table'),
                (correction_id, correction['page'], after, transcript.find(after),
                 'transcribed from scanned page')):
            db.execute('INSERT INTO facts (document_id, page, start, end, box, text, digits, '
                       + ', '.join(FACT_FIELDS) + ') VALUES (' + ', '.join('?' * (7 + len(FACT_FIELDS))) + ')',
                       (document_id, page, offset, offset + len(text), '[]', text, digits(text),
                        label, None, '期末余额', RESTATED_TITLE, None, '元', '1', '人民币',
                        source, '合并'))
            count += 1
    return count


def build(reports: Path, parsed: Path, policies: Path, output: Path,
          restatements: list[Path] = ()) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(output / 'library.sqlite')
    db.executescript(SCHEMA)
    totals = {'reports': 0, 'releases': 0, 'pages': 0, 'facts': 0, 'problems': 0}
    sources = json.loads(reports.read_text())['sources']
    started = time.time()
    for position, source in enumerate(sources, 1):
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError(f"source changed since download: {source['path']}")
        counts = add_report(db, source, parsed)
        db.commit()
        totals['reports'] += 1
        for name in ('pages', 'facts', 'problems'):
            totals[name] += counts[name]
        elapsed = time.time() - started
        log.info('%d/%d reports, %d facts, elapsed %.0f s, ETA %.0f s', position, len(sources),
                 totals['facts'], elapsed, elapsed / position * (len(sources) - position))
    for release in json.loads(policies.read_text())['documents']:
        path = Path(release['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != release['sha256']:
            raise ValueError(f"source changed since download: {path}")
        db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                   (release['id'], release['source'], release['publisher'], None, None,
                    release['title'], release['url'], release['published'], release['sha256'], 1))
        add_page(db, release['id'], 1, path.with_suffix('.txt').read_text())
        totals['releases'] += 1
        totals['pages'] += 1
    for path in restatements:
        totals['facts'] += add_restatement(db, json.loads(path.read_text()))
        totals['restatements'] = totals.get('restatements', 0) + 1
    receipt = {**totals, 'reports_sha256': hashlib.sha256(reports.read_bytes()).hexdigest(),
               'policies_sha256': hashlib.sha256(policies.read_bytes()).hexdigest()}
    db.executemany('INSERT INTO meta VALUES (?, ?)', [(k, str(v)) for k, v in receipt.items()])
    db.commit()
    db.execute("INSERT INTO page_terms(page_terms) VALUES ('optimize')")
    db.commit()
    db.close()
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    return receipt


class Library:
    """Read-only queries over a built library."""

    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row

    def _scope(self, code: str | None, as_of: str | None) -> tuple[str, list]:
        clauses, values = [], []
        if code:
            clauses.append('d.code = ?')
            values.append(code)
        if as_of:
            # Disclosure dates are day-granular and include the cutoff day.
            clauses.append('d.published <= ?')
            values.append(as_of)
        return ''.join(f' AND {clause}' for clause in clauses), values

    def search_pages(self, query: str, code: str | None = None, as_of: str | None = None,
                     limit: int = 5) -> list[dict]:
        """Pages ranked by BM25 over Chinese bigrams and Latin words."""
        wanted = list(dict.fromkeys(terms(query)))
        if not wanted:
            return []
        scope, values = self._scope(code, as_of)
        match = ' OR '.join('"' + term.replace('"', '') + '"' for term in wanted)
        rows = self.db.execute(
            'SELECT t.document_id, t.page, d.publisher, d.title, d.published, d.kind '
            'FROM page_terms t JOIN documents d ON d.id = t.document_id '
            f'WHERE page_terms MATCH ?{scope} ORDER BY bm25(page_terms) LIMIT ?',
            [match, *values, limit]).fetchall()
        return [dict(row) for row in rows]

    def find_number(self, text: str, code: str | None = None,
                    as_of: str | None = None) -> list[dict]:
        """Table facts whose written digits equal those of ``text``."""
        scope, values = self._scope(code, as_of)
        rows = self.db.execute(
            'SELECT f.*, d.publisher, d.fiscal_year, d.published, d.kind, d.title AS document '
            f'FROM facts f JOIN documents d ON d.id = f.document_id WHERE f.digits = ?{scope} '
            'ORDER BY d.published DESC, f.page', [digits(text), *values]).fetchall()
        return [dict(row) for row in rows]

    def page_text(self, document_id: str, page: int) -> str:
        row = self.db.execute('SELECT text FROM pages WHERE document_id = ? AND page = ?',
                              (document_id, page)).fetchone()
        if row is None:
            raise KeyError((document_id, page))
        return row['text']

    def close(self) -> None:
        self.db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=Path('configs/cninfo_sources.json'))
    parser.add_argument('--parsed', type=Path, default=Path('data/staged/cninfo_parsed_v1'))
    parser.add_argument('--policies', type=Path, default=Path('configs/policy_sources.json'))
    parser.add_argument('--restatement', type=Path, action='append', default=[],
                        help='a reviewed correction case, e.g. configs/jd_restatement.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    print(json.dumps(build(args.reports, args.parsed, args.policies, args.output_dir,
                           args.restatement)))


if __name__ == '__main__':
    main()
