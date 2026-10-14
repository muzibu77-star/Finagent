"""Download annual reports and their corrections from cninfo with disclosure dates.

The announcement index of the exchange-designated disclosure site supplies the
title, file and publication time of every filing.  Nothing is inferred: a
fiscal year without a matching annual report is recorded as a gap, and every
selected or skipped announcement is kept in the receipt.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

QUERY = 'http://www.cninfo.com.cn/new/hisAnnouncement/query'
FILES = 'http://static.cninfo.com.cn/'
CATEGORIES = {'annual': 'category_ndbg_szsh', 'correction': 'category_bcgz_szsh'}
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/x-www-form-urlencoded'}
BEIJING = timezone(timedelta(hours=8))
# "2023年年度报告", the banks' "2023年度报告" and one issuer's "2024年年报报告";
# "2023年半年度报告" does not match.
ANNUAL = re.compile(r'(\d{4})\s*年年?(?:度|报)报告')
EXCLUDED = ('摘要', '英文', '取消', '提示性公告')
REVISED = ('更正', '修订', '更新', '补充')
log = logging.getLogger(__name__)


def classify(title: str, category: str, fiscal_years: list[int],
             company: str = '') -> tuple[str, int | None] | None:
    """Kind and fiscal year of an annual report or its correction, else ``None``.

    A correction notice that names no year ("关于定期报告…的更正公告") is kept with
    an unknown fiscal year rather than assigned one.
    """
    title = re.sub(r'<[^>]+>', '', title)
    if any(word in title for word in EXCLUDED):
        return None
    # "中国平安：平安银行股份有限公司2023年年度报告" forwards a subsidiary's report;
    # "中国移动：2023年年度报告" is the issuer's own.
    if company and re.match(rf'{re.escape(company)}\s*[:：]\s*\D{{2,}}\d{{4}}', title):
        return None
    year = ANNUAL.search(title)
    revised = any(word in title for word in REVISED)
    if not year:
        if category == 'correction' and revised and re.search(r'定期报告|年度报告|年报', title):
            return 'correction', None
        return None
    if int(year.group(1)) not in fiscal_years:
        return None
    return ('correction' if category == 'correction' or revised else 'annual'), int(year.group(1))


def fetch(request: Request, timeout: int, attempts: int = 4) -> bytes:
    """Read a response; the site refuses connections briefly when polled too fast."""
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except OSError as error:
            if attempt == attempts:
                raise
            log.warning('retry %d after %r', attempt, error)
            time.sleep(30 * attempt)
    raise AssertionError('unreachable')


def announcements(issuer: dict, category: str, published: list[str], pause: float) -> list[dict]:
    """Every announcement of one category in the publication window."""
    shanghai = issuer['code'].startswith(('6', '9'))
    rows, page = [], 1
    while True:
        body = urlencode({
            'stock': f"{issuer['code']},{issuer['org_id']}", 'tabName': 'fulltext',
            'pageSize': 30, 'pageNum': page, 'column': 'sse' if shanghai else 'szse',
            'plate': 'sh' if shanghai else 'sz', 'category': CATEGORIES[category],
            'seDate': f'{published[0]}~{published[1]}', 'searchkey': '', 'secid': '',
            'sortName': '', 'sortType': '', 'isHLtitle': 'false'}).encode()
        payload = json.loads(fetch(Request(QUERY, data=body, headers=HEADERS), 45))
        batch = payload.get('announcements') or []
        rows.extend(batch)
        time.sleep(pause)
        if not payload.get('hasMore') or not batch:
            return rows
        page += 1


def download(url: str, path: Path, pause: float) -> str:
    """Fetch once; an existing file is hashed, never replaced."""
    if not path.exists():
        data = fetch(Request(url, headers={'User-Agent': HEADERS['User-Agent']}), 180)
        if not data.startswith(b'%PDF'):
            raise ValueError(f'not a PDF: {url}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        time.sleep(pause)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(config: dict, root: Path, pause: float) -> dict:
    sources, skipped, failures = [], [], []
    for position, issuer in enumerate(config['issuers'], 1):
        seen = set()
        for category in CATEGORIES:
            try:
                rows = announcements(issuer, category, config['published'], pause)
            except (OSError, ValueError) as error:
                failures.append({'code': issuer['code'], 'category': category, 'error': repr(error)})
                continue
            for row in rows:
                if row['announcementId'] in seen:
                    continue
                seen.add(row['announcementId'])
                title = re.sub(r'<[^>]+>', '', row['announcementTitle'])
                kind = classify(title, category, config['fiscal_years'], issuer['name'])
                if kind is None or not row['adjunctUrl'].lower().endswith('.pdf'):
                    skipped.append({'code': issuer['code'], 'title': title,
                                    'announcement_id': row['announcementId']})
                    continue
                url = FILES + row['adjunctUrl']
                path = root / issuer['code'] / f"{row['announcementId']}.pdf"
                try:
                    digest = download(url, path, pause)
                except (OSError, ValueError) as error:
                    failures.append({'code': issuer['code'], 'title': title, 'url': url,
                                     'error': repr(error)})
                    continue
                if any(s['sha256'] == digest for s in sources):
                    skipped.append({'code': issuer['code'], 'title': title,
                                    'announcement_id': row['announcementId'],
                                    'reason': 'same file as an earlier announcement'})
                    continue
                published = datetime.fromtimestamp(row['announcementTime'] / 1000, BEIJING)
                sources.append({
                    'id': f"{issuer['code']}-{row['announcementId']}", 'code': issuer['code'],
                    'company': issuer['name'], 'industry': issuer['industry'],
                    'kind': kind[0], 'fiscal_year': kind[1], 'title': title, 'url': url,
                    'disclosure_date': published.date().isoformat(), 'path': str(path),
                    'bytes': path.stat().st_size, 'sha256': digest})
        found = {s['fiscal_year'] for s in sources
                 if s['code'] == issuer['code'] and s['kind'] == 'annual'}
        log.info('%d/%d %s %s: annual reports for %s', position, len(config['issuers']),
                 issuer['code'], issuer['name'], sorted(found))
    gaps = [{'code': issuer['code'], 'company': issuer['name'], 'fiscal_year': year}
            for issuer in config['issuers'] for year in config['fiscal_years']
            if not any(s['code'] == issuer['code'] and s['fiscal_year'] == year
                       and s['kind'] == 'annual' for s in sources)]
    return {'snapshot_id': config['snapshot_id'], 'sources': sources, 'gaps': gaps,
            'failures': failures, 'skipped': skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/cninfo_issuers.json'))
    parser.add_argument('--root', type=Path, default=Path('data/raw/cninfo'))
    parser.add_argument('--receipt', type=Path, default=Path('configs/cninfo_sources.json'))
    parser.add_argument('--pause', type=float, default=1.0, help='seconds between requests')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    receipt = build(json.loads(args.config.read_text()), args.root, args.pause)
    receipt['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + '\n')
    log.info('sources=%d gaps=%d failures=%d skipped=%d', len(receipt['sources']),
             len(receipt['gaps']), len(receipt['failures']), len(receipt['skipped']))


if __name__ == '__main__':
    main()
