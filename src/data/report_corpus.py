"""Frozen report snapshots and date-granular point-in-time source authorization."""

from datetime import date, datetime
import hashlib
import json
from pathlib import Path

from src.data.corpus import Corpus
from src.retrieval.report_index import ReportIndex


class ReportCorpus(Corpus):
    def __init__(self, path: str, snapshot_id: str, mode: str = 'bm25'):
        raw = Path(path).read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        receipt = json.loads(Path(path).with_name('receipt.json').read_text())
        if receipt['snapshot_id'] != snapshot_id or receipt['corpus_sha256'] != self.fingerprint:
            raise ValueError('snapshot identity or content hash mismatch')
        self.snapshot_id, self.mode = snapshot_id, mode
        self.documents = json.loads(raw)
        self.companies = {row['company'] for row in self.documents.values()}
        reports = {row['source_report_id']: row for row in self.documents.values()}
        for key, row in self.documents.items():
            if key != row['evidence']['evidence_id']:
                raise ValueError('evidence identity mismatch')
            for field in ('version', 'ingested_at', 'report_period'):
                if not row.get(field):
                    raise ValueError(f'missing document metadata: {field}')
            if datetime.fromisoformat(row['ingested_at']).tzinfo is None:
                raise ValueError('ingestion time must include timezone')
            reference = reports[row['source_report_id']]
            if any(row.get(k) != reference.get(k) for k in
                   ('company', 'report_period', 'version', 'published_at', 'ingested_at', 'supersedes')):
                raise ValueError('inconsistent pages of the same report')
            if row.get('published_at'):
                date.fromisoformat(row['published_at'])
            previous = row.get('supersedes')
            if previous:
                if previous not in reports or previous == row['source_report_id']:
                    raise ValueError('unknown or cyclic replacement')
                old = reports[previous]
                if any(row[k] != old[k] for k in ('company', 'report_period')):
                    raise ValueError('replacement changes report scope')
                if not row.get('published_at') or not old.get('published_at') or row['published_at'] <= old['published_at']:
                    raise ValueError('replacement needs a later verified disclosure date')
        self.index = ReportIndex(self.documents)

    def allowed(self, payload: dict) -> set[str]:
        if payload['snapshot_id'] != self.snapshot_id:
            raise ValueError('unknown source snapshot')
        periods = payload['period'] if isinstance(payload['period'], list) else [payload['period']]
        cutoff = payload.get('as_of')
        if cutoff:
            if date.fromisoformat(cutoff).isoformat() != cutoff:
                raise ValueError('as_of must be an ISO calendar date, inclusive end of day')
        eligible = {key: row for key, row in self.documents.items()
                    if row['company'] == payload['company'] and row['report_period'] in periods
                    and (not cutoff or (row.get('published_at') and row['published_at'] <= cutoff))}
        superseded = {row['supersedes'] for row in eligible.values() if row.get('supersedes')}
        return {key for key, row in eligible.items() if row['source_report_id'] not in superseded}

    def search(self, query: str, allowed: set[str]) -> list[dict]:
        return [{'evidence_id': key, 'source_report_id': self.documents[key]['source_report_id'],
                 'preview': self.index.texts[key][:180]}
                for key in self.index.search(query, allowed, self.mode)]
