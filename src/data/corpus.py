"""Read-only source scope enforced independently of model tool arguments."""

import hashlib
import json
from pathlib import Path

from src.data.leakage import content_text
from src.evidence.identity import evidence_identity
from src.retrieval.bm25 import EvidenceIndex


class Corpus:
    def __init__(self, path: str, snapshot_id: str):
        raw = Path(path).read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.snapshot_id = snapshot_id
        self.documents = json.loads(raw)
        self.index = EvidenceIndex()
        seen = set()
        for key, row in sorted(self.documents.items()):
            if key != row['evidence']['evidence_id']:
                raise ValueError('corpus key and evidence ID differ')
            identity = evidence_identity(row['evidence'], row['source_report_id'])
            if identity not in seen:
                self.index.add(row['evidence'])
                seen.add(identity)
        self.companies = {r['company'] for r in self.documents.values()}

    def allowed(self, payload: dict) -> set[str]:
        if payload['snapshot_id'] != self.snapshot_id:
            raise ValueError('unknown source snapshot')
        if payload.get('as_of'):
            raise ValueError('unverified disclosure timestamps; strict PIT unavailable')
        periods = payload['period'] if isinstance(payload['period'], list) else [payload['period']]
        return {key for key, row in self.documents.items()
                if row['company'] == payload['company'] and row['period'] in periods}

    def search(self, query: str, allowed: set[str]) -> list[dict]:
        hits = [key for key in self.index.search(query, limit=len(self.documents))
                if key in allowed][:5]
        return [{'evidence_id': key, 'source_report_id': self.documents[key]['source_report_id'],
                 'preview': content_text(self.documents[key]['evidence'])[:180]} for key in hits]

    def read(self, evidence_id: str, allowed: set[str]) -> dict:
        if evidence_id not in allowed:
            raise ValueError('evidence outside authorized task scope')
        return self.documents[evidence_id]

    def close(self) -> None:
        self.index.close()
