"""SQLite FTS5 BM25 over visible evidence only."""

import json
import re
import sqlite3

from src.data.leakage import content_text


class EvidenceIndex:
    def __init__(self, path: str = ':memory:') -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute('CREATE VIRTUAL TABLE IF NOT EXISTS evidence USING fts5(id UNINDEXED, text, payload UNINDEXED)')

    def add(self, evidence: dict) -> None:
        if set(evidence) - {'evidence_id', 'table', 'pre_text', 'post_text', 'paragraphs'}:
            raise ValueError('non-evidence fields cannot enter retrieval')
        self.connection.execute('INSERT INTO evidence VALUES (?, ?, ?)',
            (evidence['evidence_id'], content_text(evidence), json.dumps(evidence)))
        self.connection.commit()

    def search(self, query: str, limit: int = 5) -> list[str]:
        terms = re.findall(r'\w+', query.lower())[:64]
        if not terms:
            return []
        expression = ' OR '.join('"' + term + '"' for term in terms)
        return [row[0] for row in self.connection.execute(
            'SELECT id FROM evidence WHERE evidence MATCH ? ORDER BY bm25(evidence), id LIMIT ?',
            (expression, limit))]

    def close(self) -> None:
        self.connection.close()
