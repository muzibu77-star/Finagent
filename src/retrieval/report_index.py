"""CPU Chinese-aware BM25, sparse TF-IDF fusion and lexical reranking."""

from collections import Counter
import math
import re

from src.data.leakage import content_text
from src.retrieval.bm25 import EvidenceIndex


def terms(text: str) -> list[str]:
    pieces = re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]+', text.lower())
    return [token for p in pieces for token in
            ([p] if not '\u4e00' <= p[0] <= '\u9fff' or len(p) == 1
             else [p[i:i + 2] for i in range(len(p) - 1)])]


class ReportIndex:
    def __init__(self, documents: dict):
        self.documents = documents
        self.index = EvidenceIndex()
        self.texts = {key: content_text(row['evidence']) for key, row in documents.items()}
        self.counts = {key: Counter(terms(text)) for key, text in self.texts.items()}
        df = Counter(token for counts in self.counts.values() for token in counts)
        self.idf = {t: math.log((1 + len(documents)) / (1 + n)) + 1 for t, n in df.items()}
        self.vectors = {key: self.vector(counts) for key, counts in self.counts.items()}
        for key, text in self.texts.items():
            self.index.add({'evidence_id': key, 'table': [],
                            'paragraphs': [{'text': ' '.join(terms(text))}]})

    def vector(self, counts: Counter) -> dict:
        values = {t: (1 + math.log(n)) * self.idf[t] for t, n in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in values.values()))
        return {t: v / norm for t, v in values.items()} if norm else {}

    def search(self, query: str, allowed: set[str], mode: str = 'bm25', limit: int = 5) -> list[str]:
        if mode not in {'bm25', 'fusion', 'rerank'}:
            raise ValueError('unknown retrieval mode')
        if not allowed <= self.documents.keys():
            raise ValueError('scope contains unknown documents')
        if not allowed:
            return []
        if allowed != self.documents.keys():
            # Excluded future sources must not influence BM25/TF-IDF statistics either.
            scoped = ReportIndex({key: self.documents[key] for key in sorted(allowed)})
            try:
                return scoped.search(query, allowed, mode, limit)
            finally:
                scoped.close()
        tokens = terms(query)
        bm25 = [key for key in self.index.search(' '.join(tokens), len(self.texts)) if key in allowed]
        if mode == 'bm25':
            return bm25[:limit]
        q = self.vector(Counter(tokens))
        scores = {key: sum(q.get(t, 0) * v for t, v in self.vectors[key].items()) for key in allowed}
        sparse = sorted((key for key, v in scores.items() if v > 0), key=lambda k: (-scores[k], k))
        fused = Counter()
        for ranking in (bm25, sparse):
            for i, key in enumerate(ranking):
                fused[key] += 1 / (60 + i + 1)
        ranking = sorted(fused, key=lambda k: (-fused[k], k))
        if mode == 'rerank':
            # A transparent CPU lexical comparator, not a learned semantic reranker.
            phrases = re.findall(r'[\u4e00-\u9fff]{3,}|[a-z0-9]+', query.lower())
            candidates = ranking[:10]
            ranking = sorted(candidates, key=lambda k: (
                -sum(len(p) for p in phrases if p in self.texts[k]), -fused[k], k))
        return ranking[:limit]

    def close(self) -> None:
        self.index.close()
