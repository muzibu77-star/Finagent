"""Frozen BGE dense/RRF/cross-encoder retrieval with Qdrant local persistence."""

from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import time

import torch
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src.data.leakage import content_text
from src.retrieval.report_index import ReportIndex


def page_ranking(chunks: dict, ranking: list[str]) -> list[str]:
    return list(dict.fromkeys(chunks[key]['page_id'] for key in ranking))


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores = Counter()
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] += 1 / (k + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))


def split_page(text: str, tokenizer, size: int, overlap: int) -> list[tuple[int, int]]:
    """Return original character spans covering all tokenized page content."""
    if not 0 <= overlap < size:
        raise ValueError('overlap must be smaller than positive chunk size')
    offsets = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True)['offset_mapping']
    spans = []
    for start in range(0, len(offsets), size - overlap):
        end = min(start + size, len(offsets))
        spans.append((offsets[start][0], offsets[end - 1][1]))
        if end == len(offsets):
            break
    return spans


class NeuralIndex:
    """Index content is hash-bound; authorization is supplied for every query."""

    def __init__(self, documents: dict, root: Path, config: dict, device: str = 'cuda'):
        self.config, self.documents, self.root = config, documents, root
        self.device = device
        self.texts = {key: content_text(row['evidence']) for key, row in documents.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(config['embedding']['path'], local_files_only=True)
        signature = hashlib.sha256(json.dumps(
            {'documents': documents, 'config': config,
             'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            sort_keys=True).encode()).hexdigest()
        root.mkdir(parents=True, exist_ok=True)
        identity = root / 'identity.json'
        if identity.exists() and json.loads(identity.read_text())['sha256'] != signature:
            raise ValueError('vector index content/config mismatch; use a new index directory')
        self.chunks = {}
        for page_id, text in sorted(self.texts.items()):
            for i, (start, end) in enumerate(split_page(
                    text, self.tokenizer, config['chunk_tokens'], config['overlap_tokens'])):
                key = f'{page_id}:c{i}'
                self.chunks[key] = {'page_id': page_id, 'start': start, 'end': end,
                                    'text': text[start:end]}
        if not self.chunks:
            raise ValueError('no parsed content to index')
        identity.write_text(json.dumps({'sha256': signature, 'chunks': len(self.chunks)}))
        self.client = QdrantClient(path=str(root / 'qdrant'))
        self.keys = list(self.chunks)
        self.ids = {key: i for i, key in enumerate(self.keys)}
        if not self.client.collection_exists('chunks'):
            self.client.create_collection('chunks', vectors_config=models.VectorParams(
                size=config['embedding']['dimension'], distance=models.Distance.COSINE))
        self.encoder = AutoModel.from_pretrained(config['embedding']['path'],
                                                local_files_only=True).to(device).eval()
        existing = set()
        offset = None
        while True:
            points, offset = self.client.scroll('chunks', limit=256, offset=offset,
                                                with_payload=False, with_vectors=False)
            existing.update(p.id for p in points)
            if offset is None:
                break
        missing = [key for key in self.keys if self.ids[key] not in existing]
        started = time.monotonic()
        batch = config['batch_size']
        for pos in range(0, len(missing), batch):
            keys = missing[pos:pos + batch]
            vectors = self.encode([self.chunks[key]['text'] for key in keys])
            self.client.upsert('chunks', points=[models.PointStruct(
                id=self.ids[key], vector=vector, payload={'chunk_id': key,
                'page_id': self.chunks[key]['page_id']}) for key, vector in zip(keys, vectors)])
            if pos % (batch * 50) == 0:
                elapsed = time.monotonic() - started
                done = min(pos + batch, len(missing))
                logging.info('index %d/%d elapsed=%.1fs ETA=%.1fs checkpoint=%s',
                             done, len(missing), elapsed, elapsed / done * (len(missing) - done), root)
        self.lexical = ReportIndex({key: {'evidence': {'evidence_id': key,
            'table': [], 'paragraphs': [{'text': row['text']}]}} for key, row in self.chunks.items()})
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(config['reranker']['path'], local_files_only=True)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            config['reranker']['path'], local_files_only=True).to(device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=False, return_tensors='pt').to(self.device)
        if inputs['input_ids'].shape[1] > 8192:
            raise ValueError('embedding input exceeds model context')
        vectors = torch.nn.functional.normalize(self.encoder(**inputs).last_hidden_state[:, 0], p=2, dim=-1)
        if (vectors.shape != (len(texts), self.config['embedding']['dimension'])
                or not torch.isfinite(vectors).all()
                or (torch.linalg.vector_norm(vectors, dim=-1) == 0).any()):
            raise ValueError('invalid nonfinite, zero or incorrectly shaped embedding')
        return vectors.float().cpu().tolist()

    @torch.inference_mode()
    def compare(self, query: str, allowed: set[str]) -> dict:
        started = time.monotonic()
        self.last_timings = {}
        if not allowed <= self.documents.keys():
            raise ValueError('unknown authorized pages')
        if not allowed:
            return {mode: [] for mode in ('bm25', 'dense', 'hybrid', 'cross_encoder')}
        chunk_ids = {key for key, row in self.chunks.items() if row['page_id'] in allowed}
        limit = self.config['candidate_chunks']
        sparse = self.lexical.search(query, chunk_ids, 'bm25', limit)
        self.last_timings['bm25_s'] = time.monotonic() - started
        started = time.monotonic()
        vector = self.encode([query])[0]
        condition = models.Filter(must=[models.HasIdCondition(
            has_id=[self.ids[key] for key in sorted(chunk_ids)])])
        points = self.client.query_points('chunks', query=vector, query_filter=condition,
                                          limit=limit).points
        dense = [point.payload['chunk_id'] for point in points]
        self.last_timings['dense_s'] = time.monotonic() - started
        started = time.monotonic()
        hybrid = reciprocal_rank_fusion([sparse, dense], self.config['rrf_k'])
        self.last_timings['fusion_s'] = time.monotonic() - started
        started = time.monotonic()
        candidates = hybrid[:self.config['rerank_chunks']]
        scores = []
        for start in range(0, len(candidates), self.config['batch_size']):
            keys = candidates[start:start + self.config['batch_size']]
            pairs = [[query, self.chunks[key]['text']] for key in keys]
            encoded = self.rerank_tokenizer(pairs, padding=True, truncation=False,
                                             return_tensors='pt').to(self.device)
            if encoded['input_ids'].shape[1] > 8192:
                raise ValueError('reranker input exceeds model context')
            logits = self.reranker(**encoded).logits.reshape(-1)
            if logits.numel() != len(keys) or not torch.isfinite(logits).all():
                raise ValueError('invalid reranker scores')
            scores.extend(logits.float().cpu().tolist())
        reranked = sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0]))
        self.last_timings['rerank_s'] = time.monotonic() - started
        return {'bm25': sparse, 'dense': dense, 'hybrid': hybrid,
                'cross_encoder': [key for key, _ in reranked]}

    def search(self, query: str, allowed: set[str], mode: str = 'cross_encoder',
               limit: int = 5) -> list[str]:
        if mode not in {'bm25', 'dense', 'hybrid', 'cross_encoder'}:
            raise ValueError('unknown neural retrieval mode')
        return page_ranking(self.chunks, self.compare(query, allowed)[mode])[:limit]

    def close(self) -> None:
        self.client.close()
        self.lexical.close()
        del self.encoder, self.reranker
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
