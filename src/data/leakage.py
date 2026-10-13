"""Content-only fingerprints and exact Jaccard near-duplicate auditing."""

from collections import defaultdict
import math
import re
from typing import Any


def content_text(evidence: dict[str, Any]) -> str:
    """Exclude arbitrary row IDs while preserving numeric and word tokens."""
    parts = [str(cell) for row in evidence['table'] for cell in row]
    parts.extend(evidence.get('pre_text', []))
    parts.extend(evidence.get('post_text', []))
    parts.extend(p['text'] for p in evidence.get('paragraphs', []))
    return ' '.join(' '.join(parts).casefold().split())


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    words = re.findall(r'\w+|[^\w\s]', text)
    if len(words) < 5:
        return frozenset([tuple(words)]) if words else frozenset()
    return frozenset(tuple(words[i:i + 5]) for i in range(len(words) - 4))


def near_pairs(
    texts: list[str], threshold: float = 0.9,
) -> list[tuple[int, int, float]]:
    """Find all pairs above Jaccard threshold with a global-order prefix join."""
    if not 0 < threshold <= 1:
        raise ValueError('threshold must be in (0, 1]')
    sets = [shingles(text) for text in texts]
    frequency = defaultdict(int)
    for values in sets:
        for value in values:
            frequency[value] += 1
    index = defaultdict(list)
    pairs = []
    for i, values in enumerate(sets):
        if not values:
            continue
        ordered = sorted(values, key=lambda x: (frequency[x], x))
        prefix = ordered[:len(values) - math.ceil(threshold * len(values)) + 1]
        candidates = set()
        for value in prefix:
            candidates.update(index[value])
        for j in sorted(candidates):
            other = sets[j]
            if min(len(values), len(other)) < threshold * max(len(values), len(other)):
                continue
            similarity = len(values & other) / len(values | other)
            if similarity >= threshold:
                pairs.append((j, i, similarity))
        for value in prefix:
            index[value].append(i)
    return pairs
