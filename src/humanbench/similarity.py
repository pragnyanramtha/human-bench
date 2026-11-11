"""Prompt-variety checks for finished task cards (DESIGN.md section 2.4, checks 3-4).

Deterministic: TF-IDF over word unigrams and bigrams. The flagged pairs are the
ones to rewrite before the tasks are frozen.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_WORD = re.compile(r"\w+(?:['’-]\w+)*")


def most_similar_pairs(
    items: Sequence[tuple[str, str]], top_fraction: float = 0.05
) -> list[tuple[str, str, float]]:
    """Return the most alike (id_a, id_b, cosine) pairs, most similar first.

    `top_fraction` of all pairs are returned (at least one if there are any pairs).
    """
    if len(items) < 2:
        return []
    ids = [i for i, _ in items]
    texts = [t for _, t in items]
    matrix = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    sims = cosine_similarity(matrix)
    pairs = [
        (ids[a], ids[b], float(sims[a, b])) for a in range(len(ids)) for b in range(a + 1, len(ids))
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    keep = max(1, round(len(pairs) * top_fraction))
    return pairs[:keep]


def shared_openings(items: Sequence[tuple[str, str]], n_words: int = 4) -> dict[str, list[str]]:
    """Group task ids whose prompts start with the same first `n_words` words."""
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id, text in items:
        words = _WORD.findall(text.lower())[:n_words]
        if len(words) == n_words:
            groups[" ".join(words)].append(task_id)
    return {opening: ids for opening, ids in groups.items() if len(ids) > 1}
