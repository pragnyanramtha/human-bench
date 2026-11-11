"""Exact and near-duplicate detection: SHA-256 plus MinHash with LSH banding.

Small and dependency-light (numpy only). Signatures are deterministic: shingle
hashes come from BLAKE2b, never Python's salted ``hash()``.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)


def canonical_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def exact_key(text: str) -> str:
    return hashlib.sha256(" ".join(canonical_words(text)).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    words = canonical_words(text)
    if len(words) <= k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class MinHasher:
    """Multiply-shift hash family over 64-bit shingle hashes (wraps mod 2**64)."""

    def __init__(self, num_perm: int = 128, shingle_size: int = 5, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.a = rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        grams = shingles(text, self.shingle_size)
        if not grams:
            return np.full(self.num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)
        h = np.fromiter(
            (int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little") for g in sorted(grams)),
            dtype=np.uint64,
            count=len(grams),
        )
        with np.errstate(over="ignore"):
            mixed = (h[:, None] * self.a[None, :] + self.b[None, :]) >> np.uint64(32)
        return mixed.min(axis=0).astype(np.uint32)


class NearDupIndex:
    """LSH index: ``bands`` x ``rows`` = signature length.

    The defaults (16 x 8) put the 50% candidate point near Jaccard 0.7;
    candidates are then confirmed with the signature-estimated Jaccard.
    """

    def __init__(self, hasher: MinHasher, threshold: float = 0.7, bands: int = 16) -> None:
        if hasher.num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.hasher = hasher
        self.threshold = threshold
        self.bands = bands
        self.rows = hasher.num_perm // bands
        self._buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
        self._sigs: list[np.ndarray] = []
        self._keys: list[str] = []

    def __len__(self) -> int:
        return len(self._sigs)

    def _band_keys(self, sig: np.ndarray) -> list[tuple[int, bytes]]:
        return [(i, sig[i * self.rows : (i + 1) * self.rows].tobytes()) for i in range(self.bands)]

    def query(self, sig: np.ndarray) -> tuple[str, float] | None:
        """Best match at or above the threshold, as (key, estimated Jaccard)."""
        seen: set[int] = set()
        best: tuple[str, float] | None = None
        for bk in self._band_keys(sig):
            for idx in self._buckets.get(bk, ()):
                if idx in seen:
                    continue
                seen.add(idx)
                est = float(np.mean(self._sigs[idx] == sig))
                if est >= self.threshold and (best is None or est > best[1]):
                    best = (self._keys[idx], est)
        return best

    def add(self, key: str, sig: np.ndarray) -> None:
        idx = len(self._sigs)
        self._sigs.append(sig)
        self._keys.append(key)
        for bk in self._band_keys(sig):
            self._buckets[bk].append(idx)


class Deduper:
    """Exact + near-duplicate filter over accepted texts."""

    def __init__(self, threshold: float = 0.7, num_perm: int = 128, shingle_size: int = 5, bands: int = 16, seed: int = 0):
        self.exact: dict[str, str] = {}
        self.near = NearDupIndex(MinHasher(num_perm, shingle_size, seed), threshold, bands)

    def check(self, text: str) -> tuple[str | None, str | None, np.ndarray, str]:
        """Return (reason, matched_key, signature, exact_key); reason is None if new."""
        ek = exact_key(text)
        sig = self.near.hasher.signature(text)
        if ek in self.exact:
            return "exact_duplicate", self.exact[ek], sig, ek
        hit = self.near.query(sig)
        if hit is not None:
            return "near_duplicate", hit[0], sig, ek
        return None, None, sig, ek

    def add(self, key: str, sig: np.ndarray, ek: str) -> None:
        self.exact.setdefault(ek, key)
        self.near.add(key, sig)
