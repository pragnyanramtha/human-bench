"""Row streams for the corpus pipeline.

A stream yields ``(cursor, row)`` in a fixed order. A cursor is
``(file_index, group_index, row_index)``; resuming from a cursor yields exactly
the rows a fresh run would have yielded from that point.

``HFParquetStream`` reads Parquet row groups over HTTP range requests at a
pinned revision, so nothing is downloaded in full. Order is a seeded
permutation of files, of row groups within a file, and of rows within a row
group; rows from a window of several row groups are interleaved, so even a
small sample is spread out (IMDb, for example, is stored sorted by label).
"""

from __future__ import annotations

import fnmatch
import random
import time
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

Cursor = tuple[int, int, int]
START: Cursor = (0, 0, 0)


def get_path(row: dict[str, Any], path: str) -> Any:
    """Look up a dotted path; integer segments index into lists."""
    cur: Any = row
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list | tuple) and part.lstrip("-").isdigit():
            i = int(part)
            cur = cur[i] if -len(cur) <= i < len(cur) else None
        else:
            return None
    return cur


def _perm(n: int, key: str) -> list[int]:
    order = list(range(n))
    random.Random(key).shuffle(order)
    return order


class RowStream(Protocol):
    revision: str

    def rows(self, start: Cursor = START) -> Iterator[tuple[Cursor, dict[str, Any]]]: ...


class InMemoryStream:
    """For tests: rows in a list, split into fixed-size groups."""

    def __init__(self, rows: Sequence[dict[str, Any]], group_size: int = 4, revision: str = "0" * 40):
        self._groups = [list(rows[i : i + group_size]) for i in range(0, len(rows), group_size)]
        self.revision = revision

    def rows(self, start: Cursor = START) -> Iterator[tuple[Cursor, dict[str, Any]]]:
        _, g0, r0 = start
        for g in range(g0, len(self._groups)):
            for r in range(r0 if g == g0 else 0, len(self._groups[g])):
                yield (0, g, r), {**self._groups[g][r], "__position__": f"g{g}r{r}"}


class HFParquetStream:
    def __init__(
        self,
        repo: str,
        revision: str,
        data_files: Sequence[str],
        exclude_files: Sequence[str] = (),
        columns: Sequence[str] | None = None,
        seed_key: str = "",
        retries: int = 4,
        window: int = 8,
    ) -> None:
        try:
            from huggingface_hub import HfApi, HfFileSystem
        except ImportError as e:  # pragma: no cover - depends on the optional extra
            raise RuntimeError("install the sourcing extra: uv sync --extra sourcing") from e
        self.repo = repo
        self.revision = revision
        self._fs = HfFileSystem()
        self._api = HfApi()
        self._columns = list(columns) if columns else None
        self._seed_key = seed_key or repo
        self._retries = retries
        self.window = window
        files = self._api.list_repo_files(repo, repo_type="dataset", revision=revision)
        chosen = sorted(
            f
            for f in files
            if f.endswith(".parquet")
            and any(fnmatch.fnmatch(f, g) for g in data_files)
            and not any(fnmatch.fnmatch(f, g) for g in exclude_files)
        )
        if not chosen:
            raise RuntimeError(f"{repo}@{revision}: no files match {list(data_files)}")
        self.files = [chosen[i] for i in _perm(len(chosen), f"{self._seed_key}:files")]

    def _retry(self, fn: Any, *args: Any) -> Any:
        for attempt in range(self._retries):
            try:
                return fn(*args)
            except Exception:  # network errors come from several libraries (httpx, fsspec, OS)
                if attempt == self._retries - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _open(self, path: str) -> Any:
        import pyarrow.parquet as pq

        return pq.ParquetFile(self._fs.open(f"datasets/{self.repo}@{self.revision}/{path}", "rb"))

    def _window(self, pf: Any, path: str, groups: list[int], cols: list[str] | None) -> list[dict[str, Any]]:
        """Read some row groups and interleave their (permuted) rows round-robin."""
        batches = []
        for rg in groups:
            table = self._retry(lambda rg=rg: pf.read_row_group(rg, columns=cols))
            rows = table.to_pylist()
            order = _perm(len(rows), f"{self._seed_key}:{path}:{rg}:rows")
            batches.append([(rg, i, rows[i]) for i in order])
        out = []
        for k in range(max((len(b) for b in batches), default=0)):
            for b in batches:
                if k < len(b):
                    rg, i, row = b[k]
                    row["__position__"] = f"{path}#rg{rg}#r{i}"
                    out.append(row)
        return out

    def rows(self, start: Cursor = START) -> Iterator[tuple[Cursor, dict[str, Any]]]:
        """Cursor is (file, window, row-in-window); a window is ``window`` row groups of one file."""
        f0, w0, r0 = start
        for fi in range(f0, len(self.files)):
            path = self.files[fi]
            pf = self._retry(self._open, path)
            groups = _perm(pf.metadata.num_row_groups, f"{self._seed_key}:{path}:groups")
            names = set(pf.schema_arrow.names)
            cols = [c for c in self._columns if c in names] if self._columns else None
            n_windows = -(-len(groups) // self.window)
            for wi in range(w0 if fi == f0 else 0, n_windows):
                rows = self._window(pf, path, groups[wi * self.window : (wi + 1) * self.window], cols)
                for ri in range((r0 if (fi == f0 and wi == w0) else 0), len(rows)):
                    yield (fi, wi, ri), rows[ri]


def needed_columns(paths: Sequence[str | None]) -> list[str]:
    return sorted({p.split(".")[0] for p in paths if p})
