"""Lane 1 pipeline: stream a source, gate every row, write provenance records.

Gate order (each rejection is counted under its reason):

    extract    source_filter, empty_text          (registry filters + cleaners)
    english    not_english
    date       date_unknown, date_after_cutoff
    normalize  empty_after_normalize               (humanbench.scoring.normalize)
    length     below_min_words, above_max_words    (band from axes.BAND_WORDS)
    dedup      exact_duplicate, near_duplicate
    pii        pii_dense, below_min_after_scrub
    quota      band_unused_for_channel, quota_full

Dedup compares PII-masked text. Masking is idempotent, so the index can be
rebuilt exactly from the scrubbed records on disk when a run resumes.

Deterministic: pinned revision, seeded read order, no clocks in the output.
Resumable: a state file stores the stream cursor and counters, and is written
only after the records it counts are flushed to disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from humanbench.axes import BAND_WORDS, Band
from humanbench.scoring.normalize import normalize, word_count
from humanbench.sourcing.cleaners import apply_cleaners
from humanbench.sourcing.dedup import Deduper
from humanbench.sourcing.paths import dump_json, write_atomic
from humanbench.sourcing.registry import Registry, RowFilter, Source
from humanbench.sourcing.streams import RowStream, get_path
from humanbench.sourcing.textgates import PII_TYPES, english_score, parse_date, scrub_pii

STATE_VERSION = 1
# Bump when gate or cleaner behaviour changes, so stale runs can't be resumed.
PIPELINE_VERSION = 3

REJECTION_REASONS = (
    "source_filter",
    "empty_text",
    "not_english",
    "date_unknown",
    "date_after_cutoff",
    "empty_after_normalize",
    "below_min_words",
    "above_max_words",
    "exact_duplicate",
    "near_duplicate",
    "pii_dense",
    "below_min_after_scrub",
    "band_unused_for_channel",
    "quota_full",
)

# Notes stored with every record and in the manifest.
USE_NOTE = "Archival public text: judge training and dev material only. Never a test reference."


@dataclass(frozen=True)
class GateConfig:
    min_words: int = BAND_WORDS[Band.SHORT][0]
    max_words: int = BAND_WORDS[Band.EXTENDED][1]
    min_function_share: float = 0.20
    min_latin_share: float = 0.90
    near_dup_threshold: float = 0.7
    num_perm: int = 128
    lsh_bands: int = 16
    shingle_size: int = 5
    max_pii_per_1k_words: float = 20.0
    seed: int = 0


def assign_band(n_words: int) -> Band | None:
    """Bands are half-open [lo, hi), except the top band, which includes 4,000."""
    for band in Band:
        lo, hi = BAND_WORDS[band]
        if lo <= n_words < hi:
            return band
    if n_words == BAND_WORDS[Band.EXTENDED][1]:
        return Band.EXTENDED
    return None


def passes_filter(f: RowFilter, row: dict[str, Any]) -> bool:
    v = get_path(row, f.field)
    match f.op:
        case "equals":
            return v == f.value
        case "in":
            return v in f.value
        case "not_in":
            return v not in f.value
        case "contains_any":
            return isinstance(v, str) and any(s in v for s in f.value)
        case "len_max":
            return v is not None and len(v) <= f.value
        case "len_min":
            return v is not None and len(v) >= f.value
    raise ValueError(f"unknown filter op {f.op}")


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, bool | int | float | str):
        return v
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


@dataclass
class Evaluation:
    reason: str | None
    record: dict[str, Any] | None = None
    dedup_payload: tuple[Any, str] | None = None
    band: str | None = None  # set for quota rejections


class SourceGates:
    """Applies the gates for one source; holds the dedup index and band counts."""

    def __init__(self, source: Source, revision: str, gates: GateConfig, quotas: dict[Band, int]):
        self.source = source
        self.revision = revision
        self.gates = gates
        self.quotas = quotas
        self.accepted_by_band: Counter[str] = Counter()
        self.deduper = Deduper(gates.near_dup_threshold, gates.num_perm, gates.shingle_size, gates.lsh_bands, gates.seed)

    def evaluate(self, row: dict[str, Any]) -> Evaluation:
        s, g = self.source, self.gates
        # extract
        for f in s.filters:
            if not passes_filter(f, row):
                return Evaluation("source_filter")
        raw = get_path(row, s.text_field)
        if not isinstance(raw, str) or not raw.strip():
            return Evaluation("empty_text")
        text = apply_cleaners(raw, s.cleaners)
        if not text.strip():
            return Evaluation("empty_text")
        # 1. English
        es = english_score(text)
        if not es.is_english(g.min_function_share, g.min_latin_share):
            return Evaluation("not_english")
        # 2. date
        date = parse_date(get_path(row, s.date_field), s.date_format) if s.date_field else None
        if date is None:
            if not s.pre_cutoff_by_construction:
                return Evaluation("date_unknown")
            date_basis = "by_construction"
        elif date >= s.cutoff:
            return Evaluation("date_after_cutoff")
        else:
            date_basis = "field"
        # 3. normalize
        text = normalize(text)
        if not text:
            return Evaluation("empty_after_normalize")
        # 4. length
        n = word_count(text)
        if n < g.min_words:
            return Evaluation("below_min_words")
        if n > g.max_words:
            return Evaluation("above_max_words")
        # 5. dedup (on PII-masked text; see module docstring)
        scrubbed, pii = scrub_pii(text)
        reason, _match, sig, ek = self.deduper.check(scrubbed)
        if reason:
            return Evaluation(reason)
        # 6. PII
        n_final = word_count(scrubbed)
        if sum(pii.values()) * 1000 / max(n_final, 1) > g.max_pii_per_1k_words:
            return Evaluation("pii_dense")
        if n_final < g.min_words:
            return Evaluation("below_min_after_scrub")
        band = assign_band(n_final)
        assert band is not None
        # 7. quota
        cap = self.quotas.get(band, 0)
        if cap == 0:
            return Evaluation("band_unused_for_channel", band=str(band))
        if self.accepted_by_band[band] >= cap:
            return Evaluation("quota_full", band=str(band))

        position = str(row.get("__position__", ""))
        oid = get_path(row, s.id_field) if s.id_field else None
        original_id = str(oid) if oid not in (None, "") else position
        record = {
            "id": f"{s.id}:{original_id}",
            "source": s.id,
            "original_id": original_id,
            "position": position,
            "repo": s.repo,
            "revision": self.revision,
            "date": date.isoformat() if date else None,
            "date_basis": date_basis,
            "licence": s.licence,
            "licence_status": str(s.licence_status),
            "allowed_uses": [str(u) for u in s.allowed_uses],
            "use_note": USE_NOTE,
            "channel": str(s.channel),
            "band": str(band),
            "word_count": n_final,
            "hash": "sha256:" + hashlib.sha256(scrubbed.encode("utf-8")).hexdigest(),
            "pii": pii,
            "english": {"function_word_share": round(es.function_word_share, 4), "latin_share": round(es.latin_share, 4)},
            "meta": {k: _jsonable(get_path(row, p)) for k, p in sorted(s.meta_fields.items())},
            "text": scrubbed,
        }
        return Evaluation(None, record, (sig, ek))

    def commit(self, ev: Evaluation) -> None:
        assert ev.record is not None and ev.dedup_payload is not None
        sig, ek = ev.dedup_payload
        self.deduper.add(ev.record["id"], sig, ek)
        self.accepted_by_band[ev.record["band"]] += 1

    def restore(self, record: dict[str, Any]) -> None:
        """Re-add an accepted record read back from disk (resume)."""
        _reason, _m, sig, ek = self.deduper.check(record["text"])
        self.deduper.add(record["id"], sig, ek)
        self.accepted_by_band[record["band"]] += 1


# --- run state ------------------------------------------------------------------


class ResumeError(RuntimeError):
    pass


@dataclass
class RunState:
    source: str
    repo: str
    revision: str
    channel: str
    config_hash: str
    quotas: dict[str, int]
    cursor: list[int] = field(default_factory=lambda: [0, 0, 0])
    rows_scanned: int = 0
    records_written: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    rejected_examples: dict[str, list[str]] = field(default_factory=dict)
    accepted_by_band: dict[str, int] = field(default_factory=dict)
    quota_full_by_band: dict[str, int] = field(default_factory=dict)
    pii_replacements: dict[str, int] = field(default_factory=lambda: {k: 0 for k in PII_TYPES})
    exhausted: bool = False
    stopped_because: str | None = None
    version: int = STATE_VERSION

    def bump(self, table: dict[str, int], key: str, n: int = 1) -> None:
        table[key] = table.get(key, 0) + n


def config_hash(source: Source, revision: str, gates: GateConfig, quotas: dict[Band, int]) -> str:
    payload = {
        "source": source.model_dump(mode="json", exclude={"verified", "notes"}),
        "revision": revision,
        "gates": asdict(gates),
        "quotas": {str(k): v for k, v in sorted(quotas.items())},
        "state_version": STATE_VERSION,
        "pipeline_version": PIPELINE_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _state_path(out_dir: Path, source_id: str) -> Path:
    return out_dir / ".state" / f"{source_id}.json"


def _records_path(out_dir: Path, source_id: str) -> Path:
    return out_dir / f"{source_id}.jsonl"


def _read_records(path: Path, keep: int) -> list[dict[str, Any]]:
    """Read the first ``keep`` records and truncate anything written after the last checkpoint."""
    if not path.exists():
        if keep:
            raise ResumeError(f"{path} is missing but the state says {keep} records were written")
        return []
    with path.open("rb") as f:
        lines = f.read().split(b"\n")
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) < keep:
        raise ResumeError(f"{path} has {len(lines)} records, state expects {keep}; rerun with --restart")
    if len(lines) > keep:
        write_atomic(path, "".join(ln.decode("utf-8") + "\n" for ln in lines[:keep]))
    return [json.loads(ln) for ln in lines[:keep]]


def run_source(
    source: Source,
    stream: RowStream,
    out_dir: Path,
    *,
    gates: GateConfig | None = None,
    quotas: dict[Band, int],
    limit: int | None = None,
    max_scan: int | None = None,
    restart: bool = False,
    checkpoint_every: int = 500,
    log: Callable[[str], None] | None = None,
) -> RunState:
    """Fill ``out_dir/<source>.jsonl``. ``limit`` and ``max_scan`` are totals across resumes."""
    log = log or (lambda msg: print(msg, file=sys.stderr))
    gates = gates or GateConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path, records_path = _state_path(out_dir, source.id), _records_path(out_dir, source.id)
    chash = config_hash(source, stream.revision, gates, quotas)
    if restart:
        for p in (state_path, records_path):
            p.unlink(missing_ok=True)

    pipe = SourceGates(source, stream.revision, gates, quotas)
    if state_path.exists():
        state = RunState(**json.loads(state_path.read_text(encoding="utf-8")))
        if state.config_hash != chash:
            raise ResumeError(
                f"{source.id}: registry entry, gates, quotas or revision changed since the last run; "
                "rerun with --restart to rebuild this source from scratch"
            )
        for rec in _read_records(records_path, state.records_written):
            pipe.restore(rec)
        log(f"[{source.id}] resuming at row {state.rows_scanned}, {state.records_written} records")
    else:
        if records_path.exists():
            raise ResumeError(f"{records_path} exists without run state; rerun with --restart")
        state = RunState(
            source=source.id,
            repo=source.repo,
            revision=stream.revision,
            channel=str(source.channel),
            config_hash=chash,
            quotas={str(k): v for k, v in quotas.items()},
        )

    def all_quotas_full() -> bool:
        return all(pipe.accepted_by_band[str(b)] >= cap for b, cap in quotas.items())

    out = records_path.open("a", encoding="utf-8")

    def checkpoint() -> None:
        out.flush()
        os.fsync(out.fileno())
        state.accepted_by_band = {k: v for k, v in sorted(pipe.accepted_by_band.items())}
        write_atomic(state_path, dump_json(asdict(state)))

    rows = stream.rows(tuple(state.cursor))  # type: ignore[arg-type]
    state.stopped_because = None
    try:
        while True:
            if limit is not None and state.records_written >= limit:
                state.stopped_because = "limit"
                break
            if max_scan is not None and state.rows_scanned >= max_scan:
                state.stopped_because = "max_scan"
                break
            if all_quotas_full():
                state.stopped_because = "quotas_full"
                break
            try:
                cursor, row = next(rows)
            except StopIteration:
                state.exhausted = True
                state.stopped_because = "exhausted"
                break
            state.rows_scanned += 1
            state.cursor = [cursor[0], cursor[1], cursor[2] + 1]
            ev = pipe.evaluate(row)
            if ev.reason:
                state.bump(state.rejected, ev.reason)
                examples = state.rejected_examples.setdefault(ev.reason, [])
                if len(examples) < 5:
                    examples.append(str(row.get("__position__", "")))
                if ev.band is not None:
                    state.bump(state.quota_full_by_band, ev.band)
            else:
                assert ev.record is not None
                out.write(json.dumps(ev.record, sort_keys=True, ensure_ascii=False) + "\n")
                pipe.commit(ev)
                state.records_written += 1
                for k, v in ev.record["pii"].items():
                    state.bump(state.pii_replacements, k, v)
            if state.rows_scanned % checkpoint_every == 0:
                checkpoint()
                log(f"[{source.id}] scanned {state.rows_scanned}, kept {state.records_written}")
    finally:
        checkpoint()
        out.close()
    log(f"[{source.id}] done: scanned {state.rows_scanned}, kept {state.records_written} ({state.stopped_because})")
    return state


# --- manifest --------------------------------------------------------------------


def build_manifest(out_dir: Path, registry: Registry) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    by_cell: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    accepted = scanned = 0
    for path in sorted((out_dir / ".state").glob("*.json")):
        st = json.loads(path.read_text(encoding="utf-8"))
        try:
            src = registry.get(st["source"])
        except KeyError:
            continue
        sources[src.id] = {
            "file": f"{src.id}.jsonl",
            "repo": src.repo,
            "revision": st["revision"],
            "channel": str(src.channel),
            "licence": src.licence,
            "licence_status": str(src.licence_status),
            "allowed_uses": [str(u) for u in src.allowed_uses],
            "cutoff": src.cutoff.isoformat(),
            "rows_scanned": st["rows_scanned"],
            "accepted": st["records_written"],
            "rejected": st["rejected"],
            "rejected_examples": st["rejected_examples"],
            "accepted_by_band": st["accepted_by_band"],
            "quota_full_by_band": st["quota_full_by_band"],
            "quotas": st["quotas"],
            "pii_replacements": st["pii_replacements"],
            "exhausted": st["exhausted"],
            "stopped_because": st["stopped_because"],
            "cursor": st["cursor"],
            "config_hash": st["config_hash"],
        }
        accepted += st["records_written"]
        scanned += st["rows_scanned"]
        rejected.update(st["rejected"])
        for band, n in st["accepted_by_band"].items():
            by_cell[f"{src.channel}/{band}"] += n
    return {
        "use_note": USE_NOTE,
        "gate_order": list(REJECTION_REASONS),
        "channels_without_sources": {str(k): v for k, v in registry.no_source.items()},
        "sources": sources,
        "totals": {
            "accepted": accepted,
            "rows_scanned": scanned,
            "by_channel_band": dict(sorted(by_cell.items())),
            "rejected": dict(rejected),
        },
    }


def write_manifest(out_dir: Path, registry: Registry) -> Path:
    path = out_dir / "manifest.json"
    write_atomic(path, dump_json(build_manifest(out_dir, registry)))
    return path
