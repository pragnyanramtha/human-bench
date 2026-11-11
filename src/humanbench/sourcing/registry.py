"""Source registry for Lane 1 (public, pre-LLM human text).

The registry lives in ``sources.yaml`` next to this file. Each entry pins a
Hugging Face dataset revision, names the fields to read, and records the
licence and the uses it allows. Archival text is never a test reference, so
``test_ref`` is not a valid use and the schema rejects it.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from humanbench.axes import ALLOWED_BANDS, Band, Channel

DEFAULT_CUTOFF = dt.date(2022, 11, 30)  # ChatGPT release; text must be dated strictly before this


class LicenceStatus(StrEnum):
    CLEAR = "clear"  # licence on the card is explicit and permits redistribution
    QUESTIONABLE = "questionable"  # unclear, inherited from scraped content, or disputed
    RESTRICTIVE = "restrictive"  # explicit terms that limit use (e.g. non-commercial, no redistribution)


class Use(StrEnum):
    JUDGE_TRAIN = "judge_train"
    DEV_REF = "dev_ref"


class RowFilter(BaseModel):
    """A declarative row filter. ``field`` is a dotted path; list indices are integers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str
    op: Literal["equals", "in", "not_in", "contains_any", "len_max", "len_min"]
    value: Any
    why: str = ""


class Verified(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checked: dt.date
    method: str
    revision_ref: str = "main"  # branch the pinned sha came from


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    channel: Channel
    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")  # pinned commit sha
    data_files: list[str]  # globs inside the repo
    exclude_files: list[str] = []
    split: str
    text_field: str
    id_field: str | None = None  # None: the row position is the id
    date_field: str | None = None
    date_format: str | None = None  # "iso", "unix", or a strptime pattern
    pre_cutoff_by_construction: bool = False
    construction_note: str = ""
    cutoff: dt.date = DEFAULT_CUTOFF
    licence: str
    licence_status: LicenceStatus
    licence_notes: str
    allowed_uses: list[Use] = Field(min_length=1)
    cleaners: list[str] = []
    filters: list[RowFilter] = []
    meta_fields: dict[str, str] = {}  # output key -> dotted path
    quotas: dict[Band, int] | None = None  # override the channel default
    verified: Verified
    notes: str = ""

    @field_validator("allowed_uses", mode="before")
    @classmethod
    def _no_test_refs(cls, v: Any) -> Any:
        if isinstance(v, list) and any(str(u) in {"test_ref", "test"} for u in v):
            raise ValueError("archival text can never be a test reference (DESIGN.md section 3)")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> Source:
        if Use.DEV_REF in self.allowed_uses and self.licence_status is not LicenceStatus.CLEAR:
            raise ValueError(f"{self.id}: dev_ref is public, so it needs a clear licence")
        if self.pre_cutoff_by_construction and not self.construction_note:
            raise ValueError(f"{self.id}: say why the source is pre-cutoff by construction")
        if self.date_field is None and not self.pre_cutoff_by_construction:
            raise ValueError(f"{self.id}: no date field and not pre-cutoff by construction")
        from humanbench.sourcing.cleaners import CLEANERS  # local import avoids a cycle

        unknown = [c for c in self.cleaners if c not in CLEANERS]
        if unknown:
            raise ValueError(f"{self.id}: unknown cleaners {unknown}")
        return self

    def band_quotas(self, defaults: dict[Band, int]) -> dict[Band, int]:
        """Per-band caps for this source. Bands the channel never uses get 0."""
        if self.quotas is not None:
            return {b: self.quotas.get(b, 0) for b in Band}
        used = channel_bands(self.channel)
        return {b: (defaults.get(b, 0) if b in used else 0) for b in Band}


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_quotas: dict[Band, int]
    no_source: dict[Channel, str] = {}  # channels with no honest public source, and why
    rejected_candidates: list[dict[str, str]] = []
    sources: list[Source]

    @model_validator(mode="after")
    def _unique(self) -> Registry:
        ids = [s.id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate source ids")
        covered = {s.channel for s in self.sources}
        for ch in Channel:
            if ch not in covered and ch not in self.no_source:
                raise ValueError(f"channel {ch} has no source and no 'no_source' explanation")
        return self

    def get(self, source_id: str) -> Source:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)

    def for_channel(self, channel: Channel) -> list[Source]:
        return [s for s in self.sources if s.channel == channel]


def channel_bands(channel: Channel) -> frozenset[Band]:
    """Length bands the benchmark uses for this channel (union over modes)."""
    out: set[Band] = set()
    for (ch, _mode), bands in ALLOWED_BANDS.items():
        if ch == channel:
            out |= bands
    return frozenset(out)


def load_registry(path: Path | None = None) -> Registry:
    if path is None:
        raw = resources.files("humanbench.sourcing").joinpath("sources.yaml").read_text(encoding="utf-8")
    else:
        raw = Path(path).read_text(encoding="utf-8")
    return Registry.model_validate(yaml.safe_load(raw))
