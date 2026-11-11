"""Commissioned-writing submissions and their editor keystroke logs.

The writing editor logs events, not content: each event has a time and a
character count. That is enough to measure paste share, typing speed and
whether the final text could have come out of the logged session, without
storing every keystroke's character.
"""

from __future__ import annotations

import datetime as dt
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EventKind = Literal[
    "insert",  # ordinary typing; n is usually 1
    "delete",  # backspace / delete / selection delete
    "paste",  # should not happen: the editor blocks paste; logged if a paste gets through
    "cut",
    "composition",  # IME commit (CJK, accented input): multi-char inserts are normal
    "autocorrect",  # spellcheck / OS text replacement
    "dictation",  # speech-to-text (accessibility): large inserts are normal
    "undo",
    "redo",
    "focus",
    "blur",
]
ASSISTED_KINDS = frozenset({"composition", "autocorrect", "dictation"})
ADDING_KINDS = frozenset({"insert", "paste"}) | ASSISTED_KINDS
REMOVING_KINDS = frozenset({"delete", "cut"})


class KeystrokeEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    t_ms: int = Field(ge=0)  # milliseconds since the editor opened
    kind: EventKind
    n: int = Field(default=0, ge=0)  # characters added (insert kinds) or removed (delete, cut)
    net: int = 0  # undo / redo only: signed change in document length


class KeystrokeLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    editor: str
    editor_version: str
    paste_blocked: bool = True
    events: list[KeystrokeEvent]

    @model_validator(mode="after")
    def _monotonic(self) -> KeystrokeLog:
        if any(b.t_ms < a.t_ms for a, b in pairwise(self.events)):
            raise ValueError("keystroke events must be in time order")
        return self


class KeystrokeMetrics(BaseModel):
    events: int
    typed_chars: int
    assisted_chars: int
    pasted_chars: int
    removed_chars: int
    paste_ratio: float  # pasted chars / final length, capped at 1
    large_pastes: list[int]  # sizes of paste events at or above the threshold
    unassisted_multichar_inserts: int  # "insert" events that added more than a few chars at once
    burst_windows: int  # 1-second windows with more typed chars than a fast typist can manage
    fast_gap_share: float  # share of inserts that came < min_gap_ms after the previous one
    reconstruction_gap: float  # |net chars logged - final length| / final length
    active_seconds: float  # time between events, with idle gaps capped
    session_seconds: float


def keystroke_metrics(
    log: KeystrokeLog,
    final_len: int,
    *,
    large_paste_chars: int = 80,
    multichar_insert: int = 4,
    burst_cps: int = 20,
    min_gap_ms: int = 15,
    idle_cap_s: float = 60.0,
) -> KeystrokeMetrics:
    ev = log.events
    typed = sum(e.n for e in ev if e.kind == "insert")
    assisted = sum(e.n for e in ev if e.kind in ASSISTED_KINDS)
    pasted = sum(e.n for e in ev if e.kind == "paste")
    added = sum(e.n for e in ev if e.kind in ADDING_KINDS)
    removed = sum(e.n for e in ev if e.kind in REMOVING_KINDS)
    history = sum(e.net for e in ev if e.kind in ("undo", "redo"))

    inserts = [e for e in ev if e.kind == "insert"]
    # sliding 1-second window over typed characters
    bursts, j, window = 0, 0, 0
    in_burst = False
    for e in inserts:
        window += e.n
        while inserts[j].t_ms <= e.t_ms - 1000:
            window -= inserts[j].n
            j += 1
        if window > burst_cps and not in_burst:
            bursts += 1
            in_burst = True
        elif window <= burst_cps:
            in_burst = False
    fast = sum(1 for a, b in pairwise(inserts) if b.t_ms - a.t_ms < min_gap_ms)

    active = 0.0
    for a, b in pairwise(ev):
        active += min((b.t_ms - a.t_ms) / 1000, idle_cap_s)
    session = (ev[-1].t_ms - ev[0].t_ms) / 1000 if ev else 0.0

    final = max(final_len, 1)
    return KeystrokeMetrics(
        events=len(ev),
        typed_chars=typed,
        assisted_chars=assisted,
        pasted_chars=pasted,
        removed_chars=removed,
        paste_ratio=min(pasted / final, 1.0),
        large_pastes=[e.n for e in ev if e.kind == "paste" and e.n >= large_paste_chars],
        unassisted_multichar_inserts=sum(1 for e in inserts if e.n >= multichar_insert),
        burst_windows=bursts,
        fast_gap_share=fast / max(len(inserts) - 1, 1),
        reconstruction_gap=abs((added - removed + history) - final_len) / final,
        active_seconds=round(active, 3),
        session_seconds=round(session, 3),
    )


class Attestations(BaseModel):
    """What the writer confirmed in the contract (DESIGN.md section 3)."""

    original_work: bool
    written_without_ai_tools: bool
    never_published: bool
    rights_assigned: bool

    @property
    def complete(self) -> bool:
        return all((self.original_work, self.written_without_ai_tools, self.never_published, self.rights_assigned))


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    task_id: str
    writer_id: str  # pseudonymous id from the recruiting platform, never a name
    text: str
    started_at: dt.datetime
    submitted_at: dt.datetime
    keystroke_log: KeystrokeLog | None = None
    attestations: Attestations
    writer_note: str = ""  # the comments box; never part of the reference

    @model_validator(mode="after")
    def _times(self) -> Submission:
        if self.submitted_at < self.started_at:
            raise ValueError("submitted_at is before started_at")
        return self

    @property
    def seconds_on_task(self) -> float:
        return (self.submitted_at - self.started_at).total_seconds()
