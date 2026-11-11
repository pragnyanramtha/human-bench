"""On-disk layout for Lane 2 (commissioned references), all under private/refs/.

    private/refs/
      briefs/<task_id>.md                        writer brief
      screeners/<persona_id>.json                recruiting criteria
      submissions/<task_id>/<submission_id>.json raw submission + keystroke log, as received
      qa/<task_id>/<submission_id>.json          QA report (gates, reasons, decision)
      reviews/<task_id>/<submission_id>.json     a person's decision on a needs_human_review item
      accepted/<task_id>/<submission_id>.json    the reference: normalized text + provenance

Submissions are write-once: saving different content under an existing id is
an error, so a resubmission needs a new submission_id.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from humanbench.schema import Persona, TaskCard, load_personas
from humanbench.scoring.normalize import normalize, word_count
from humanbench.sourcing.paths import DEFAULT_PRIVATE, dump_json, refs_dir, write_atomic
from humanbench.sourcing.qa import Decision, QAReport
from humanbench.sourcing.screener import ScreenerSpec
from humanbench.sourcing.submission import Submission


class Review(BaseModel):
    submission_id: str
    task_id: str
    reviewer: str
    decision: Literal["accept", "reject"]
    note: str
    reviewed_at: dt.datetime


def load_card(path: Path, personas: list[Persona] | None = None) -> TaskCard:
    """Load a task card from YAML or JSON. ``persona`` may be omitted; it is filled from persona_id."""
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("persona"), dict):
        pid = data.get("persona_id") or data.get("persona")
        pool = {p.id: p for p in (personas or load_personas())}
        if pid not in pool:
            raise ValueError(f"{path}: unknown persona {pid!r}")
        data["persona"] = pool[pid]
        data["persona_id"] = pid
    return TaskCard.model_validate(data)


class RefStore:
    def __init__(self, private: Path = DEFAULT_PRIVATE):
        self.root = refs_dir(private)

    def _path(self, kind: str, *parts: str) -> Path:
        return self.root.joinpath(kind, *parts)

    def write_brief(self, task_id: str, markdown: str) -> Path:
        path = self._path("briefs", f"{task_id}.md")
        write_atomic(path, markdown)
        return path

    def write_screener(self, spec: ScreenerSpec) -> Path:
        path = self._path("screeners", f"{spec.persona_id}.json")
        write_atomic(path, dump_json(spec.model_dump(mode="json")))
        return path

    def save_submission(self, sub: Submission) -> Path:
        path = self._path("submissions", sub.task_id, f"{sub.submission_id}.json")
        body = dump_json(sub.model_dump(mode="json"))
        if path.exists() and path.read_text(encoding="utf-8") != body:
            raise FileExistsError(f"{path} exists with different content; use a new submission_id")
        write_atomic(path, body)
        return path

    def submissions(self, task_id: str | None = None) -> list[Submission]:
        base = self._path("submissions")
        pattern = f"{task_id}/*.json" if task_id else "*/*.json"
        return [Submission.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(base.glob(pattern))]

    def save_qa(self, report: QAReport) -> Path:
        path = self._path("qa", report.task_id, f"{report.submission_id}.json")
        write_atomic(path, dump_json(report.model_dump(mode="json")))
        return path

    def qa_report(self, task_id: str, submission_id: str) -> QAReport:
        return QAReport.model_validate_json(self._path("qa", task_id, f"{submission_id}.json").read_text(encoding="utf-8"))

    def accept(self, sub: Submission, card: TaskCard, decided_by: str) -> Path:
        text = normalize(sub.text)
        record = {
            "ref_id": f"h_{card.id}_{sub.submission_id}",
            "task_id": card.id,
            "split": card.split,
            "use": "test_ref" if card.split == "test" else "dev_ref",
            "never_publish": card.split == "test",
            "submission_id": sub.submission_id,
            "writer_id": sub.writer_id,
            "persona_id": card.persona_id,
            "channel": str(card.channel),
            "band": str(card.band),
            "word_count": word_count(text),
            "raw_sha256": hashlib.sha256(sub.text.encode("utf-8")).hexdigest(),
            "decided_by": decided_by,
            "provenance": "commissioned, written in the HumanBench editor; see submissions/ and qa/",
            "text": text,
        }
        path = self._path("accepted", card.id, f"{sub.submission_id}.json")
        write_atomic(path, dump_json(record))
        return path

    def record_review(self, review: Review, card: TaskCard) -> Path:
        report = self.qa_report(review.task_id, review.submission_id)
        if report.decision is not Decision.REVIEW:
            raise ValueError(f"{review.submission_id} was {report.decision}, not needs_human_review")
        path = self._path("reviews", review.task_id, f"{review.submission_id}.json")
        write_atomic(path, dump_json(review.model_dump(mode="json")))
        if review.decision == "accept":
            sub_path = self._path("submissions", review.task_id, f"{review.submission_id}.json")
            sub = Submission.model_validate_json(sub_path.read_text(encoding="utf-8"))
            self.accept(sub, card, decided_by=f"review:{review.reviewer}")
        return path

    def review_queue(self) -> list[QAReport]:
        out = []
        for p in sorted(self._path("qa").glob("*/*.json")):
            report = QAReport.model_validate_json(p.read_text(encoding="utf-8"))
            if report.decision is Decision.REVIEW and not self._path("reviews", report.task_id, p.name).exists():
                out.append(report)
        return out


def read_model_outputs(directory: Path) -> list[str]:
    """Model outputs for the copy check: *.txt files, or *.json / *.jsonl with a "text" field."""
    outs: list[str] = []
    for p in sorted(directory.rglob("*")):
        if p.suffix == ".txt":
            outs.append(p.read_text(encoding="utf-8"))
        elif p.suffix == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            outs.extend(o["text"] for o in (obj if isinstance(obj, list) else [obj]) if "text" in o)
        elif p.suffix == ".jsonl":
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    outs.append(json.loads(line)["text"])
    return outs
