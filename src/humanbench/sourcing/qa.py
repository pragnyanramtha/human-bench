"""QA gates for commissioned submissions.

Gates run in this order, each with reasons:

    contract     attestations present                      (fail -> reject)
    verifiers    humanbench.scoring.verifiers.run_verifiers (per-verifier policy)
    time         minimum time on task, composition speed
    keystrokes   paste share, large pastes, typing bursts, log/text mismatch
    similarity   copy check against model outputs and other submissions
    detector     pluggable AI-detector screen (flag -> review, NEVER reject)

Outcomes: pass, flag (send to a person), fail (reject), skip, error. The
decision is reject if any non-detector gate fails, otherwise needs_human_review
if anything flagged or errored, otherwise accept.

A detector-only flag never rejects. Detectors have high false-positive rates
on human text, worst for non-native writers (DESIGN.md section 6), so their
output is advice for the human reviewer. ``decide`` enforces this even for a
detector gate that reports a failure.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from humanbench.schema import TaskCard
from humanbench.scoring.normalize import normalize, word_count
from humanbench.scoring.verifiers import run_verifiers
from humanbench.sourcing.dedup import jaccard, shingles
from humanbench.sourcing.submission import Submission, keystroke_metrics

QA_VERSION = 1
DETECTOR_GATES = frozenset({"detector"})


class Outcome(StrEnum):
    PASS = "pass"
    FLAG = "flag"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class Decision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    REVIEW = "needs_human_review"


Action = Literal["reject", "review", "ignore"]


class GateResult(BaseModel):
    gate: str
    outcome: Outcome
    reasons: list[str] = []
    metrics: dict[str, Any] = {}


class QAConfig(BaseModel):
    # What a failed verifier means for a human submission. Humans legitimately
    # write "I hope this helps" (wrapper) or carry odd characters from their
    # keyboard (hidden_chars), so those go to review rather than rejection.
    verifier_policy: dict[str, Action] = {
        "length": "reject",
        "format": "reject",
        "wrapper": "review",
        "must_include": "reject",
        "must_not_include": "reject",
        "hidden_chars": "review",
        "judge_injection": "reject",
        "copy": "reject",
        "noise": "review",
    }
    # Writers are never shown the requirement lists (see briefs.requirements_shown),
    # so missing one is a judgement call, and often a sign the anchor is unfair.
    agent_requirement_action: Action = "review"  # applies to every mode
    min_seconds: float = 180.0
    review_wpm: float = 80.0  # faster composition than this goes to review
    reject_wpm: float = 150.0  # faster than this is not plausible even retyping a draft
    require_keystroke_log: bool = False  # missing log: review (False) or reject (True)
    paste_ratio_review: float = 0.02
    paste_ratio_reject: float = 0.25
    large_paste_chars: int = 80
    multichar_insert_review: int = 5
    burst_cps: int = 20
    fast_gap_share_review: float = 0.05
    reconstruction_gap_review: float = 0.15
    copy_ngram: int = 13
    model_runs_review: int = 1  # shared n-gram runs with a model output
    model_runs_reject: int = 3
    jaccard_review: float = 0.25  # 5-gram shingle Jaccard
    jaccard_reject: float = 0.5
    detector_threshold: float = 0.8


@runtime_checkable
class Detector(Protocol):
    """Pluggable AI-detector screen. ``score`` returns P(AI-written) in [0, 1]."""

    name: str
    version: str

    def score(self, text: str) -> float: ...


class QAReport(BaseModel):
    qa_version: int = QA_VERSION
    submission_id: str
    task_id: str
    writer_id: str
    decision: Decision
    gates: list[GateResult]
    word_count: int
    text_sha256: str
    config: QAConfig


@dataclass
class QAContext:
    card: TaskCard
    model_outputs: Sequence[str] = ()
    other_submissions: Sequence[Submission] = ()
    detectors: Sequence[Detector] = ()
    context_text: str = ""  # the thread / environment text the writer saw
    extra_material: Sequence[str] = field(default_factory=tuple)

    def task_material(self) -> str:
        n = self.card.notes_pack
        parts = [self.card.prompt, *n.facts, *n.anecdotes, *n.opinions, *n.wants, self.context_text,
                 *(r if isinstance(r, str) else " ".join(r) for r in self.card.must_include), *self.extra_material]
        return "\n".join(p for p in parts if p)


_WORD = re.compile(r"\w+", re.UNICODE)


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = [x.lower() for x in _WORD.findall(text)]
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def _combine(outcomes: Sequence[Outcome]) -> Outcome:
    for o in (Outcome.FAIL, Outcome.FLAG, Outcome.ERROR):
        if o in outcomes:
            return o
    return Outcome.PASS


# --- gates --------------------------------------------------------------------------


def gate_contract(sub: Submission) -> GateResult:
    a = sub.attestations
    missing = [k for k, v in a.model_dump().items() if not v]
    if missing:
        return GateResult(gate="contract", outcome=Outcome.FAIL, reasons=[f"attestation not confirmed: {k}" for k in missing])
    return GateResult(gate="contract", outcome=Outcome.PASS)


def gate_verifiers(text: str, card: TaskCard, cfg: QAConfig, material: Sequence[str] = ()) -> GateResult:
    """Same deterministic layer the models get, including the material-copy share."""
    report = run_verifiers(
        text,
        channel=card.channel,
        band=card.words,
        must_include=card.must_include,
        must_not_include=card.must_not_include,
        format_overrides=card.format_overrides,
        material=material,
    )
    outcomes, reasons = [], []
    for r in report.failures:
        action = cfg.verifier_policy.get(r.name, "reject")
        if r.name in ("must_include", "must_not_include"):
            action = cfg.agent_requirement_action
        if action == "ignore":
            continue
        outcomes.append(Outcome.FAIL if action == "reject" else Outcome.FLAG)
        reasons.append(f"{r.name} ({action}): {r.detail}")
    return GateResult(gate="verifiers", outcome=_combine(outcomes), reasons=reasons,
                      metrics={"failed": [r.name for r in report.failures]})


def gate_time(sub: Submission, words: int, cfg: QAConfig) -> GateResult:
    secs = sub.seconds_on_task
    wpm = words / max(secs / 60, 1e-9)
    m = {"seconds_on_task": round(secs, 1), "words": words, "words_per_minute": round(wpm, 1)}
    if secs < cfg.min_seconds:
        return GateResult(gate="time", outcome=Outcome.FAIL, metrics=m,
                          reasons=[f"{secs:.0f}s on task, minimum is {cfg.min_seconds:.0f}s"])
    if wpm > cfg.reject_wpm:
        return GateResult(gate="time", outcome=Outcome.FAIL, metrics=m,
                          reasons=[f"{wpm:.0f} words/minute over the whole task (limit {cfg.reject_wpm:.0f})"])
    if wpm > cfg.review_wpm:
        return GateResult(gate="time", outcome=Outcome.FLAG, metrics=m,
                          reasons=[f"{wpm:.0f} words/minute is fast for composing (review above {cfg.review_wpm:.0f})"])
    return GateResult(gate="time", outcome=Outcome.PASS, metrics=m)


def gate_keystrokes(sub: Submission, cfg: QAConfig) -> GateResult:
    log = sub.keystroke_log
    if log is None:
        outcome = Outcome.FAIL if cfg.require_keystroke_log else Outcome.FLAG
        return GateResult(gate="keystrokes", outcome=outcome, reasons=["no keystroke log with the submission"])
    km = keystroke_metrics(log, len(sub.text), large_paste_chars=cfg.large_paste_chars, burst_cps=cfg.burst_cps)
    outcomes, reasons = [], []
    if km.paste_ratio >= cfg.paste_ratio_reject:
        outcomes.append(Outcome.FAIL)
        reasons.append(f"{km.paste_ratio:.0%} of the text was pasted (limit {cfg.paste_ratio_reject:.0%})")
    elif km.paste_ratio >= cfg.paste_ratio_review:
        outcomes.append(Outcome.FLAG)
        reasons.append(f"{km.paste_ratio:.0%} of the text was pasted")
    if km.large_pastes:
        outcomes.append(Outcome.FLAG)
        reasons.append(f"large paste events: {km.large_pastes} chars")
    if not log.paste_blocked:
        outcomes.append(Outcome.FLAG)
        reasons.append("editor reports paste was not blocked in this session")
    if km.unassisted_multichar_inserts >= cfg.multichar_insert_review:
        outcomes.append(Outcome.FLAG)
        reasons.append(f"{km.unassisted_multichar_inserts} multi-character inserts not from IME, autocorrect or dictation")
    if km.burst_windows:
        outcomes.append(Outcome.FLAG)
        reasons.append(f"{km.burst_windows} typing bursts above {cfg.burst_cps} chars/second")
    if km.fast_gap_share > cfg.fast_gap_share_review:
        outcomes.append(Outcome.FLAG)
        reasons.append(f"{km.fast_gap_share:.0%} of keystrokes less than 15 ms apart")
    if km.reconstruction_gap > cfg.reconstruction_gap_review:
        outcomes.append(Outcome.FLAG)
        reasons.append(f"logged edits don't add up to the submitted text (gap {km.reconstruction_gap:.0%})")
    return GateResult(gate="keystrokes", outcome=_combine(outcomes), reasons=reasons, metrics=km.model_dump())


def gate_similarity(sub: Submission, text: str, ctx: QAContext, cfg: QAConfig) -> GateResult:
    n = cfg.copy_ngram
    material = _ngrams(normalize(ctx.task_material()), n)
    material_sh = shingles(ctx.task_material())
    mine = _ngrams(text, n) - material
    mine_sh = shingles(text) - material_sh
    outcomes, reasons = [], []
    metrics: dict[str, Any] = {"max_model_runs": 0, "max_model_jaccard": 0.0, "max_peer_jaccard": 0.0}

    for i, out in enumerate(ctx.model_outputs):
        out_n = normalize(out)
        runs = len(mine & _ngrams(out_n, n))
        jac = jaccard(mine_sh, shingles(out_n) - material_sh)
        metrics["max_model_runs"] = max(metrics["max_model_runs"], runs)
        metrics["max_model_jaccard"] = round(max(metrics["max_model_jaccard"], jac), 4)
        if runs >= cfg.model_runs_reject or jac >= cfg.jaccard_reject:
            outcomes.append(Outcome.FAIL)
            reasons.append(f"model output {i}: {runs} shared {n}-word runs, Jaccard {jac:.2f}")
        elif runs >= cfg.model_runs_review or jac >= cfg.jaccard_review:
            outcomes.append(Outcome.FLAG)
            reasons.append(f"model output {i}: {runs} shared {n}-word runs, Jaccard {jac:.2f}")

    for other in ctx.other_submissions:
        if other.submission_id == sub.submission_id:
            continue
        if other.writer_id == sub.writer_id and other.task_id == sub.task_id:
            continue  # the same writer's earlier draft of this task
        jac = jaccard(mine_sh, shingles(normalize(other.text)) - material_sh)
        metrics["max_peer_jaccard"] = round(max(metrics["max_peer_jaccard"], jac), 4)
        if jac < cfg.jaccard_review:
            continue
        same_writer = other.writer_id == sub.writer_id
        severe = jac >= cfg.jaccard_reject and not same_writer
        outcomes.append(Outcome.FAIL if severe else Outcome.FLAG)
        who = "same writer, other task" if same_writer else f"writer {other.writer_id}"
        reasons.append(f"submission {other.submission_id} ({who}): Jaccard {jac:.2f}")
    return GateResult(gate="similarity", outcome=_combine(outcomes), reasons=reasons, metrics=metrics)


def gate_detector(text: str, ctx: QAContext, cfg: QAConfig, already_rejected: bool) -> GateResult:
    if already_rejected:
        return GateResult(gate="detector", outcome=Outcome.SKIP, reasons=["already rejected; detector not called"])
    if not ctx.detectors:
        return GateResult(gate="detector", outcome=Outcome.SKIP, reasons=["no detector configured"])
    outcomes, reasons, scores = [], [], {}
    for det in ctx.detectors:
        label = f"{det.name} {det.version}"
        try:
            p = float(det.score(text))
        except Exception as e:  # a detector outage must not block or reject anyone
            outcomes.append(Outcome.ERROR)
            reasons.append(f"{label}: error {type(e).__name__}: {e}")
            continue
        scores[label] = round(p, 4)
        if p >= cfg.detector_threshold:
            outcomes.append(Outcome.FLAG)
            reasons.append(f"{label}: P(AI)={p:.2f} >= {cfg.detector_threshold:.2f}. Detector flags go to a person, never auto-reject.")
    if Outcome.FLAG in outcomes and not ctx.card.persona.native:
        reasons.append("Writer persona is a non-native English writer; detectors over-flag non-native writing.")
    return GateResult(gate="detector", outcome=_combine(outcomes), reasons=reasons, metrics={"scores": scores})


def decide(gates: Sequence[GateResult]) -> Decision:
    """Reject only on a non-detector failure. A detector can at most send the piece to review."""
    if any(g.outcome is Outcome.FAIL and g.gate not in DETECTOR_GATES for g in gates):
        return Decision.REJECT
    if any(g.outcome in (Outcome.FLAG, Outcome.ERROR, Outcome.FAIL) for g in gates):
        return Decision.REVIEW
    return Decision.ACCEPT


def run_qa(sub: Submission, ctx: QAContext, cfg: QAConfig | None = None) -> QAReport:
    cfg = cfg or QAConfig()
    if sub.task_id != ctx.card.id:
        raise ValueError(f"submission is for task {sub.task_id}, card is {ctx.card.id}")
    text = normalize(sub.text)
    words = word_count(text)
    gates = [
        gate_contract(sub),
        gate_verifiers(text, ctx.card, cfg, material=[ctx.task_material()]),
        gate_time(sub, words, cfg),
        gate_keystrokes(sub, cfg),
        gate_similarity(sub, text, ctx, cfg),
    ]
    rejected = any(g.outcome is Outcome.FAIL for g in gates)
    gates.append(gate_detector(text, ctx, cfg, rejected))
    return QAReport(
        submission_id=sub.submission_id,
        task_id=sub.task_id,
        writer_id=sub.writer_id,
        decision=decide(gates),
        gates=gates,
        word_count=words,
        text_sha256=hashlib.sha256(sub.text.encode("utf-8")).hexdigest(),
        config=cfg,
    )
