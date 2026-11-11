import datetime as dt
import random

import pytest

from humanbench.schema import NotesPack, TaskCard, load_personas
from humanbench.sourcing.qa import Decision, GateResult, Outcome, QAContext, decide, run_qa
from humanbench.sourcing.refs import RefStore, Review
from humanbench.sourcing.submission import Attestations, KeystrokeEvent, KeystrokeLog, Submission, keystroke_metrics

PERSONAS = {p.id: p for p in load_personas()}
FUNC = "the and to of a in that it is was for on with as at by this but not or from have they we you he she".split()
SYLL = "ka lo mi ren tu sha vel dor pi nax ber lin qua so fen tor gal mu".split()
T0 = dt.datetime(2026, 11, 1, 10, 0, tzinfo=dt.UTC)


def make_text(seed: int, n_words: int = 320) -> str:
    rng = random.Random(seed)
    words = [rng.choice(FUNC) if rng.random() < 0.5 else "".join(rng.choice(SYLL) for _ in range(2))
             for _ in range(n_words)]
    sentences = [" ".join(words[i : i + 12]).capitalize() + "." for i in range(0, n_words, 12)]
    return "\n\n".join(" ".join(sentences[i : i + 4]) for i in range(0, len(sentences), 4))


def card(persona_id: str = "maths-phd") -> TaskCard:
    p = PERSONAS[persona_id]
    return TaskCard(
        id="forum-03", split="test", channel="forum", mode="direct", tone="casual", audience="community",
        stakes="routine", band="short", persona_id=p.id, persona=p,
        prompt="Help me write a post for the number theory forum about the reading group moving to Thursdays.",
        notes_pack=NotesPack(facts=["the reading group moves to Thursdays"]),
    )


def typed_log(text: str, seed: int = 0) -> KeystrokeLog:
    rng = random.Random(seed)
    t, events = 0, []
    for _ in text:
        t += rng.randint(150, 400)
        events.append(KeystrokeEvent(t_ms=t, kind="insert", n=1))
    events.insert(10, KeystrokeEvent(t_ms=events[9].t_ms, kind="delete", n=0))
    return KeystrokeLog(editor="hb-editor", editor_version="1", events=events)


def submission(text: str, *, sid: str = "s1", writer: str = "w1", minutes: float = 25, log="typed", **kw) -> Submission:
    return Submission(
        submission_id=sid, task_id="forum-03", writer_id=writer, text=text, started_at=T0,
        submitted_at=T0 + dt.timedelta(minutes=minutes),
        keystroke_log=typed_log(text) if log == "typed" else log,
        attestations=kw.pop("attestations", Attestations(original_work=True, written_without_ai_tools=True,
                                                          never_published=True, rights_assigned=True)),
        **kw,
    )


GOOD = make_text(1) + " Also the reading group moves to Thursdays."


class StubDetector:
    name, version = "stub", "0"

    def __init__(self, p: float):
        self.p = p

    def score(self, text: str) -> float:
        return self.p


class BrokenDetector:
    name, version = "broken", "0"

    def score(self, text: str) -> float:
        raise TimeoutError("vendor down")


def gate(report, name):
    return next(g for g in report.gates if g.gate == name)


def test_clean_submission_is_accepted():
    r = run_qa(submission(GOOD), QAContext(card=card(), detectors=[StubDetector(0.1)]))
    assert r.decision is Decision.ACCEPT, [(g.gate, g.outcome, g.reasons) for g in r.gates]
    assert [g.gate for g in r.gates] == ["contract", "verifiers", "time", "keystrokes", "similarity", "detector"]


def test_detector_only_flag_never_rejects():
    r = run_qa(submission(GOOD), QAContext(card=card("truck-driver"), detectors=[StubDetector(1.0)]))
    assert r.decision is Decision.REVIEW
    det = gate(r, "detector")
    assert det.outcome is Outcome.FLAG
    assert any("non-native" in x for x in det.reasons)


def test_decide_coerces_a_detector_failure_to_review():
    gates = [GateResult(gate="verifiers", outcome=Outcome.PASS), GateResult(gate="detector", outcome=Outcome.FAIL)]
    assert decide(gates) is Decision.REVIEW


def test_detector_error_goes_to_review_not_reject():
    r = run_qa(submission(GOOD), QAContext(card=card(), detectors=[BrokenDetector()]))
    assert r.decision is Decision.REVIEW and gate(r, "detector").outcome is Outcome.ERROR


def test_detector_is_skipped_once_rejected():
    r = run_qa(submission(GOOD, minutes=1), QAContext(card=card(), detectors=[StubDetector(1.0)]))
    assert r.decision is Decision.REJECT
    assert gate(r, "detector").outcome is Outcome.SKIP


def test_verifier_failures_follow_policy():
    short = submission("Too short. The reading group moves to Thursdays.", minutes=10)
    r = run_qa(short, QAContext(card=card()))
    assert r.decision is Decision.REJECT and gate(r, "verifiers").outcome is Outcome.FAIL
    wrapper = submission(GOOD + "\n\nThanks,\n[Your Name]")
    r = run_qa(wrapper, QAContext(card=card()))
    assert gate(r, "verifiers").outcome is Outcome.FLAG and r.decision is Decision.REVIEW


def test_time_gate():
    assert run_qa(submission(GOOD, minutes=2), QAContext(card=card())).decision is Decision.REJECT
    r = run_qa(submission(GOOD, minutes=3.5), QAContext(card=card()))  # ~95 wpm
    assert gate(r, "time").outcome is Outcome.FLAG


def test_keystroke_paste_and_bursts():
    pasted = KeystrokeLog(editor="hb-editor", editor_version="1", events=[
        KeystrokeEvent(t_ms=0, kind="insert", n=1), KeystrokeEvent(t_ms=500, kind="paste", n=len(GOOD))])
    r = run_qa(submission(GOOD, log=pasted), QAContext(card=card()))
    assert r.decision is Decision.REJECT and gate(r, "keystrokes").outcome is Outcome.FAIL

    burst = KeystrokeLog(editor="hb-editor", editor_version="1",
                         events=[KeystrokeEvent(t_ms=i * 5, kind="insert", n=1) for i in range(len(GOOD))])
    r = run_qa(submission(GOOD, log=burst), QAContext(card=card()))
    ks = gate(r, "keystrokes")
    assert ks.outcome is Outcome.FLAG and r.decision is Decision.REVIEW
    assert ks.metrics["burst_windows"] >= 1 and ks.metrics["fast_gap_share"] > 0.9


def test_assisted_input_is_not_penalised():
    events = [KeystrokeEvent(t_ms=i * 4000, kind="dictation", n=40) for i in range(len(GOOD) // 40)]
    events.append(KeystrokeEvent(t_ms=len(events) * 4000, kind="insert", n=len(GOOD) - 40 * (len(GOOD) // 40)))
    log = KeystrokeLog(editor="hb-editor", editor_version="1", events=events)
    r = run_qa(submission(GOOD, log=log), QAContext(card=card()))
    assert gate(r, "keystrokes").outcome is Outcome.PASS


def test_missing_log_goes_to_review():
    r = run_qa(submission(GOOD, log=None), QAContext(card=card()))
    assert r.decision is Decision.REVIEW and gate(r, "keystrokes").outcome is Outcome.FLAG


def test_keystroke_metrics_reconstruction():
    log = KeystrokeLog(editor="e", editor_version="1", events=[
        KeystrokeEvent(t_ms=0, kind="insert", n=1), KeystrokeEvent(t_ms=200, kind="insert", n=1),
        KeystrokeEvent(t_ms=400, kind="delete", n=1), KeystrokeEvent(t_ms=600, kind="undo", net=1)])
    km = keystroke_metrics(log, final_len=2)
    assert km.reconstruction_gap == 0 and km.typed_chars == 2 and km.session_seconds == 0.6
    with pytest.raises(ValueError):
        KeystrokeLog(editor="e", editor_version="1",
                     events=[KeystrokeEvent(t_ms=5, kind="insert", n=1), KeystrokeEvent(t_ms=1, kind="insert", n=1)])


def test_copy_of_a_model_output_is_rejected():
    model_out = GOOD
    r = run_qa(submission(GOOD), QAContext(card=card(), model_outputs=[model_out, make_text(50)]))
    assert r.decision is Decision.REJECT and gate(r, "similarity").outcome is Outcome.FAIL


def test_quoting_the_notes_pack_is_not_a_copy():
    notes = "the reading group moves to Thursdays and the room is the same as before for everyone"
    c = card()
    c.notes_pack.facts.append(notes)
    mine = make_text(7) + " " + notes + "."
    model = make_text(8) + " " + notes + "."
    r = run_qa(submission(mine), QAContext(card=c, model_outputs=[model]))
    assert gate(r, "similarity").outcome is Outcome.PASS


def test_peer_copy_rejected_but_own_draft_ignored():
    other = submission(GOOD, sid="s0", writer="w2")
    r = run_qa(submission(GOOD), QAContext(card=card(), other_submissions=[other]))
    assert gate(r, "similarity").outcome is Outcome.FAIL
    own_draft = submission(GOOD, sid="s0", writer="w1")
    r = run_qa(submission(GOOD), QAContext(card=card(), other_submissions=[own_draft]))
    assert gate(r, "similarity").outcome is Outcome.PASS


def test_missing_attestation_rejects():
    att = Attestations(original_work=True, written_without_ai_tools=False, never_published=True, rights_assigned=True)
    assert run_qa(submission(GOOD, attestations=att), QAContext(card=card())).decision is Decision.REJECT


def test_task_mismatch_raises():
    sub = submission(GOOD).model_copy(update={"task_id": "other"})
    with pytest.raises(ValueError):
        run_qa(sub, QAContext(card=card()))


def test_refstore_flow(tmp_path):
    store = RefStore(tmp_path / "private")
    assert (tmp_path / "private" / ".gitignore").read_text().strip().endswith("*")
    c = card()
    good, flagged = submission(GOOD), submission(make_text(2) + " The reading group moves to Thursdays.", sid="s2")
    for sub, det in ((good, 0.1), (flagged, 0.99)):
        store.save_submission(sub)
        report = run_qa(sub, QAContext(card=c, other_submissions=store.submissions(), detectors=[StubDetector(det)]))
        store.save_qa(report)
        if report.decision is Decision.ACCEPT:
            store.accept(sub, c, decided_by="auto")
    assert (store.root / "accepted" / "forum-03" / "s1.json").exists()
    queue = store.review_queue()
    assert [q.submission_id for q in queue] == ["s2"]
    store.record_review(Review(submission_id="s2", task_id="forum-03", reviewer="ops1", decision="accept",
                               note="reads like the writer's screener sample", reviewed_at=T0), c)
    assert store.review_queue() == []
    assert (store.root / "accepted" / "forum-03" / "s2.json").exists()
    with pytest.raises(FileExistsError):
        store.save_submission(good.model_copy(update={"text": "changed"}))


def test_agent_task_requirement_misses_go_to_review():
    p = PERSONAS["leeds-plumber"]
    agent = TaskCard(
        id="forum-03", split="test", channel="email", mode="agent", tone="angry", audience="business",
        stakes="tense", band="short", persona_id=p.id, persona=p, environment="envs/inbox-11/",
        prompt="They've sent the wrong valves again. Sort it. Draft replies to whoever needs one.",
        tools=["search", "read_item", "create_draft"], must_include=["third wrong delivery"],
        agent_checks={"delivery": "draft", "required": [{"target": "orders@valve-supplier.example"}]},
    )
    r = run_qa(submission(GOOD), QAContext(card=agent))
    assert gate(r, "verifiers").outcome is Outcome.FLAG and r.decision is Decision.REVIEW
