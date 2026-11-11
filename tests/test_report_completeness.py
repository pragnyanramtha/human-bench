"""The results report must not print a complete rate for incomplete data (review of 2026-11-11)."""
from pathlib import Path

from humanbench.harness import report_v01 as rep
from humanbench.harness.cards import fingerprint
from humanbench.taskpkg import load_package

REPO = Path(__file__).parents[1]
CARDS = {name: load_package(REPO / "tasks" / "dev" / name, require_oracle=False) for name in ("blog-01", "blog-03")}


class Judge:
    """Every text is scored and reads as human, except those listed as unknown."""

    def __init__(self, unknown: tuple[str, ...] = ()) -> None:
        self.unknown = unknown

    def score(self, detector: str, text: str) -> float | None:
        return None if text in self.unknown else 0.01

    def human(self, detector: str, text: str) -> bool | None:
        return None if text in self.unknown else True


def record(model: str, task: str, sample: int, run: str = "run-a") -> dict:
    return {"model": model, "task": task, "sample": sample, "output": "A short answer.", "_run": run,
            "card_fingerprint": fingerprint(CARDS[task])}


def headline_cell(lines: list[str], model: str) -> str:
    start = next(i for i, x in enumerate(lines) if x.startswith("**Pass all gates**"))
    row = next(x for x in lines[start:] if x.startswith(f"| {model} |"))
    return row.split("|")[2].strip().strip("*")


def test_one_output_out_of_many_is_not_a_complete_rate():
    lines = rep.section("t", [record("m", "blog-01", 1)], CARDS, Judge(), "dev", {("run-a", "m"): 3})
    assert headline_cell(lines, "m") == "incomplete (5 outputs missing)"  # 2 cards x 3 samples, 1 present


def test_without_a_manifest_the_cards_with_no_output_still_count_as_missing():
    lines = rep.section("t", [record("m", "blog-01", 1)], CARDS, Judge(), "dev")
    assert headline_cell(lines, "m") == "incomplete (1 outputs missing)"  # the other card has nothing


def test_a_model_the_run_was_asked_for_but_that_has_no_output_is_shown_as_missing():
    recs = [record("m", t, 1) for t in CARDS]
    lines = rep.section("t", recs, CARDS, Judge(), "dev", {("run-a", "m"): 1, ("run-a", "absent"): 1})
    assert headline_cell(lines, "absent") == "incomplete (2 outputs missing)"
    assert not headline_cell(lines, "m").startswith("incomplete (2 outputs")


def test_the_same_model_in_two_runs_is_counted_per_run():
    recs = [record("m", t, 1, run) for t in CARDS for run in ("run-a", "run-b")]
    lines = rep.section("t", recs, CARDS, Judge(), "dev")
    row = next(x for x in lines if x.startswith("| m | 2 |"))  # two samples per card, one from each run
    assert "4 (0)" in row


def test_a_reference_card_with_an_unscored_human_piece_is_left_out(monkeypatch):
    monkeypatch.setattr(rep.refs, "eligible", lambda slot: (True, "memorisation probe: clean"))
    monkeypatch.setattr(rep, "reference_judge_text", lambda pkg: f"human piece for {pkg.card.id}")
    usable, note = rep.reference_cards(CARDS, Judge(unknown=("human piece for blog-03",)))
    assert list(usable) == ["blog-01"]
    assert any("blog-03" in n and "not scored" in n for n in note)


def test_an_output_that_passes_the_checks_but_has_no_floor_verdict_is_unscored(monkeypatch):
    """Exercises the path where the floor verdict is looked up (it once clashed with the sample bookkeeping)."""
    passing = {"deterministic": 1.0, "failures": [], "task_failures": [], "scored_text": "x", "authored_text": "x"}
    monkeypatch.setattr(rep, "score_output", lambda pkg, text: passing)
    recs = [record("m", t, 1) for t in CARDS]
    lines = rep.section("t", recs, CARDS, Judge(), "dev", {("run-a", "m"): 1})
    assert headline_cell(lines, "m") == "incomplete (2 unscored)"
    assert next(x for x in lines if x.startswith("| m | 1 |")).split("|")[3].strip() == "2 (0)"


def test_published_cutoffs_stand_in_when_the_human_texts_are_absent(tmp_path, monkeypatch):
    """A public checkout has no human calibration texts: the report then scores against the published cutoffs."""
    import json

    from humanbench.harness import report_v01 as rv

    monkeypatch.setattr(rv, "CACHE", tmp_path)  # no detector scores, no human pool
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({d: {"cutoff": 0.25, "humans_scored": 801} for d in rv.DETECTORS}))
    monkeypatch.setattr(rv, "CALIBRATION", cal)
    judge = rv.Judge()
    assert judge.published == list(rv.DETECTORS)
    assert judge.cut[rv.HEADLINE] == 0.25 and judge.n_pool[rv.HEADLINE] == 801
    monkeypatch.setattr(rv, "CALIBRATION", tmp_path / "missing.json")
    assert rv.Judge().cut[rv.HEADLINE] is None  # nothing to stand in: outputs stay unscored
