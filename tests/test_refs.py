from pathlib import Path

import yaml

from humanbench import refs
from humanbench.taskpkg import load_package

CARD = next(iter(sorted((Path(__file__).parents[1] / "tasks" / "ref").glob("*/card.yaml"))))
REFERENCE = ("The first frost came early that year and took the tomatoes before anyone had thought to cover them, which "
             "my neighbour took as a personal insult from the weather and complained about until well into December. ") * 6


def _pkg(tmp_path, **changes):
    raw = yaml.safe_load(CARD.read_text())
    raw.update(changes)
    d = tmp_path / raw["id"]
    d.mkdir()
    (d / "card.yaml").write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    return load_package(d, require_oracle=False), raw


def test_a_real_card_does_not_leak_an_unrelated_text(tmp_path):
    pkg, _ = _pkg(tmp_path)
    assert refs.leakage(pkg, REFERENCE) == []


def test_the_whole_text_in_a_persona_field_is_caught(tmp_path):
    raw = yaml.safe_load(CARD.read_text())
    persona = {**raw["persona"], "role": REFERENCE}
    pkg, _ = _pkg(tmp_path, persona=persona)
    problems = refs.leakage(pkg, REFERENCE)
    assert any("8+ words" in p for p in problems) and any("persona.role" in p for p in problems)


def test_lightly_altered_prose_in_the_notes_is_caught(tmp_path):
    raw = yaml.safe_load(CARD.read_text())
    words = REFERENCE.split()
    altered = " ".join("zzz" if i % 7 == 6 else w for i, w in enumerate(words))  # every seventh word replaced
    chunks = [" ".join(altered.split()[i:i + 28]) for i in range(0, 200, 28)]
    notes = {**raw["notes_pack"], "facts": chunks}
    pkg, _ = _pkg(tmp_path, notes_pack=notes)
    assert any("5-word sequences" in p for p in refs.leakage(pkg, REFERENCE))


def test_an_unprobed_card_is_not_eligible(monkeypatch, tmp_path):
    monkeypatch.setattr(refs, "PROBE", tmp_path / "none.json")
    monkeypatch.setattr(refs, "LAUNCH", tmp_path / "launch.txt")
    (tmp_path / "launch.txt").write_text("model-a\nmodel-b\n")
    assert refs.eligible("ref-x") == (False, "memorisation probe: unprobed")
    (tmp_path / "none.json").write_text('{"ref-x|model-a": {"flag": false}, "ref-x|model-b": {"flag": true}}')
    assert refs.eligible("ref-x") == (False, "memorisation probe: flagged")
    (tmp_path / "none.json").write_text('{"ref-x|model-a": {"flag": false}, "ref-x|model-b": {"flag": false}}')
    assert refs.eligible("ref-x")[0]
