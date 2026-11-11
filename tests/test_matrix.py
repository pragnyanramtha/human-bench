from collections import Counter

import pytest

from humanbench import axes
from humanbench.axes import Band
from humanbench.checks import check_slots
from humanbench.matrix import sample_matrix
from humanbench.schema import load_personas


@pytest.fixture(scope="module")
def personas():
    return load_personas()


@pytest.fixture(scope="module")
def matrix(personas):
    slots, _ = sample_matrix(seed=1, personas=personas)
    return slots


def test_matrix_passes_every_rule(matrix, personas):
    assert check_slots(matrix, personas) == []


def test_matrix_is_deterministic(matrix, personas):
    again, _ = sample_matrix(seed=1, personas=personas)
    assert [s.model_dump() for s in again] == [s.model_dump() for s in matrix]


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_other_seeds_also_valid(seed, personas):
    slots, _ = sample_matrix(seed=seed, personas=personas)
    assert check_slots(slots, personas) == []


def test_length_rule(matrix):
    over_400 = sum(1 for s in matrix if s.band is not Band.SHORT)
    assert over_400 / len(matrix) >= 0.5
    assert all(s.words[0] >= 250 for s in matrix)


def test_split_sizes(matrix):
    split = Counter(s.split for s in matrix)
    assert split == {"dev": 80, "test": 80}


def test_checks_catch_a_repeated_persona_in_channel(matrix, personas):
    broken = [s.model_copy() for s in matrix]
    first, second = [s for s in broken if s.channel is axes.Channel.EMAIL][:2]
    broken[broken.index(second)] = second.model_copy(update={"persona_id": first.persona_id})
    errors = check_slots(broken, personas)
    assert any("twice in email" in e for e in errors)


def test_promotion_keeps_existing_dev_slots(personas):
    import dataclasses
    from humanbench.matrix import promote_to_dev
    old_rule = 5
    slots, _ = sample_matrix(seed=1, personas=personas)
    # start from a 5-per-channel dev split, as v1 originally had
    base = []
    for ch in axes.Channel:
        ch_slots = [s for s in slots if s.channel is ch]
        for i, s in enumerate(ch_slots):
            base.append(s.model_copy(update={"split": "dev" if i < old_rule else "test"}))
    promoted = promote_to_dev(base, per_channel=10)
    before = {s.id for s in base if s.split == "dev"}
    after = {s.id for s in promoted if s.split == "dev"}
    assert before <= after and len(after) == 80
    assert check_slots(promoted, personas) == []


def test_split_files_and_public_report(tmp_path):
    import json
    from humanbench.matrix import load_slots, main

    public = tmp_path / 'slots.json'
    private = tmp_path / 'private' / 'slots-test.json'
    report = tmp_path / 'coverage.md'
    main(['--seed', '2', '--out', str(public), '--private-out', str(private), '--report', str(report)])
    public_rows = load_slots(public)
    all_rows = load_slots(public, private)
    assert len(public_rows) == 80 and all(s['split'] == 'dev' for s in public_rows)
    assert len(all_rows) == 160
    assert len(load_slots(public, tmp_path / 'absent.json')) == 80
    assert 'seed' not in json.loads(public.read_text())
    private_ids = {s['id'] for s in all_rows if s['split'] == 'test'}
    assert not any(sid in report.read_text() for sid in private_ids)
    assert 'Private slots: 80' in report.read_text()
    assert 'seed' not in report.read_text()
    main(['--input', str(public), '--private-out', str(tmp_path / 'absent.json'), '--report', str(report)])
    assert 'Private slots: not available' in report.read_text()
