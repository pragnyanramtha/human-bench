import datetime as dt

import pytest
from pydantic import ValidationError

from humanbench.axes import Band, Channel
from humanbench.sourcing.registry import Source, channel_bands, load_registry


def _src(**overrides):
    base = dict(
        id="t", channel="review", repo="org/ds", revision="a" * 40, data_files=["*.parquet"], split="train",
        text_field="text", pre_cutoff_by_construction=True, construction_note="released 2011",
        licence="cc-by-4.0", licence_status="clear", licence_notes="", allowed_uses=["judge_train"],
        verified={"checked": dt.date(2026, 11, 9), "method": "test"},
    )
    base.update(overrides)
    return Source(**base)


def test_shipped_registry_loads_and_explains_gaps():
    reg = load_registry()
    assert len(reg.sources) >= 6
    covered = {s.channel for s in reg.sources}
    assert covered | set(reg.no_source) == set(Channel)
    for s in reg.sources:
        assert "test_ref" not in s.allowed_uses
        assert s.cutoff == dt.date(2022, 11, 30)


def test_archival_text_can_never_be_a_test_ref():
    with pytest.raises(ValidationError, match="never be a test reference"):
        _src(allowed_uses=["judge_train", "test_ref"])


def test_dev_ref_needs_a_clear_licence():
    with pytest.raises(ValidationError, match="clear licence"):
        _src(allowed_uses=["dev_ref"], licence_status="questionable")
    assert _src(allowed_uses=["dev_ref"]).allowed_uses == ["dev_ref"]


def test_undated_source_must_be_pre_cutoff_by_construction():
    with pytest.raises(ValidationError, match="by construction"):
        _src(pre_cutoff_by_construction=False, construction_note="")


def test_unknown_cleaner_rejected():
    with pytest.raises(ValidationError, match="unknown cleaners"):
        _src(cleaners=["nope"])


def test_default_quotas_follow_the_channel_bands():
    assert channel_bands(Channel.REVIEW) == {Band.SHORT, Band.MEDIUM}
    defaults = {b: 10 for b in Band}
    assert _src().band_quotas(defaults) == {Band.SHORT: 10, Band.MEDIUM: 10, Band.LONG: 0, Band.EXTENDED: 0}
    assert _src(quotas={"long": 3}).band_quotas(defaults)[Band.LONG] == 3
