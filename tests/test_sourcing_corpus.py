import datetime as dt
import json
import random

import pytest

from humanbench.axes import Band, Channel
from humanbench.sourcing.corpus import ResumeError, assign_band, build_manifest, run_source
from humanbench.sourcing.dedup import Deduper
from humanbench.sourcing.registry import Registry, Source
from humanbench.sourcing.streams import InMemoryStream, get_path

FUNC = "the and to of a in that it is was for on with as at by this but not or from have they we you he she".split()
SYLL = "ka lo mi ren tu sha vel dor pi nax ber lin qua so fen tor gal mu".split()


def make_text(seed: int, n_words: int = 300) -> str:
    """Deterministic English-looking prose with mostly unique content words."""
    rng = random.Random(seed)
    words: list[str] = []
    while len(words) < n_words:
        if rng.random() < 0.5:
            words.append(rng.choice(FUNC))
        else:
            words.append("".join(rng.choice(SYLL) for _ in range(rng.randint(2, 3))))
    sentences = [" ".join(words[i : i + 12]).capitalize() + "." for i in range(0, n_words, 12)]
    return "\n\n".join(" ".join(sentences[i : i + 4]) for i in range(0, len(sentences), 4))


def make_source(**overrides) -> Source:
    base = dict(
        id="toy", channel="review", repo="org/toy", revision="b" * 40, data_files=["*.parquet"], split="train",
        text_field="text", id_field="rid", date_field="date", date_format="iso", licence="cc-by-4.0",
        licence_status="clear", licence_notes="", allowed_uses=["judge_train"],
        filters=[{"field": "meta.keep", "op": "equals", "value": True}], meta_fields={"stars": "meta.stars"},
        verified={"checked": dt.date(2026, 11, 9), "method": "test"},
    )
    base.update(overrides)
    return Source(**base)


def registry_with(source: Source) -> Registry:
    return Registry(
        default_quotas={b: 100 for b in Band},
        no_source={c: "none in test" for c in Channel if c != source.channel},
        sources=[source],
    )


def row(i: int, text: str, date: str | None = "2010-06-01", keep: bool = True) -> dict:
    return {"rid": f"r{i}", "text": text, "date": date, "meta": {"keep": keep, "stars": i % 5}}


SPANISH = ("Volví a la tienda el martes porque la tetera había dejado de funcionar otra vez y no esperaba "
           "mucho de nadie. ") * 20
GOOD = make_text(1, 300)
NEAR = GOOD.replace("Th", "Xh", 1).rsplit(" ", 3)[0] + " de la fin."
LONG_URL = "https://example.com/" + "/".join("abcdefghij")


GATE_ROWS = [
    row(0, make_text(2), keep=False),                      # source_filter
    row(1, "   "),                                          # empty_text
    row(2, SPANISH),                                        # not_english
    row(3, make_text(3), date=None),                        # date_unknown
    row(4, make_text(4), date="2023-05-01"),                # date_after_cutoff
    row(5, make_text(5, 100)),                              # below_min_words
    row(6, make_text(6, 4200)),                             # above_max_words
    row(7, GOOD),                                           # accepted (short)
    row(8, GOOD),                                           # exact_duplicate
    row(9, NEAR),                                           # near_duplicate
    row(10, make_text(10) + " " + " ".join(f"see https://x.io/{i}" for i in range(30))),  # pii_dense
    row(11, make_text(11, 240) + f" {LONG_URL} and {LONG_URL}"),                           # below_min_after_scrub
    row(12, make_text(12, 900)),                            # band_unused_for_channel (review has no long)
    row(13, make_text(13, 300)),                            # quota_full (short quota is 1)
    row(14, make_text(14, 500)),                            # accepted (medium)
]
QUOTAS = {Band.SHORT: 1, Band.MEDIUM: 5, Band.LONG: 0, Band.EXTENDED: 0}


def test_assign_band_boundaries():
    assert assign_band(249) is None
    assert assign_band(250) is Band.SHORT
    assert assign_band(400) is Band.MEDIUM
    assert assign_band(2000) is Band.EXTENDED
    assert assign_band(4000) is Band.EXTENDED
    assert assign_band(4001) is None


def test_every_gate_rejects_with_its_reason(tmp_path):
    state = run_source(make_source(), InMemoryStream(GATE_ROWS), tmp_path, quotas=QUOTAS, log=lambda m: None)
    assert state.rejected == {
        "source_filter": 1, "empty_text": 1, "not_english": 1, "date_unknown": 1, "date_after_cutoff": 1,
        "below_min_words": 1, "above_max_words": 1, "exact_duplicate": 1, "near_duplicate": 1, "pii_dense": 1,
        "below_min_after_scrub": 1, "band_unused_for_channel": 1, "quota_full": 1,
    }
    assert state.records_written == 2 and state.exhausted
    assert state.quota_full_by_band == {"short": 1, "long": 1}
    recs = [json.loads(line) for line in (tmp_path / "toy.jsonl").read_text().splitlines()]
    assert [r["original_id"] for r in recs] == ["r7", "r14"]
    r = recs[0]
    for key in ("source", "original_id", "date", "licence", "channel", "band", "word_count", "hash"):
        assert r[key] is not None
    assert r["channel"] == "review" and r["band"] == "short" and r["date"] == "2010-06-01"
    assert r["allowed_uses"] == ["judge_train"] and r["meta"] == {"stars": 2}
    assert r["hash"].startswith("sha256:")


def test_pre_cutoff_by_construction_accepts_missing_dates(tmp_path):
    src = make_source(date_field=None, pre_cutoff_by_construction=True, construction_note="2011 release")
    state = run_source(src, InMemoryStream([row(1, make_text(1), date=None)]), tmp_path,
                       quotas=QUOTAS, log=lambda m: None)
    assert state.records_written == 1
    rec = json.loads((tmp_path / "toy.jsonl").read_text())
    assert rec["date"] is None and rec["date_basis"] == "by_construction"


def test_pii_is_scrubbed_in_output(tmp_path):
    text = make_text(20) + " Write to sam.lee@example.org or ring 713-555-0199."
    run_source(make_source(), InMemoryStream([row(1, text)]), tmp_path, quotas=QUOTAS, log=lambda m: None)
    rec = json.loads((tmp_path / "toy.jsonl").read_text())
    assert "[EMAIL]" in rec["text"] and "[PHONE]" in rec["text"] and "example.org" not in rec["text"]
    assert rec["pii"]["email"] == 1 and rec["pii"]["phone"] == 1


MANY = [row(i, make_text(100 + i, 260 + 7 * i)) for i in range(40)]
BIG_QUOTAS = {Band.SHORT: 100, Band.MEDIUM: 100, Band.LONG: 0, Band.EXTENDED: 0}


def _run(path, limit, **kw):
    return run_source(make_source(), InMemoryStream(MANY, group_size=3), path, quotas=BIG_QUOTAS,
                      limit=limit, checkpoint_every=4, log=lambda m: None, **kw)


def test_deterministic_and_resume_equals_fresh(tmp_path):
    _run(tmp_path / "a", 25)
    _run(tmp_path / "b", 25)
    _run(tmp_path / "c", 10)
    st = _run(tmp_path / "c", 25)  # resume
    a = (tmp_path / "a" / "toy.jsonl").read_bytes()
    assert a == (tmp_path / "b" / "toy.jsonl").read_bytes() == (tmp_path / "c" / "toy.jsonl").read_bytes()
    assert st.records_written == 25 and st.stopped_because == "limit"
    reg = registry_with(make_source())
    assert build_manifest(tmp_path / "a", reg) == build_manifest(tmp_path / "c", reg)


def test_resume_truncates_records_written_after_the_last_checkpoint(tmp_path):
    _run(tmp_path / "fresh", 20)
    _run(tmp_path / "crash", 10)
    with (tmp_path / "crash" / "toy.jsonl").open("a") as f:
        f.write('{"partial": "line written after the checkpoint"}\n')
    _run(tmp_path / "crash", 20)
    assert (tmp_path / "crash" / "toy.jsonl").read_bytes() == (tmp_path / "fresh" / "toy.jsonl").read_bytes()


def test_resume_refuses_changed_config(tmp_path):
    _run(tmp_path, 5)
    with pytest.raises(ResumeError, match="restart"):
        run_source(make_source(), InMemoryStream(MANY), tmp_path, quotas={**BIG_QUOTAS, Band.SHORT: 3},
                   log=lambda m: None)
    st = run_source(make_source(), InMemoryStream(MANY, group_size=3), tmp_path,
                    quotas={**BIG_QUOTAS, Band.SHORT: 3}, restart=True, log=lambda m: None)
    assert st.accepted_by_band["short"] == 3


def test_manifest_counts(tmp_path):
    run_source(make_source(), InMemoryStream(GATE_ROWS), tmp_path, quotas=QUOTAS, log=lambda m: None)
    m = build_manifest(tmp_path, registry_with(make_source()))
    s = m["sources"]["toy"]
    assert s["rows_scanned"] == len(GATE_ROWS) and s["accepted"] == 2
    assert sum(s["rejected"].values()) == len(GATE_ROWS) - 2
    assert m["totals"]["by_channel_band"] == {"review/medium": 1, "review/short": 1}
    assert "Never a test reference" in m["use_note"]


def test_dedup_exact_near_and_distinct():
    d = Deduper()
    reason, _, sig, ek = d.check(GOOD)
    assert reason is None
    d.add("a", sig, ek)
    assert d.check(GOOD)[0] == "exact_duplicate"
    assert d.check(GOOD.upper())[0] == "exact_duplicate"  # canonical form ignores case
    assert d.check(NEAR)[0] == "near_duplicate"
    assert d.check(make_text(999))[0] is None


def test_get_path_and_stream_cursor():
    assert get_path({"a": {"b": [10, 20]}}, "a.b.1") == 20
    assert get_path({"a": None}, "a.b") is None
    s = InMemoryStream([{"x": i} for i in range(7)], group_size=3)
    allrows = list(s.rows())
    cur, _ = allrows[4]
    resumed = list(s.rows((cur[0], cur[1], cur[2] + 1)))
    assert [r["x"] for _, r in resumed] == [5, 6]
