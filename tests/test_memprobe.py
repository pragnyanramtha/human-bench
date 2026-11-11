import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "memprobe", Path(__file__).parents[1] / "scripts" / "refcards" / "memprobe.py"
)
memprobe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memprobe)

REST = (
    "the kettle had been broken for a week before anyone admitted to dropping it and by then the whole office "
    "had taken sides over whether tea made in the microwave counted as tea at all which it does not and I will "
    "not be taking questions on that point from anyone who has ever reheated a cup"
)


def test_a_stock_phrase_alone_is_not_a_flag():
    other = ("we finally agreed on the paint for the hallway after three weekends of arguing and I have to admit "
             "the green looks better than I expected though I will deny saying so if my brother asks about it")
    r = memprobe.compare(other + " I look forward to hearing from you soon", "anyway I look forward to hearing from you soon. " + REST)
    assert r["longest_run"] == 8 and not r["flag"]


def test_verbatim_and_near_verbatim_continuations_are_flagged():
    assert memprobe.compare(REST, REST)["flag"]
    words = REST.split()
    near = " ".join("zzz" if i % 8 == 7 else w for i, w in enumerate(words))  # every eighth word changed
    r = memprobe.compare(near, REST)
    assert r["longest_run"] < 8 and r["five_gram_share"] >= 0.25 and r["flag"]


def test_an_unrelated_continuation_is_clean():
    r = memprobe.compare("the committee met on Tuesday to discuss the drainage budget and agreed nothing", REST)
    assert not r["flag"] and r["runs"] == []
