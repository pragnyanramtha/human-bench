from humanbench.scoring.normalize import normalize, word_count


def test_strips_headers_quotes_and_signature():
    raw = (
        "From: a@example.com\nTo: b@example.com\n\n"
        "Hi Sam,\n\nThe valves are wrong again — third time.\n\n"
        "-- \nDave\n07700 900123\n\n"
        "On Tue, 3 Oct, Sam wrote:\n> old text\n"
    )
    out = normalize(raw)
    assert out == "Hi Sam,\n\nThe valves are wrong again — third time."


def test_keeps_style_signals_and_invisible_chars():
    raw = "**Big news** — we’re live!​\n\n- one\n- two"
    out = normalize(raw)
    assert "**Big news**" in out and "—" in out and "’" in out and "​" in out
    assert "- one\n- two" in out


def test_unwraps_hard_wrapped_paragraph():
    line = "word " * 13  # 65 chars
    raw = "\n".join([line.strip()] * 3 + ["end."])
    out = normalize(raw)
    assert "\n" not in out


def test_drops_mobile_footer():
    assert normalize("See you then.\n\nSent from my iPhone") == "See you then."


def test_word_count():
    assert word_count("It's a twenty-one-word... no, it isn't.") == 6
