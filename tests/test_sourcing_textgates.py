import datetime as dt

from humanbench.sourcing.cleaners import (
    enron_cut_quoted,
    html_to_text,
    join_broken_lines,
    strip_tldr,
    unescape_newlines,
    wp_detokenize,
)
from humanbench.sourcing.textgates import english_score, parse_date, scrub_pii

ENGLISH = (
    "I went back to the shop on Tuesday because the kettle had stopped working again, and to be honest "
    "I was not expecting much. The woman at the counter was kind about it, but she said they could only "
    "swap it if I still had the box, which of course I had thrown out the week before. "
) * 3
SPANISH = (
    "Volví a la tienda el martes porque la tetera había dejado de funcionar otra vez, y para ser sincero "
    "no esperaba mucho. La mujer del mostrador fue amable, pero dijo que solo podían cambiarla si todavía "
    "tenía la caja, que por supuesto había tirado la semana anterior. "
) * 3
GERMAN = (
    "Ich bin am Dienstag wieder in den Laden gegangen, weil der Wasserkocher schon wieder kaputt war, und "
    "ehrlich gesagt habe ich nicht viel erwartet. Die Frau an der Kasse war freundlich, aber sie sagte, "
    "dass sie ihn nur umtauschen können, wenn ich die Verpackung noch habe. "
) * 3


def test_english_score_separates_languages():
    assert english_score(ENGLISH).is_english()
    assert not english_score(SPANISH).is_english()
    assert not english_score(GERMAN).is_english()
    assert not english_score("Это длинный текст на русском языке. " * 20).is_english()


def test_short_text_is_left_to_the_length_gate():
    assert english_score("FYI\n\nVince").is_english()


def test_parse_date_formats():
    assert parse_date("2001-05-14T23:39:00", "iso") == dt.date(2001, 5, 14)
    assert parse_date("2021-03-01T23:30:00Z", "iso") == dt.date(2021, 3, 1)
    assert parse_date(dt.datetime(2020, 1, 2, 3, 4), "iso") == dt.date(2020, 1, 2)
    assert parse_date("14,May,2004", "%d,%B,%Y") == dt.date(2004, 5, 14)
    assert parse_date("14,mayo,2004", "%d,%B,%Y") is None  # non-English month: unknown
    assert parse_date("2016/01/12", "%Y/%m/%d") == dt.date(2016, 1, 12)
    assert parse_date(0, "unix") == dt.date(1970, 1, 1)
    assert parse_date(None, "iso") is None
    assert parse_date("not a date", "iso") is None


def test_scrub_pii_types():
    text = (
        "Mail me at jane.doe@example.com or call (713) 853-1234, or +44 20 7946 0958, or 07700 900123. "
        "Docs at https://example.com/docs?x=1. and www.foo.org, thanks @janedoe and /u/some_user."
    )
    out, counts = scrub_pii(text)
    assert counts == {"url": 2, "email": 1, "handle": 2, "phone": 3}
    assert "[EMAIL]" in out and "[PHONE]" in out and "[URL]." in out and "[HANDLE]" in out
    assert "example" not in out and "853" not in out


def test_scrub_pii_is_idempotent_and_leaves_numbers_alone():
    text = "In 1999-2003 we paid $1,250.00 for 3 tickets; the score was 21-14. Email a@b.co now."
    once, counts = scrub_pii(text)
    twice, counts2 = scrub_pii(once)
    assert once == twice and sum(counts2.values()) == 0
    assert "1999-2003" in once and "$1,250.00" in once and "21-14" in once
    assert counts["email"] == 1


def test_html_to_text_keeps_paragraphs():
    out = html_to_text("<p>One &amp; two.</p><p>Three<br />four</p><ul><li>five</li></ul>")
    assert out == "One & two.\n\nThree\nfour\n\n- five"
    assert html_to_text("I liked it.<br /><br />A lot.") == "I liked it.\n\nA lot."


def test_unescape_newlines():
    assert unescape_newlines("Good.\\n\\nBad.") == "Good.\n\nBad."


def test_strip_tldr_end_and_start():
    body = "Long story here.\nMore story.\nEven more.\ntl;dr: it was fine"
    assert strip_tldr(body) == "Long story here.\nMore story.\nEven more."
    top = "TL;DR it was fine\nLong story here.\nMore story.\nEven more."
    assert strip_tldr(top) == "Long story here.\nMore story.\nEven more."
    assert strip_tldr("no summary at all") == "no summary at all"


def test_wp_detokenize():
    raw = "`` Hello , '' she said . <newline> <newline> I do n't know what 's wrong ( really ) ."
    assert wp_detokenize(raw) == "\"Hello,\" she said.\n\nI don't know what's wrong (really)."


def test_enron_cut_quoted():
    body = (
        "Sounds good, see you Thursday.\n\n"
        "-----Original Message-----\nFrom: Someone\nSent: Monday\n\nold text"
    )
    assert enron_cut_quoted(body) == "Sounds good, see you Thursday."
    notes = (
        "Please review.\n\n\n\n\tJohn Smith\n\t05/14/2001 04:30 PM\n\t\t To: Jane Doe/HOU/ECT@ECT\n\t\t cc:\n"
        "\t\t Subject: Re: deal\n\nold text"
    )
    assert enron_cut_quoted(notes) == "Please review."
    fwd = "---------------------- Forwarded by Jane Doe/HOU/ECT on 05/14/2001 ---------\nold"
    assert enron_cut_quoted(fwd) == ""


def test_join_broken_lines():
    text = "I think the key point is\nrecognition of risks.\n1. first\n- second\nNext line Starts upper"
    assert join_broken_lines(text) == (
        "I think the key point is recognition of risks.\n1. first\n- second\nNext line Starts upper"
    )
