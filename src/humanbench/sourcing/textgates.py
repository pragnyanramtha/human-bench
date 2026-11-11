"""Small deterministic text gates: English check, dates, PII scrubbing.

No models and no network, so the corpus pipeline is reproducible and the tests
run offline.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

# --- English ---------------------------------------------------------------------

# High-frequency English function words. On 250+ words of English prose, 30-50%
# of tokens are in this list; other Latin-script languages share only a few
# (a, in, so, no, was) and land well under 15%. Code and logs also score low,
# which is intended: the gate is "English prose".
_EN_FUNCTION_WORDS = frozenset(
    """
    a about after all also am an and any are as at be because been before being both
    but by can could did do does doing don't down during each even ever for from get
    got had has have having he her here hers him his how i i'm if in into is isn't it
    it's its just like me more most much my no not now of off on once one only or other
    our out over own really same she should so some such than that that's the their
    them then there these they this those through to too under until up us very was
    wasn't we were what when where which while who whom why will with would you your
    """.split()
)
_TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)


@dataclass(frozen=True)
class EnglishScore:
    function_word_share: float
    latin_share: float
    tokens: int

    def is_english(self, min_function_share: float = 0.2, min_latin_share: float = 0.9, min_tokens: int = 50) -> bool:
        """Too-short texts can't be judged; they pass here and fail the length gate instead."""
        if self.tokens < min_tokens:
            return True
        return self.function_word_share >= min_function_share and self.latin_share >= min_latin_share


def english_score(text: str) -> EnglishScore:
    tokens = [t.lower().replace("’", "'") for t in _TOKEN.findall(text)]
    letters = [c for c in text if c.isalpha()]
    latin = sum(1 for c in letters if c.isascii() or "À" <= c <= "ɏ")
    return EnglishScore(
        function_word_share=sum(t in _EN_FUNCTION_WORDS for t in tokens) / max(len(tokens), 1),
        latin_share=latin / max(len(letters), 1),
        tokens=len(tokens),
    )


# --- dates --------------------------------------------------------------------------


def parse_date(value: Any, fmt: str | None) -> dt.date | None:
    """Parse a date field. Returns None for missing or unparseable values."""
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.UTC)
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        if fmt == "unix" or isinstance(value, int | float):
            return dt.datetime.fromtimestamp(float(value), tz=dt.UTC).date()
        s = str(value).strip()
        if fmt in (None, "iso"):
            s = s.replace("Z", "+00:00")
            parsed = dt.datetime.fromisoformat(s)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(dt.UTC)
            return parsed.date()
        return dt.datetime.strptime(s, fmt).date()
    except (ValueError, OverflowError, OSError):
        return None


# --- PII -------------------------------------------------------------------------------

PII_TYPES = ("url", "email", "handle", "phone")
PLACEHOLDER = {"url": "[URL]", "email": "[EMAIL]", "handle": "[HANDLE]", "phone": "[PHONE]"}

_URL = re.compile(
    r"(?i)\b(?:(?:https?|ftp)://|www\.)[^\s<>\"'\]\[)]+"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|uk|de|info|biz|ly|me)/[^\s<>\"'\]\[)]*"
)
_EMAIL = re.compile(r"(?i)(?<![\w.+-])[a-z0-9._%+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}\b")
_HANDLE = re.compile(r"(?<![\w@./])@[A-Za-z0-9_](?:[A-Za-z0-9_.]{0,28}[A-Za-z0-9_])?")
_REDDIT_USER = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_-]{3,20}\b")
_PHONE_PATTERNS = (
    # North American: (713) 853-1234, 713-853-1234, 713.853.1234, 1-800-555-1234
    (re.compile(r"(?<![\w+])(?:1[\s.-])?\(?\d{3}\)?[\s.-]?\d{3}[.-]\d{4}(?![\w-])"), 10, 11),
    # international: +44 20 7946 0958, +91-98765-43210
    (re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.-]?\(?\d{1,5}\)?){2,5}(?![\w-])"), 8, 15),
    # UK-style national: 07700 900123, 020 7946 0958, 0161-496-0000
    (re.compile(r"(?<![\w+])0\d{2,4}[\s-]\d{3,6}(?:[\s-]\d{3,4})?(?![\w-])"), 10, 11),
)
_TRAILING = ".,;:!?'\""


def _sub_url(text: str, counts: Counter[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        url = m.group(0)
        stripped = url.rstrip(_TRAILING)
        counts["url"] += 1
        return PLACEHOLDER["url"] + url[len(stripped) :]

    return _URL.sub(repl, text)


def _sub_phone(text: str, counts: Counter[str]) -> str:
    for pat, lo, hi in _PHONE_PATTERNS:
        def repl(m: re.Match[str], lo: int = lo, hi: int = hi) -> str:
            digits = sum(c.isdigit() for c in m.group(0))
            if lo <= digits <= hi:
                counts["phone"] += 1
                return PLACEHOLDER["phone"]
            return m.group(0)

        text = pat.sub(repl, text)
    return text


def scrub_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace URLs, emails, @handles and phone numbers with typed placeholders.

    Idempotent: scrubbing scrubbed text changes nothing. Names are NOT removed.
    """
    counts: Counter[str] = Counter()
    text = _sub_url(text, counts)

    def count(kind: str) -> Any:
        def repl(_m: re.Match[str]) -> str:
            counts[kind] += 1
            return PLACEHOLDER[kind]

        return repl

    text = _EMAIL.sub(count("email"), text)
    text = _HANDLE.sub(count("handle"), text)
    text = _REDDIT_USER.sub(count("handle"), text)
    text = _sub_phone(text, counts)
    return text, {k: counts[k] for k in PII_TYPES}
