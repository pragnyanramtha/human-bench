"""Deterministic verifiers (DESIGN.md section 4.2).

Each check returns a VerifierResult. A text that fails any verifier counts as
"caught" for that sample. Thresholds that depend on human writing (noise
rates) are passed in, calibrated from the human references.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from collections import Counter
from decimal import Decimal

from humanbench.axes import Channel
from humanbench.scoring.normalize import word_count


@dataclass(frozen=True)
class VerifierResult:
    name: str
    passed: bool
    detail: str = ""


# --- length -----------------------------------------------------------------


MIN_WORDS, MAX_WORDS = 250, 4000  # the shortest and longest range a card may ask for


def check_length(text: str, band: tuple[int, int], tolerance: float = 0.10) -> VerifierResult:
    """The piece is within the range the request gave. When the request says "roughly lo-hi words" the
    tolerance applies at both ends, including below 250 and above 4,000: a request for roughly 250 words is
    met by 240. (Until 2026-11-11 the tolerance stopped at 250 and 4,000, which failed answers that did what
    was asked.) A card with an exact range passes tolerance 0."""
    lo, hi = band
    n = word_count(text)
    lo_ok = lo * (1 - tolerance)
    hi_ok = hi * (1 + tolerance)
    ok = lo_ok <= n <= hi_ok
    return VerifierResult("length", ok, f"{n} words, allowed {lo_ok:.0f}-{hi_ok:.0f} (band {lo}-{hi} ±{tolerance:.0%})")


# --- format -----------------------------------------------------------------

# What each channel tolerates. Headings and tables are wrong almost everywhere a
# person writes casually; bold is wrong where the platform doesn't render it.
FORMAT_POLICY: dict[Channel, dict[str, bool]] = {
    Channel.EMAIL: {"headings": False, "tables": False, "bold": True, "bullets": True},
    Channel.CHAT: {"headings": False, "tables": False, "bold": True, "bullets": True},
    Channel.FORUM: {"headings": False, "tables": False, "bold": True, "bullets": True},
    Channel.SOCIAL: {"headings": False, "tables": False, "bold": False, "bullets": True},
    Channel.REVIEW: {"headings": False, "tables": False, "bold": False, "bullets": True},
    Channel.BLOG: {"headings": True, "tables": True, "bold": True, "bullets": True},
    Channel.WORK_DOCS: {"headings": True, "tables": True, "bold": True, "bullets": True},
    Channel.PERSONAL: {"headings": False, "tables": False, "bold": False, "bullets": False},
}

_FORMAT_PATTERNS: dict[str, re.Pattern[str]] = {
    "headings": re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE),
    "tables": re.compile(r"^\s*\|.*\|\s*$\n^\s*\|?\s*:?-{3,}", re.MULTILINE),
    # Emphasis wraps text that starts and ends with a non-space and has a letter in it, so star
    # ratings ("**★★★★☆**", "**1/2 out of **") and censored blanks ("__ and __") are not bold.
    "bold": re.compile(r"(?<![*\w])\*\*(?=\S)(?=[^*\n]*[A-Za-z])[^*\n]+?(?<=\S)\*\*(?!\*)"
                       r"|(?<![_\w])__(?=\S)(?=[^_\n]*[A-Za-z])[^_\n]+?(?<=\S)__(?![_\w])"),
    "bullets": re.compile(r"^\s*([-*•]|\d+[.)])\s+\S", re.MULTILINE),
}


def check_format(text: str, channel: Channel, overrides: dict[str, bool] | None = None) -> VerifierResult:
    policy = {**FORMAT_POLICY[channel], **(overrides or {})}
    found = [name for name, pat in _FORMAT_PATTERNS.items() if not policy[name] and pat.search(text)]
    return VerifierResult("format", not found, f"not allowed in {channel}: {', '.join(found)}" if found else "")


# --- wrapper text -----------------------------------------------------------

_WRAPPER_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        # An assistant introducing its draft. "Sure, I can cover Thursday." is a person.
        r"\A\s*(?:sure|certainly|absolutely|of course)\b[!,.]?\s*(?:here(?:'s| is| are)|i'd be (?:happy|glad)|i can help)\b",
        r"\A\s*here(?:'s| is| are) (?:a|an|the|your) (?:draft|version|revised|rewrite|rewritten|updated|suggested|possible|sample|polished)\b",
        r"\A\s*here(?:'s| is| are) (?:a|an|the|your) (?:\w+ )?(?:reply|response|message|email|post|review|letter|toast|speech|caption) (?:you|that you|for you)\b",
        r"\[(?:your|recipient'?s?|sender'?s?|insert|company|first|last|full)?\s*name\]",
        r"\[insert [^\]]+\]",
        r"<\s*your name\s*>",
        r"\bas an ai\b",
        # Offers to edit the draft, addressed to the user, not to the recipient.
        # Only as the closing line: mid-text, these are normal things to say to a recipient.
        r"\blet me know if you(?:'d| would) like (?:me to (?:make )?)?(?:any )?(?:changes|edits|adjustments|tweaks|a (?:shorter|longer|different|more \w+) version)[!.]?\s*\Z",
        r"\bfeel free to (?:adjust|tweak|customi[sz]e|modify|edit) (?:this|the|it)(?: (?:draft|wording|tone|template))?(?: (?:as (?:needed|you see fit)|to (?:fit|suit) your \w+))?[!.]?\s*\Z",
        r"(?:would you like me to|want me to|shall i) (?:make it|adjust|change|shorten|lengthen|rewrite|revise|tweak|soften|tone)\b[^?\n]*\?\s*\Z",
    )
]


def check_wrapper(text: str) -> VerifierResult:
    hits = [m.group(0).strip() for p in _WRAPPER_PATTERNS if (m := p.search(text))]
    return VerifierResult("wrapper", not hits, "; ".join(hits))


# --- required content -------------------------------------------------------

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


_FRACTIONS = {"½": ".5", "¼": ".25", "¾": ".75"}
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_NUM = "(?:" + "|".join([*_UNITS, *_TENS, *_SCALES]) + ")"
_NUMWORD = re.compile(
    rf"\b{_NUM}(?:[\s-]+(?:and[\s-]+)?{_NUM})*(?:[\s-]+and an? half)?\b",
    re.IGNORECASE,
)


def _words_to_number(phrase: str) -> str | None:
    tokens = [t for t in re.split(r"[\s-]+", phrase.lower()) if t and t != "and"]
    half = tokens[-2:] in (["a", "half"], ["an", "half"])
    if half:
        tokens = tokens[:-2]
    # Year readings use two two-digit groups; digit lists are separate numbers.
    for cut in range(1, len(tokens)):
        left = tokens[:cut]
        right = tokens[cut:]
        if len(left) <= 2 and len(right) <= 2 and not any(t in _SCALES for t in tokens):
            a = sum(_UNITS.get(t, _TENS.get(t, 0)) for t in left)
            b = sum(_UNITS.get(t, _TENS.get(t, 0)) for t in right)
            if a in (19, 20) and 10 <= b <= 99:
                return str(a * 100 + b)
    if len(tokens) > 1 and all(t in _UNITS and _UNITS[t] < 10 for t in tokens):
        return " ".join(str(_UNITS[t]) for t in tokens)
    total = current = 0
    seen = False
    for t in tokens:
        if t in _UNITS:
            current += _UNITS[t]
        elif t in _TENS:
            current += _TENS[t]
        elif t in _SCALES:
            current = max(current, 1) * _SCALES[t]
            if _SCALES[t] >= 1000:
                total, current = total + current, 0
        else:
            return None
        seen = True
    if not seen:
        return None
    value = total + current
    return f"{value}.5" if half else str(value)


_MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
_MONTH_RE = "(?:" + "|".join(sorted({*(_MONTHS), *(m[:3] for m in _MONTHS), "sept"}, key=len, reverse=True)) + r")\.?"
_DAY_RE = r"(\d{1,2})(?:st|nd|rd|th)?"
_DATE_MD = re.compile(rf"\b({_MONTH_RE})\s+{_DAY_RE}\b(?!\s*(?:am|pm|:))", re.IGNORECASE)
_DATE_DM = re.compile(rf"\b{_DAY_RE}\s+(?:of\s+)?({_MONTH_RE})(?![a-z])", re.IGNORECASE)


def _month(name: str) -> int | None:
    key = name.lower().rstrip(".")
    if key == "sept":
        return 9
    for full, i in _MONTHS.items():
        if full == key or full[:3] == key:
            return i
    return None


_ORD_UNITS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
              "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
              "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
              "twentieth": 20, "thirtieth": 30}
_ORD_WORD = r"(?:(?:twenty|thirty)[\s-])?(?:" + "|".join(sorted(_ORD_UNITS, key=len, reverse=True)) + ")"


def _ordinal_value(word: str) -> int | None:
    parts = re.split(r"[\s-]+", word.lower())
    if len(parts) == 1:
        return _ORD_UNITS.get(parts[0])
    tens = {"twenty": 20, "thirty": 30}.get(parts[0])
    unit = _ORD_UNITS.get(parts[1])
    return tens + unit if tens and unit and unit < 10 else None


def _ordinal_dates_to_digits(s: str) -> str:
    """"the nineteenth of November" -> "19 November"; "December the twelfth" -> "December 12".
    Ordinal words are only converted next to a month name, so "a second time" stays."""
    month = _MONTH_RE
    def before(m: re.Match) -> str:
        n = _ordinal_value(m.group(1))
        return f"{n} {m.group(2)}" if n else m.group(0)

    def after(m: re.Match) -> str:
        n = _ordinal_value(m.group(2))
        return f"{m.group(1)} {n}" if n else m.group(0)

    s = re.sub(rf"\b(?:the\s+)?({_ORD_WORD})\s+(?:of\s+)?({month})(?![a-z])", before, s, flags=re.IGNORECASE)
    s = re.sub(rf"\b({month})\s+(?:the\s+)?({_ORD_WORD})\b", after, s, flags=re.IGNORECASE)
    s = re.sub(rf"\b({month})\s+the\s+(\d{{1,2}}(?:st|nd|rd|th)?)\b", r"\1 \2", s, flags=re.IGNORECASE)
    s = re.sub(rf"\bthe\s+(\d{{1,2}}(?:st|nd|rd|th)?)\s+of\s+({month})(?![a-z])", r"\1 \2", s, flags=re.IGNORECASE)
    return s


def canon_dates(s: str) -> str:
    """Write calendar dates one way, so "January 15th", "15th January", "Jan 15", "15 Jan",
    "the fifteenth of January" and "January the 15th" all become "date0115". A year after
    the date is left as its own token."""
    s = _ordinal_dates_to_digits(s)
    def md(m: re.Match) -> str:
        mon, day = _month(m.group(1)), int(m.group(2))
        return f" date{mon:02d}{day:02d} " if mon and 1 <= day <= 31 else m.group(0)

    def dm(m: re.Match) -> str:
        mon, day = _month(m.group(2)), int(m.group(1))
        return f" date{mon:02d}{day:02d} " if mon and 1 <= day <= 31 else m.group(0)

    return _DATE_DM.sub(dm, _DATE_MD.sub(md, s))


_MAGNITUDES = {"k": 1000, "thousand": 1000, "m": 1000000, "million": 1000000,
               "bn": 1000000000, "billion": 1000000000}
_SCALED = re.compile(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*(billion|million|thousand|bn|[km])\b", re.I)


def expand_magnitudes(s: str) -> str:
    return _SCALED.sub(lambda m: format(Decimal(m[1]) * _MAGNITUDES[m[2].lower()], "f"), s)


def canon_numbers(s: str) -> str:
    """Write every number the same way, so "40k", "40,000", "40 000" and "forty thousand"
    match, as do "5.5", "5,5", "5½" and "five and a half"."""
    s = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", s, flags=re.I)
    s = re.sub(rf"\b({_ORD_WORD})\b", lambda m: str(_ordinal_value(m[1])), s, flags=re.I)
    for sym, dec in _FRACTIONS.items():
        s = s.replace(sym, dec)
    s = re.sub(r"(?<=[^\W\d_])(?=\d)", " ", s)  # "N10,000" -> "N 10,000", "R40" -> "R 40"

    s = _NUMWORD.sub(lambda m: (f" {n} " if (n := _words_to_number(m.group(0).strip())) else m.group(0)), s)
    # spoken decimals: "1 point 4" (after the word pass) -> "1.4"
    s = re.sub(r"(\d+)\s+point\s+((?:\d\s?)+)\b", lambda m: f"{m.group(1)}.{m.group(2).replace(' ', '')}", s,
               flags=re.IGNORECASE)
    prev = None
    while prev != s:  # thousands separators: 40,000 / 40 000 / 1,250,000
        prev = s
        s = re.sub(r"(\d)[, ](\d{3})(?!\d)", r"\1\2", s)
    s = re.sub(r"(\d),(\d{1,2})(?!\d)", r"\1.\2", s)  # decimal comma: 5,5 -> 5.5

    s = re.sub(r"(\d+(?:\.\d+)?)\s?k\b", lambda m: format(Decimal(m[1]) * 1000, "f"), s, flags=re.I)
    # "1271 pounds and 75 pence", "£12 and 50p", "3 dollars 20 cents" -> 1271.75 / 12.50 / 3.20
    s = re.sub(r"(\d+)\s*(?:pounds?|quid|dollars?|euros?)?\s*(?:and\s+)?(\d{1,2})\s*(?:pence|p|cents?)\b",
               lambda m: f"{m.group(1)}.{int(m.group(2)):02d}", s, flags=re.IGNORECASE)
    s = re.sub(r"\b\d+\.\d+\b", lambda m: format(Decimal(m[0]).normalize(), "f"), s)
    return s


def canon_times(s: str) -> str:
    s = re.sub(r"\bhalf past (one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
               lambda m: f"{_UNITS[m[1].lower()]}:30", s, flags=re.I)
    s = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve) (thirty|fifteen|forty five)\b",
               lambda m: f"{_UNITS[m[1].lower()]}:{ {'thirty':30,'fifteen':15,'forty five':45}[m[2].lower()]}", s, flags=re.I)
    s = re.sub(r"\b(\d{1,2})(?::([0-5]\d)|\.(30))(?:\s*[ap]m)?\b", lambda m: f" time{m[1]}x{m[2] or m[3]} ", s, flags=re.I)
    return s


def _canon(s: str, *, times: bool = True) -> str:
    s = canon_dates(s.replace("’", "'"))
    s = canon_numbers(canon_times(s) if times else s)
    s = re.sub(r"(\d)\.(\d)", "\\1ǂ\\2", s)  # keep decimals intact through punctuation stripping
    return _SPACE.sub(" ", _PUNCT.sub(" ", s.lower())).replace("ǂ", ".").strip()


def check_must_include(text: str, requirements: Sequence[str | Sequence[str]]) -> VerifierResult:
    """Each requirement is a phrase, or a list of alternatives where any one counts."""
    canon = f" {_canon(text)} "
    decimals = f" {_canon(text, times=False)} "
    missing: list[str] = []
    for req in requirements:
        options = [req] if isinstance(req, str) else list(req)
        if not any(f" {_canon(o)} " in canon or f" {_canon(o, times=False)} " in decimals for o in options):
            missing.append(" | ".join(options))
    return VerifierResult("must_include", not missing, f"missing: {missing}" if missing else "")


def check_must_not_include(text: str, phrases: Sequence[str]) -> VerifierResult:
    canon = f" {_canon(text)} "
    present = [p for p in phrases if f" {_canon(p)} " in canon]
    return VerifierResult("must_not_include", not present, f"present: {present}" if present else "")


# --- surface noise and hidden characters ------------------------------------

_HIDDEN = re.compile(
    "[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\U000e0000-\U000e007f]"
)
_WORDLIKE = re.compile(r"\w+", re.UNICODE)


def _case_flips(word: str) -> int:
    letters = [c for c in word if c.isalpha()]
    return sum(1 for a, b in zip(letters, letters[1:]) if a.islower() != b.islower())


def _alternating_case(word: str) -> bool:
    """True for words like "aLtErNaTe"; brand names such as "JavaScript" have 3 flips."""
    return _case_flips(word) >= 5


def _scripts(word: str) -> set[str]:
    scripts = set()
    for ch in word:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if name.startswith(("GREEK ", "MATHEMATICAL ")):
                continue
            scripts.add(name.split(" ")[0] if name else "UNKNOWN")
    return scripts


@dataclass(frozen=True)
class NoiseStats:
    words: int
    hidden_chars: int
    mixed_script_words: int
    alternating_case_words: int

    @property
    def per_1k_words(self) -> dict[str, float]:
        scale = 1000 / max(self.words, 1)
        return {
            "mixed_script_words": self.mixed_script_words * scale,
            "alternating_case_words": self.alternating_case_words * scale,
        }


def _emoji_join(text: str, i: int) -> bool:
    def emoji(c: str) -> bool:
        return 0x1F000 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27FF
    left = text[:i].rstrip("\ufe0f")
    return bool(left and i + 1 < len(text) and emoji(left[-1]) and emoji(text[i + 1]))


def noise_stats(text: str) -> NoiseStats:
    words = _WORDLIKE.findall(text)
    mixed = sum(1 for w in words if len(s := _scripts(w)) > 1 and "LATIN" in s)
    return NoiseStats(
        words=len(words),
        hidden_chars=sum(not (m[0] == "\u200d" and _emoji_join(text, m.start())) for m in _HIDDEN.finditer(text)),
        mixed_script_words=mixed,
        alternating_case_words=sum(1 for w in words if _alternating_case(w)),
    )


def check_hidden_chars(text: str) -> VerifierResult:
    stats = noise_stats(text)
    bad = stats.hidden_chars + stats.mixed_script_words
    return VerifierResult(
        "hidden_chars",
        bad == 0,
        f"{stats.hidden_chars} invisible chars, {stats.mixed_script_words} mixed-script words" if bad else "",
    )


def check_noise(text: str, max_per_1k: dict[str, float] | None = None) -> VerifierResult:
    """Fail if any noise rate exceeds the human threshold (e.g. 95th percentile)."""
    max_per_1k = max_per_1k or {}
    words = _WORDLIKE.findall(text)
    degenerate = len(words) >= 250 and (
        max(Counter(w.lower() for w in words).values(), default=0) > len(words) * .30
        or sum(_alternating_case(w) for w in words) > len(words) * .30
    )
    rates = noise_stats(text).per_1k_words
    over = {k: round(v, 2) for k, v in rates.items() if k in max_per_1k and v > max_per_1k[k]}
    return VerifierResult("noise", not over and not degenerate, f"above human rate: {over}" if over else "degenerate text" if degenerate else "")


# --- text aimed at the judge -------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:note|message|instructions?) (?:to|for) (?:the |any )?(?:evaluator|judge|grader|reviewer|annotator|classifier|detector|ai)\b",
        r"\bignore (?:all |any )?(?:previous|prior|above|earlier) instructions\b",
        r"\b(?:this|the following|the above) (?:text|essay|message|post|email|piece|review) (?:was|is) (?:written|authored) by a (?:real )?(?:human|person)\b",
        r"\bnot (?:written|generated) by (?:an )?(?:ai|llm|language model)\b",
    )
]


def check_injection(text: str) -> VerifierResult:
    hits = [m.group(0) for p in _INJECTION_PATTERNS if (m := p.search(text))]
    return VerifierResult("judge_injection", not hits, "; ".join(hits))


# --- copying ------------------------------------------------------------------


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = [w.lower() for w in _WORDLIKE.findall(text)]
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def copied_share(text: str, sources: Sequence[str], n: int = 13) -> float:
    """Share of the text's words that sit inside an n-word run also found in a source."""
    words = [w.lower() for w in _WORDLIKE.findall(text)]
    if len(words) < n:
        return 0.0
    theirs: set[tuple[str, ...]] = set()
    for src in sources:
        theirs |= _ngrams(src, n)
    covered = [False] * len(words)
    for i in range(len(words) - n + 1):
        if tuple(words[i : i + n]) in theirs:
            covered[i : i + n] = [True] * n
    return sum(covered) / len(words)


def check_material_copy(text: str, material: Sequence[str], max_share: float = 0.25, n: int = 13) -> VerifierResult:
    """Quoting the task's own material (a log line, a ticket, the message being answered) is
    normal; reproducing a large part of it is not."""
    share = copied_share(text, material, n)
    ok = share <= max_share
    return VerifierResult("material_copy", ok, "" if ok else f"{share:.0%} of the text is copied from the task material (max {max_share:.0%})")


def check_copy(text: str, sources: Sequence[str], n: int = 13, exclude: Sequence[str] = ()) -> VerifierResult:
    """Fail if the text shares any run of `n` words with a reference or source text.

    Runs that also occur in `exclude` (the task's own material, which both sides may
    quote) don't count."""
    mine = _ngrams(text, n)
    for src in exclude:
        mine -= _ngrams(src, n)
    shared = sum(len(mine & _ngrams(src, n)) for src in sources)
    return VerifierResult("copy", shared == 0, f"{shared} shared {n}-word runs" if shared else "")


# --- run everything -------------------------------------------------------------


@dataclass
class VerifierReport:
    results: list[VerifierResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[VerifierResult]:
        return [r for r in self.results if not r.passed]


def run_verifiers(
    text: str,
    *,
    channel: Channel,
    band: tuple[int, int],
    must_include: Sequence[str | Sequence[str]] = (),
    must_not_include: Sequence[str] = (),
    format_overrides: dict[str, bool] | None = None,
    copy_sources: Sequence[str] = (),
    material: Sequence[str] = (),
    noise_thresholds: dict[str, float] | None = None,
    length_tolerance: float = 0.10,
) -> VerifierReport:
    """`copy_sources`: texts that must share no 13-word run (human references, public corpora).
    `material`: what the task gave the writer, which may be quoted but not reproduced."""
    results = [
        check_length(text, band, length_tolerance),
        check_format(text, channel, format_overrides),
        check_wrapper(text),
        check_must_include(text, must_include),
        check_must_not_include(text, must_not_include),
        check_hidden_chars(text),
        check_injection(text),
        check_copy(text, copy_sources, exclude=material),
        check_material_copy(text, material),
    ]
    if noise_thresholds is not None:
        results.append(check_noise(text, noise_thresholds))
    return VerifierReport(results)
