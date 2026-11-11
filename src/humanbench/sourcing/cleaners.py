"""Source-specific text cleaners, run before the gates.

These undo packaging artifacts of a particular dataset (HTML, escaped
newlines, tokenization, quoted mail). General collection artifacts are left to
``humanbench.scoring.normalize``, which runs as a gate on every source.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from html.parser import HTMLParser

# --- HTML -----------------------------------------------------------------------

_BLOCK_TAGS = {"p", "div", "pre", "blockquote", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK_TAGS and not self._after_bullet():
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS and not self._after_bullet():
            self.parts.append("\n\n" if tag != "li" else "")

    def _after_bullet(self) -> bool:
        """True right after "- " was emitted for <li>, so <li><p>x</p></li> stays one bullet."""
        return bool(self.parts) and self.parts[-1] == "\n- "

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(text: str) -> str:
    """Strip tags, keep paragraph breaks, unescape entities."""
    if "<" not in text and "&" not in text:
        return text
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    out = html.unescape("".join(parser.parts))
    out = re.sub(r"[ \t]+\n", "\n", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


# --- escaped newlines (Yelp) ------------------------------------------------------


def unescape_newlines(text: str) -> str:
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace('\\"', '"')


# --- Reddit TL;DR (tldr-17) ------------------------------------------------------

_TLDR = re.compile(r"^\W{0,3}(?:tl\s*[;:,.]?\s*dr|tldr)\b", re.IGNORECASE)


def strip_tldr(text: str) -> str:
    """Remove the TL;DR every tldr-17 post was selected for.

    A TL;DR in the second half of the post is cut with everything after it;
    one near the top (a summary-first post) is removed on its own.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if _TLDR.match(line.strip()):
            if i >= len(lines) / 2:
                return "\n".join(lines[:i]).strip()
            return "\n".join(lines[:i] + lines[i + 1 :]).strip()
    # inline "... tl;dr: summary" at the end of the last line
    m = re.search(r"\b(?:tl\s*;\s*dr|tldr)\b.*\Z", text, re.IGNORECASE | re.DOTALL)
    if m and m.start() > len(text) / 2:
        return text[: m.start()].strip()
    return text


# --- tokenized WritingPrompts ------------------------------------------------------

_WP_SUBS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\s*<newline>\s*"), "\n"),
    (re.compile(r" ?\n ?"), "\n"),
    (re.compile(r"``\s*"), '"'),
    (re.compile(r"\s*''"), '"'),
    (re.compile(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", re.IGNORECASE), r"\1"),
    (re.compile(r" ([.,!?;:%)\]])"), r"\1"),
    (re.compile(r"([(\[$]) "), r"\1"),
    (re.compile(r" \.\.\.", re.UNICODE), "..."),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def wp_detokenize(text: str) -> str:
    """Undo the PTB-style tokenization of the Fan et al. WritingPrompts data."""
    for pat, rep in _WP_SUBS:
        text = pat.sub(rep, text)
    # straight double quotes: glue to the word they open or close
    text = re.sub(r'" ([^"\n]*?) "', r'"\1"', text)
    return text.strip()


# --- quoted / forwarded mail (Enron) ------------------------------------------------

_MAIL_CUT = re.compile(
    r"^\s*(?:-{3,}\s*(?:original message|forwarded by)|_{5,}|from:\s.+|to:\s.+|subject:\s.*|"
    r"\S.*\s+on\s+\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*[ap]m\b.*)",
    re.IGNORECASE,
)
_NOTES_STAMP = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[AP]M", re.IGNORECASE)


def _drop_blank_tail(lines: list[str]) -> None:
    while lines and not lines[-1].strip():
        lines.pop()


def enron_cut_quoted(text: str) -> str:
    """Keep only the newly written part of an Enron email.

    Cuts at the first forwarded/original-message marker or at an embedded
    header line that follows some new text (Outlook and Lotus Notes styles).
    """
    lines = text.replace("\r\n", "\n").split("\n")
    seen_text = False
    for i, line in enumerate(lines):
        lowered = line.lower()
        if _MAIL_CUT.match(line) and (seen_text or "forwarded by" in lowered or "original message" in lowered):
            kept = lines[:i]
            _drop_blank_tail(kept)
            # Lotus Notes puts "Sender Name" and "05/14/2001 04:30 PM" above the header
            if kept and _NOTES_STAMP.search(kept[-1]):
                kept.pop()
                _drop_blank_tail(kept)
                if kept and len(kept[-1].split()) <= 4:
                    kept.pop()
            return "\n".join(kept).strip()
        if line.strip():
            seen_text = True
    return text.strip()


_LIST_START = re.compile(r"^\s*(?:[-*•>]|\d+[.)]|[a-z][.)])\s")


def join_broken_lines(text: str) -> str:
    """Re-join lines broken mid-sentence by mail clients (Enron is wrapped at odd widths).

    A line is joined to the next when it doesn't end a sentence and the next
    line starts with a lowercase letter and isn't a list item.
    """
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        if (
            out
            and out[-1].strip()
            and line.strip()
            and line.lstrip()[0].islower()
            and not _LIST_START.match(line)
            and not out[-1].rstrip().endswith((".", "!", "?", ":", ";"))
        ):
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


CLEANERS: dict[str, Callable[[str], str]] = {
    "join_broken_lines": join_broken_lines,
    "html_to_text": html_to_text,
    "unescape_newlines": unescape_newlines,
    "strip_tldr": strip_tldr,
    "wp_detokenize": wp_detokenize,
    "enron_cut_quoted": enron_cut_quoted,
}


def apply_cleaners(text: str, names: list[str]) -> str:
    for name in names:
        text = CLEANERS[name](text)
    return text
