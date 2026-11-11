"""Normalize texts before judging (DESIGN.md section 4.1).

The same function runs on the model's text and the human's. It removes
collection artifacts (mail headers, quoted replies, signatures, hard wraps)
and keeps real style signals: em dashes, curly quotes, markdown, sign-offs.
It never removes invisible characters; the verifiers need to see those.
"""

from __future__ import annotations

import re
import unicodedata

_HEADER = re.compile(r"^(from|sent|to|cc|bcc|date|reply-to)\s*:", re.IGNORECASE)
_QUOTE_START = re.compile(
    r"^(-{2,}\s*original message\s*-{2,}|on .{3,200} wrote:\s*$|-{2,}\s*forwarded message\s*-{2,})",
    re.IGNORECASE,
)
_MOBILE_FOOTER = re.compile(r"^\s*sent from my (iphone|ipad|android|phone|mobile|galaxy)[^\n]*$", re.IGNORECASE)
_SIG_DELIMITER = "-- "
_BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+")


def _strip_header_block(lines: list[str]) -> list[str]:
    i = 0
    while i < len(lines) and _HEADER.match(lines[i]):
        i += 1
    if i and i < len(lines) and not lines[i].strip():
        i += 1
    return lines[i:]


def _strip_quoted(lines: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        if _QUOTE_START.match(line.strip()):
            rest = [v for v in lines[i + 1:] if v.strip()]
            if not rest or all(v.lstrip().startswith(">") for v in rest):
                return lines[:i]
    return lines


def _strip_signature(lines: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        tail = lines[i + 1:]
        while tail and not tail[-1].strip():
            tail = tail[:-1]
        if line in ("--", "-- ") and len(tail) <= 6 and all(
            v.strip() and len(v) <= 80 for v in tail
        ):
            return lines[:i]
        if _MOBILE_FOOTER.match(line) and not any(v.strip() for v in tail):
            return lines[:i]
    return lines


def _unwrap_paragraph(par: list[str], min_len: int, max_len: int) -> list[str]:
    """Join a paragraph that was hard-wrapped at a fixed width."""
    if len(par) < 3 or any(_BULLET.match(line) for line in par):
        return par
    if all(min_len <= len(line) <= max_len for line in par[:-1]):
        return [" ".join(line.strip() for line in par)]
    return par


def normalize(text: str, unwrap_width: tuple[int, int] = (60, 80)) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    lines = _strip_header_block(lines)
    lines = _strip_quoted(lines)
    lines = _strip_signature(lines)

    paragraphs: list[list[str]] = [[]]
    for line in lines:
        if line.strip():
            paragraphs[-1].append(line)
        elif paragraphs[-1]:
            paragraphs.append([])
    joined = [
        "\n".join(_unwrap_paragraph(p, *unwrap_width)) for p in paragraphs if p
    ]
    return "\n\n".join(joined).strip()


_WORD = re.compile(r"\w+(?:['’-]\w+)*")


def word_count(text: str) -> int:
    return len(_WORD.findall(text))
