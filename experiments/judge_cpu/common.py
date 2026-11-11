"""Shared paths and helpers for the judge_cpu experiment."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "private" / "judge_eval"
RAW_DIR = EVAL_DIR / "raw"
HUMAN_PATH = EVAL_DIR / "human.jsonl"
BRIEFS_PATH = EVAL_DIR / "briefs.jsonl"
GEN_DIR = EVAL_DIR / "generations"
PAIRS_PATH = EVAL_DIR / "pairs.jsonl"
SCORES_DIR = EVAL_DIR / "scores"
RESULTS_DIR = EVAL_DIR / "results"

WORD_RE = re.compile(r"\b[\w'’-]+\b")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": " - ", "―": " - ", "−": "-",
    "…": "...", " ": " ", " ": " ", " ": " ", "​": "",
}


def fold(text: str) -> str:
    """Primary evaluation view.

    The human sources were scraped and re-encoded by different pipelines (some
    flatten paragraphs, some are ASCII-only), so typography and line breaks say
    more about the dataset than about the writer. To stop detectors winning on
    those artefacts, both sides are folded the same way: curly quotes become
    straight, dashes become hyphens, markdown emphasis markers are kept, and every
    whitespace run (including paragraph breaks) becomes one space.
    """
    text = unicodedata.normalize("NFKC", text)
    for k, v in _FOLD.items():
        text = text.replace(k, v)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def light(text: str) -> str:
    """Secondary 'raw' view: keep typography and paragraphs, tidy spaces only."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def prose_view(text: str) -> str:
    """Versioned prose-only judge view; preserve visible wording, discard Markdown syntax.

    Run before fold so block markers still have line boundaries. Fenced and
    inline code retain their contents; link labels and image alt text survive.
    This is normalization, not a full Markdown renderer or an HTML sanitizer.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^ {0,3}(`{3,}|~{3,})[^\n]*$", "", text, flags=re.M)
    text = re.sub(r"^ {0,3}(?:[-*_] *){3,}$", "", text, flags=re.M)
    text = re.sub(r"^ {0,3}(?:=+|-+)\s*$", "", text, flags=re.M)
    text = re.sub(r"^ {0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*$", r"\1", text, flags=re.M)
    text = re.sub(r"^\s{0,3}(?:>\s*)+", "", text, flags=re.M)
    text = re.sub(r"^ {0,3}(?:[-+*]|\d+[.)])\s+(?:\[[ xX]\]\s*)?", "", text, flags=re.M)
    text = re.sub(r"^ {0,3}\[[^\]\n]+\]:\s+\S+.*$", "", text, flags=re.M)
    text = re.sub(r"!?\[([^\]\n]*)\]\([^\n]*?\)", r"\1", text)
    text = re.sub(r"!?\[([^\]\n]+)\]\[[^\]\n]*\]", r"\1", text)
    text = re.sub(r"<((?:https?://|mailto:)[^>]+)>", r"\1", text)
    text = re.sub(r"(`+)(.+?)\1", r"\2", text)
    for marker in ("**", "__", "~~", "*", "_"):
        pat = (
            r"(?<!\w)"
            + re.escape(marker)
            + r"(?=\S)(.+?)(?<=\S)"
            + re.escape(marker)
            + r"(?!\w)"
        )
        text = re.sub(pat, r"\1", text)
    text = re.sub(r"\\([\\`*_{}\[\]()#+.!>~-])", r"\1", text)
    return fold(text)
