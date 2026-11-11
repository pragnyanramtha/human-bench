"""Memorisation eligibility of cards built from real writing.

eligible() checks only that every launch model was probed and none reproduced
its human text. An unprobed card is not eligible. The launch-model set is frozen
in private/refcards/launch_models.txt (one model per line).
Audit certification is separate: python scripts/audit/audit_cards.py --certify
checks that the latest clean audit is bound to unchanged card content.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PRIV = REPO / "private"
PROBE = PRIV / "refcards" / "memprobe.json"
LAUNCH = PRIV / "refcards" / "launch_models.txt"


def launch_models() -> list[str]:
    return [x.strip() for x in LAUNCH.read_text().split("\n") if x.strip()] if LAUNCH.is_file() else []


def probe_status(slot: str) -> str:
    """'clean', 'flagged' or 'unprobed' (some launch model has no completed probe)."""
    probes = json.loads(PROBE.read_text()) if PROBE.is_file() else {}
    models = launch_models()
    if not models or any(f"{slot}|{m}" not in probes for m in models):
        return "unprobed"
    return "flagged" if any(probes[f"{slot}|{m}"]["flag"] for m in models) else "clean"


def eligible(slot: str) -> tuple[bool, str]:
    status = probe_status(slot)
    return status == "clean", f"memorisation probe: {status}"


# ----------------------------------------------------------------- leakage of the human text into the card

MAX_FIVE_GRAM_SHARE = 0.08  # measured on the v0.1 cards: median 0.003, maximum 0.053
MAX_FIELD_WORDS = 15


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9']+", text.lower())


def leakage(pkg, reference: str) -> list[str]:
    """Problems with how much of the human text shows through what the model is sent.

    Checks everything rendered to the model (prompt, notes, persona line), not only the notes:
    no exact run of 8+ words, at most 8% of its 5-word sequences found in the human text (catches
    lightly altered prose that leaves no long exact run), and short persona fields (a long field could
    carry the text)."""
    from humanbench.scoring.verifiers import copied_share

    problems = []
    shown = pkg.model_prompt()
    if copied_share(shown, [reference], n=8) > 0:
        problems.append("what the model is sent shares a run of 8+ words with the human text")
    a, b = _tokens(shown), _tokens(reference)
    five = {tuple(b[i:i + 5]) for i in range(len(b) - 4)}
    grams = [tuple(a[i:i + 5]) for i in range(len(a) - 4)]
    share = sum(g in five for g in grams) / len(grams) if grams else 0.0
    if share > MAX_FIVE_GRAM_SHARE:
        problems.append(f"{share:.0%} of the 5-word sequences the model is sent occur in the human text "
                        f"(limit {MAX_FIVE_GRAM_SHARE:.0%})")
    for name, value in pkg.card.persona.model_dump().items():
        if isinstance(value, str) and len(value.split()) > MAX_FIELD_WORDS:
            problems.append(f"persona.{name} is {len(value.split())} words (limit {MAX_FIELD_WORDS})")
    return problems
