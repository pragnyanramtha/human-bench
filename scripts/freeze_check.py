"""Whole-split checks before a split is frozen (DESIGN.md 2.4, checks 2-4).

    uv run python scripts/freeze_check.py --split dev

Per-card gates already ran in the pipeline. This looks across the whole split:
  - every matrix slot of the split has an accepted card, and nothing else does
  - every card still passes the current per-card rules (with its private oracle)
  - the 5% most alike prompt pairs (TF-IDF cosine), to rewrite
  - prompts sharing their first four words
  - coverage: channel x mode, tone, audience, band, persona use
  - provenance: model-drafted context is a stated v0.1 limitation

Private reference solutions in private/oracle/ are required to run these checks in full.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import sys
from collections import Counter
from pathlib import Path

from humanbench.matrix import load_slots
from humanbench.cardcheck import check_card, oracle_report
from humanbench.similarity import most_similar_pairs, shared_openings
from humanbench.taskpkg import PackageError, load_package

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    print("Full validation requires private reference solutions in private/oracle/.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    root = REPO / "tasks" / "dev" if args.split == "dev" else REPO / "private" / "tasks" / args.split
    slots = {s["id"]: s for s in load_slots(REPO / "matrix/slots.json", REPO / "private/matrix/slots-test.json") if s["split"] == args.split}
    cards = {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")} if root.is_dir() else set()
    problems: list[str] = []
    missing = sorted(set(slots) - cards)
    extra = sorted(cards - set(slots))
    print(f"{len(cards)} cards for {len(slots)} {args.split} slots; missing {len(missing)}: {missing}")
    if extra:
        problems.append(f"cards with no {args.split} slot: {extra}")

    pkgs = {}
    for sid in sorted(cards & set(slots)):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp) / sid
            shutil.copytree(root / sid, t)
            (t / "oracle").mkdir(exist_ok=True)
            for f in ("draft.md", "actions.json"):
                src = REPO / "private/oracle" / sid / f
                if src.is_file():
                    shutil.copy2(src, t / "oracle" / f)
            try:
                pkg = load_package(t)
            except PackageError as e:
                problems.append(f"{sid}: does not load: {e.problems}")
                continue
            errs = check_card(pkg, slots[sid]) + oracle_report(pkg)
            problems += [f"{sid}: {e}" for e in errs]
            pkgs[sid] = load_package(root / sid, require_oracle=False)

    items = [(sid, p.card.prompt) for sid, p in pkgs.items()]
    pairs = most_similar_pairs(items, top_fraction=0.05)
    print("\nMost alike prompt pairs (top 5%):")
    for a, b, sim in pairs[:15]:
        print(f"  {sim:.2f}  {a} ~ {b}")
    for opening, ids in shared_openings(items).items():
        problems.append(f"prompts share the opening {opening!r}: {ids}")

    def table(name, key):
        c = Counter(key(p) for p in pkgs.values())
        print(f"\n{name}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))

    table("channel/mode", lambda p: f"{p.card.channel.value}/{p.card.mode.value}")
    table("tone", lambda p: p.card.tone.value)
    table("band", lambda p: p.card.band.value)
    table("audience", lambda p: p.card.audience.value)
    persona = Counter(p.card.persona_id for p in pkgs.values())
    print(f"\npersonas used: {len(persona)}; most used: {persona.most_common(3)}")
    drafted = [sid for sid, p in pkgs.items()
               if (p.thread and p.thread.provenance == "model_draft") or (p.voice and p.voice.provenance == "model_draft")]
    print(f"\nv0.1 information: {len(drafted)} cards have model-drafted threads or voice samples; a stated v0.1 limitation. "
          f"Commissioned human replacements are planned work: {sorted(drafted)}")

    print(f"\n{len(problems)} structural problems")
    for p in problems:
        print(f"  - {p}")
    ready = not problems and not missing
    print("\nREADY TO FREEZE" if ready else
          f"\nNOT READY TO FREEZE: {len(problems)} structural problems, {len(missing)} missing cards, "
          f"{len(drafted)} cards with model-drafted context")

    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
