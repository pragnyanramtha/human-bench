"""Mechanical gates for one reference card, wherever it lives.

    uv run python scripts/refcards/check_card.py private/tasks/ref_pending/ref-blog-21 [more card dirs ...]

Prints one line per card and exits non-zero if any card fails. A card passes when:
  - it loads (schema);
  - the real human piece (private/oracle/<id>/draft.md) scores 1.0 on the card's own scripted checks;
  - every do-nothing answer (empty, wrapper, echo of the request) scores 0.0;
  - what the model would be sent does not leak the piece (humanbench.refs.leakage: no shared run of 8+ words,
    at most 8% of its 5-word sequences, short persona fields).
The human piece is never changed to make a card pass: fix the card, or drop it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def check(card_dir: Path) -> list[str]:
    from humanbench import refs
    from humanbench.taskpkg import PackageError, load_package, nop_outputs

    try:
        pkg = load_package(card_dir, require_oracle=False)
    except PackageError as e:
        return [f"does not load: {e}"]
    ref = REPO / "private" / "oracle" / pkg.card.id / "draft.md"
    if not ref.is_file():
        return [f"no human piece at {ref}"]
    text = ref.read_text()
    problems = list(refs.leakage(pkg, text))
    score, why = pkg.score(text=text)
    if score != 1.0:
        problems.append(f"the human piece scores {score}: {why}")
    for name, kwargs in nop_outputs(pkg):
        s, _ = pkg.score(**kwargs)
        if s != 0.0:
            problems.append(f"do-nothing answer '{name}' scores {s}")
    return problems


def main() -> None:
    bad = 0
    for arg in sys.argv[1:]:
        problems = check(Path(arg))
        bad += bool(problems)
        print(f"{Path(arg).name}: " + ("ok" if not problems else "FAIL: " + "; ".join(problems)))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
