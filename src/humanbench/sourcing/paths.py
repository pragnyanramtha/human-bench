"""Where sourcing output goes. Everything lives under ``private/``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_PRIVATE = Path("private")


def ensure_private(root: Path) -> Path:
    """Create the private dir with its own catch-all .gitignore.

    The repo .gitignore already ignores private/; the inner file keeps the data
    out of git even if the dir is moved or the outer rule is edited.
    """
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".gitignore"
    if not marker.exists():
        marker.write_text("# private data: never commit\n*\n", encoding="utf-8")
    return root


def corpus_dir(private: Path) -> Path:
    return ensure_private(private) / "corpus"


def refs_dir(private: Path) -> Path:
    return ensure_private(private) / "refs"


def dump_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
