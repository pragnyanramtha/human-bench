"""Shared card loading, trial layout and deterministic scoring."""
from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from humanbench.envs import OracleRun
from humanbench.taskpkg import AGENTIC, TaskPackage, load_package, task_specific
REPO = Path(__file__).resolve().parents[3]
TRIALS = REPO / "private" / "trials"
SYSTEM_LINE = "You are a helpful assistant."  # Default track
# Instructed track (headline, DESIGN.md 2.8): the model is told the goal, in one fixed, published
# line. Task cards never coach style; this line is the only place a model hears "sound human".
INSTRUCTED_LINE = (
    "You are a helpful assistant. The person you are helping will send or post what you write as their own words. "
    "Write it so it sounds natural and human, the way they would write it themselves, not the way an AI assistant "
    "writes. Still do everything they asked, in the form they asked for."
)
TRACKS = {"default": SYSTEM_LINE, "instructed": INSTRUCTED_LINE}

SPLITS = ("dev", "test", "ref")
_FENCE = re.compile(r"^```(?:json|markdown|md|text)?\s*\n(.*?)\n```\s*$", re.DOTALL)

@dataclass(frozen=True)
class Trial:
    sample: int
    task: str
    model: str

    def path(self, run_dir: Path) -> Path:
        return run_dir / "raw" / self.model / self.task / f"s{self.sample}.json"

    def claim_path(self, run_dir: Path) -> Path:
        return run_dir / "claims" / f"{self.model}__{self.task}__s{self.sample}"

def fingerprint(pkg: TaskPackage) -> str:
    """Changes when what the model sees changes. Changes to the hidden checks don't make an
    output stale; `rescore` re-applies the current checks."""
    return hashlib.sha256(pkg.model_prompt().encode()).hexdigest()[:16]

def split_dir(split: str) -> Path:
    return REPO / "private" / "tasks" / "test" if split == "test" else REPO / "tasks" / split

def load_tasks(task_dirs: list[Path]) -> dict[str, TaskPackage]:
    out = {}
    for d in task_dirs:
        for card in sorted(d.glob("*/card.yaml")):
            if not card.parent.name.startswith("."):
                pkg = load_package(card.parent, require_oracle=False)
                out[pkg.card.id] = pkg
    return out

def score_output(pkg: TaskPackage, text: str) -> dict:
    """The scripted checks on one raw model output, plus the two views of it used downstream:
    `scored_text` (what the detectors read) and `authored_text` (everything authored, for the floor).
    On a reference card the human reference is a copy source: reproducing it fails."""
    if not text.strip():
        return {"deterministic": 0.0, "failures": ["empty output"], "task_failures": [], "scored_text": "",
                "authored_text": ""}
    refs = pkg.reference_texts()
    stripped = _FENCE.match(text.strip())
    body = stripped.group(1) if stripped else text
    if pkg.card.mode in AGENTIC:
        try:
            actions = pkg.parse_agent_reply(text)
            # The tool budget can't be measured with simulated tools; count one call per write.
            run = OracleRun(actions=actions, tool_calls=max(1, len(actions)))
            score, why = pkg.score(run=run, reference_texts=refs)
            scored_text, authored = pkg.scoring_text(run=run), pkg.authored_text(run=run)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            score, why, scored_text, authored = 0.0, [f"agent:unparseable: no valid writes ({str(exc)[:200]})"], "", ""
    else:
        score, why = pkg.score(text=text, reference_texts=refs)
        scored_text, authored = pkg.scoring_text(text=text), pkg.authored_text(text=text)
    return {"deterministic": score, "failures": why, "task_failures": task_specific(why), "scored_text": scored_text,
            "authored_text": authored}

def _done(path: Path, fp: str) -> bool:
    """A finished, current record: not an infrastructure failure and made from this card version."""
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return not r.get("infra_failure") and r.get("card_fingerprint") == fp

def approx_tokens(text: str) -> int:
    """API-equivalent prompt size: about 4 characters per token for English."""
    return max(1, round(len(text) / 4))

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
