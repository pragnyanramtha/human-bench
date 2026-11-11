"""Write the texts the judge should score, as {"key", "text"} rows under private/judge_cache/in/.

    uv run python scripts/judge_export.py humans            # the human calibration pool
    uv run python scripts/judge_export.py oracles           # every card's oracle (dev, test, ref)
    uv run python scripts/judge_export.py run --run-id R    # a trial run's scored outputs

Every text goes through the same preparation as a model output (TaskPackage.scoring_text /
normalize), so humans, oracles and model outputs reach the judge in the same form.
Then: cd experiments/judge_cpu && uv run python judge_run.py --in ../../private/judge_cache/in/<name>.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from humanbench.envs import OracleRun
from humanbench.scoring.normalize import normalize
from humanbench.taskpkg import load_package

REPO = Path(__file__).resolve().parents[1]
PRIV = REPO / "private"
OUT = PRIV / "judge_cache" / "in"
SPLITS = {"dev": REPO / "tasks" / "dev", "test": PRIV / "tasks" / "test", "ref": REPO / "tasks" / "ref"}


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def humans() -> list[dict]:
    rows = [{"key": f"eval|{h['id']}", "channel": h["channel"], "source": h.get("source"), "text": normalize(h["text"])}
            for h in _rows(PRIV / "judge_eval" / "human.jsonl")]
    for f in sorted((PRIV / "corpus").glob("*.jsonl")):
        rows += [{"key": f"corpus|{r['id']}", "channel": r["channel"], "source": r.get("source"),
                  "text": normalize(r["text"])} for r in _rows(f)]
    return rows


def oracles() -> list[dict]:
    rows = []
    for split, root in SPLITS.items():
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            if not d.is_dir() or d.name.startswith("."):
                continue
            pkg, od = load_package(d, require_oracle=False), PRIV / "oracle" / d.name
            if (od / "actions.json").is_file():
                text = pkg.scoring_text(run=OracleRun.model_validate_json((od / "actions.json").read_text()))
            elif (od / "draft.md").is_file():
                text = pkg.scoring_text(text=(od / "draft.md").read_text())
            else:
                continue
            rows.append({"key": f"{split}|{d.name}", "channel": pkg.card.channel.value, "text": text})
    return rows


def run(run_id: str) -> list[dict]:
    rows = []
    for p in sorted((PRIV / "trials" / run_id / "raw").glob("*/*/s*.json")):
        r = json.loads(p.read_text())
        if not r.get("infra_failure") and (r.get("scored_text") or "").strip():
            rows.append({"key": f"{r['model']}|{r['task']}|s{r['sample']}", "text": r["scored_text"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["humans", "oracles", "run"])
    ap.add_argument("--run-id")
    a = ap.parse_args()
    rows = humans() if a.what == "humans" else oracles() if a.what == "oracles" else run(a.run_id)
    name = a.what if a.what != "run" else f"run__{a.run_id}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows))
    print(f"{len(rows)} texts -> {OUT / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
