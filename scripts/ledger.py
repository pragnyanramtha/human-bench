"""Text-free scoring ledger: everything needed to recompute the results tables, and nothing else.

    uv run python scripts/ledger.py write --run-id RUN [--run-id RUN2 ...]   -> reports/ledger-v0.1.jsonl,
                                                                               reports/calibration-v0.1.json
    uv run python scripts/ledger.py tables                                    recompute "pass all gates" from them

One ledger row per model output: run, system-line track, split, task, model, sample, the hash of the scored
text, whether it passed the programmatic checks (and which checks failed, by name only), the LLM-judge
verdict, each detector's score, and for reference tasks each detector's score of the human reference.
The calibration file holds each detector's human-text scores and threshold. No task text, model output or
human text is included; the held-out test split appears by task id only.
Rows for human-reference tasks are written only for the tasks the results report uses (memorisation screen passed
for the whole launch panel, human reference scored and read as human by the headline detector) and only
for models in that panel: one rule, `reference_cards` in the report module, serves both.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "reports" / "ledger-v0.1.jsonl"
CALIB = REPO / "reports" / "calibration-v0.1.json"


def write(runs: list[str], caps: list[str] | None = None) -> None:
    """`caps`: "RUN[/MODEL-PREFIX]=K" entries, as in the results report: only samples 1..K are written."""
    from humanbench import refs
    from humanbench.harness.cards import SPLITS, fingerprint, load_tasks, score_output, split_dir
    from humanbench.harness.report_v01 import (DETECTORS, HEADLINE, PRIV, Judge, reference_cards, reference_judge_text,
                                               rows, sha)
    from humanbench.scoring import floor as floor_mod

    judge = Judge()
    tasks = {s: load_tasks([split_dir(s)]) for s in SPLITS}
    usable = reference_cards(tasks["ref"], judge)[0]  # the same rule as the results report
    panel = set(refs.launch_models())
    where = {t: s for s, d in tasks.items() for t in d}
    limits = [(k.partition("/")[0], k.partition("/")[2], int(v)) for k, v in (x.split("=") for x in caps or [])]

    def limit(run: str, model: str) -> int:
        return min([k for r, prefix, k in limits if r == run and model.startswith(prefix)], default=10**6)

    out = []
    for run in runs:
        floor = {(r["key"], r["sha"]): r for r in rows(PRIV / "trials" / run / "floor.jsonl")}
        tf = PRIV / "trials" / run / "track.txt"
        track = tf.read_text().strip() if tf.is_file() else "default"
        for p in sorted((PRIV / "trials" / run / "raw").glob("*/*/s*.json")):
            r = json.loads(p.read_text())
            split = where.get(r["task"])
            if r.get("infra_failure") or split is None or r["sample"] > limit(run, r["model"]):
                continue
            pkg = tasks[split][r["task"]]
            if r.get("card_fingerprint") != fingerprint(pkg):
                continue  # made from an older version of the card
            if split == "ref" and (r["task"] not in usable or r["model"] not in panel):
                continue  # card not usable, or model never screened for memorisation
            res = score_output(pkg, r["output"])
            text = res["scored_text"]
            row = floor.get((f"{r['model']}|{r['task']}|s{r['sample']}", sha(text)))
            want = floor_mod.prompt_id(floor_mod.build_prompt(pkg, res["authored_text"]))
            fl = row["floor"] if row and row.get("prompt_id") == want and row.get("floor") in (0.0, 1.0) else None
            ref = reference_judge_text(pkg) if split == "ref" else None
            out.append({
                "run": run, "track": track, "split": split, "task": r["task"], "model": r["model"], "sample": r["sample"],
                "sha": sha(text), "empty": not text.strip(), "scripted": int(res["deterministic"] == 1.0),
                "failed_checks": sorted({f.split(":")[1].strip() if f.count(":") else f for f in res["failures"]}),
                "floor": fl, "floor_failed": (row or {}).get("failed") if fl == 0.0 else None,
                "detectors": {d: judge.score(d, text) for d in DETECTORS},
                "reference": {d: judge.score(d, ref) for d in DETECTORS} if ref else None,
            })
    LEDGER.write_text("".join(json.dumps(r) + "\n" for r in out))
    cal = {}
    for d, (label, pool) in DETECTORS.items():
        hs = sorted(judge.scores[d][sha(h["text"])] for h in judge.humans[pool] if sha(h["text"]) in judge.scores[d])
        cal[d] = {"label": label, "headline": d == HEADLINE, "humans_in_pool": len(judge.humans[pool]),
                  "humans_scored": len(hs), "cutoff": judge.cut[d], "human_scores": hs}
    CALIB.write_text(json.dumps(cal) + "\n")
    print(f"{len(out)} outputs -> {LEDGER.name}; calibration for {len(cal)} detectors -> {CALIB.name}")


def leaderboard_lines(rows: list[dict], cal: dict) -> list[str]:
    """Rebuild the report's leaderboard from the text-free release evidence."""
    from humanbench.harness.report_v01 import HEADLINE, ci, score_table

    summary = {}
    for split in ("ref", "dev", "test"):
        mine = [r for r in rows if r["split"] == split]
        if not mine:
            continue
        summary[split] = {"_cards": len({r["task"] for r in mine})}
        for model in sorted({r["model"] for r in mine}):
            by_card, good, lost = defaultdict(list), defaultdict(list), defaultdict(int)
            incomplete = False
            for r in mine:
                if r["model"] != model:
                    continue
                ok, floor = r["scripted"], r["floor"]
                score = r["detectors"].get(HEADLINE)
                cutoff = cal[HEADLINE]["cutoff"]
                gate = 0.0 if r["empty"] else None if score is None or cutoff is None else float(score <= cutoff)
                if not ok or floor is not None:
                    good[r["task"]].append(float(ok and floor == 1.0))
                if not ok or floor == 0.0 or gate == 0.0:
                    value = 0.0
                elif floor is None or gate is None:
                    incomplete = True
                    continue
                else:
                    value = gate
                by_card[r["task"]].append(value)
                lost["failed checks" if not ok else "rejected by judge" if floor == 0.0 else
                     "flagged as AI" if gate == 0.0 else "passes"] += 1
            summary[split][model] = {"score": None if incomplete else ci(by_card),
                "by_card": dict(by_card), "good_by_card": dict(good), "lost": dict(lost)}
    # Use the same pooling, bootstrap and formatting as the report.
    lines = score_table(summary)
    start = next(i for i, line in enumerate(lines) if line.startswith("| model | HumanBench score"))
    end = next(i for i in range(start + 1, len(lines)) if not lines[i])
    return lines[start:end]


def tables() -> None:
    """Print the bootstrapped leaderboard (HumanBench score over all tasks, its public and held-out parts), then per-set gate rates.

    For the human-reference tasks the comparison with the human reference is printed too (the model's piece scores as
    more human than the real one; tie = half): it is reported beside the headline, not as the headline."""
    rows = [json.loads(x) for x in LEDGER.read_text().split("\n") if x.strip()]
    cal = json.loads(CALIB.read_text())
    print("\n".join(leaderboard_lines(rows, cal)))
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        for d, c in cal.items():
            s = r["detectors"].get(d)
            gates = {"reads as human": None if s is None or c["cutoff"] is None else float(s <= c["cutoff"])}
            if r["split"] == "ref":
                h = (r["reference"] or {}).get(d)
                gates["beats the real piece"] = None if s is None or h is None else 1.0 if s < h else 0.5 if s == h else 0.0
            for name, v in gates.items():
                if r["empty"]:
                    v = 0.0
                if not r["scripted"] or r["floor"] == 0.0 or v == 0.0:
                    p = 0.0
                elif r["floor"] is None or v is None:
                    p = None
                else:
                    p = v
                groups[(r["split"], name)][(r["model"], d)][r["task"]].append(p)
    for (split, name), g in sorted(groups.items()):
        print(f"\n{split}: programmatic checks x LLM judge x {name.replace('beats the real piece', 'beats the human reference')}")
        for (model, d), by_task in sorted(g.items()):
            vals = [x for v in by_task.values() for x in v]
            if any(x is None for x in vals):
                print(f"  {model:26s} {cal[d]['label']:20s} incomplete ({sum(x is None for x in vals)} unscored)")
            else:
                m = sum(sum(v) / len(v) for v in by_task.values()) / len(by_task)
                print(f"  {model:26s} {cal[d]['label']:20s} {100 * m:5.1f}%  ({len(vals)} outputs, {len(by_task)} tasks)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--run-id", action="append", required=True)
    w.add_argument("--max-sample", action="append", default=[], metavar="RUN[/MODEL-PREFIX]=K",
                   help="only samples 1..K of that run, or of its models whose names start with the prefix")
    sub.add_parser("tables")
    a = ap.parse_args()
    write(a.run_id, a.max_sample) if a.cmd == "write" else tables()


if __name__ == "__main__":
    main()
