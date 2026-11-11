"""Texts that still need a headline-detector score, bundled for a rented GPU, and the scores merged back.

    uv run python scripts/gpu_bundle.py export --run-id R [--run-id R2 ...]   -> private/gpu/texts.jsonl
    # copy texts.jsonl and experiments/judge_cpu/gpu_score.py to the GPU machine, then there:
    #   python gpu_score.py --in texts.jsonl --out scores.jsonl
    #   python judge_run.py --in texts.jsonl --device cuda --cache fast/     (the three smaller detectors)
    # copy scores.jsonl, scores.manifest.json and fast/ back into private/gpu/, then:
    uv run python scripts/gpu_bundle.py merge [--fast-dir private/gpu/fast]   -> private/judge_cache/<detector>.jsonl

The bundle holds every text the results report will look up and the cache does not have yet: each run's
outputs, prepared against the card as it is now (the same text the report hashes), the human calibration
pool and the references. Rows are {"sha", "text"}; nothing else leaves this machine.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GPU = REPO / "private" / "gpu"


def export(runs: list[str], everything: bool = False) -> None:
    from humanbench.harness.cards import SPLITS, fingerprint, load_tasks, score_output, split_dir
    from humanbench.harness.report_v01 import CACHE, DETECTORS, HEADLINE, PRIV, Judge, reference_judge_text, rows, sha

    have = {r["sha"] for r in rows(CACHE / f"{HEADLINE}.jsonl")}
    tasks = {t: p for s in SPLITS for t, p in load_tasks([split_dir(s)]).items()}
    texts = [h["text"] for h in Judge().humans[DETECTORS[HEADLINE][1]]]
    texts += [t for t in (reference_judge_text(p) for p in tasks.values()) if t]
    stale = 0
    for run in runs:
        for p in sorted((PRIV / "trials" / run / "raw").glob("*/*/s*.json")):
            r = json.loads(p.read_text())
            pkg = tasks.get(r["task"])
            if r.get("infra_failure") or pkg is None:
                continue
            if r.get("card_fingerprint") != fingerprint(pkg):
                stale += 1
                continue
            texts.append(score_output(pkg, r["output"])["scored_text"])
    todo = {sha(t): t for t in texts if t.strip() and (everything or sha(t) not in have)}
    GPU.mkdir(parents=True, exist_ok=True)
    for old in (GPU / "scores.jsonl", GPU / "scores.manifest.json"):  # a new bundle starts a new score file
        if old.is_file() and not everything:
            old.rename(old.with_name(f"{old.stem}.before-{len(list(GPU.glob(old.stem + '.before-*')))}{old.suffix}"))
    (GPU / ("texts-all.jsonl" if everything else "texts.jsonl")).write_text("".join(json.dumps({"sha": k, "text": v}) + "\n" for k, v in sorted(todo.items())))
    print(f"{len(texts)} texts looked up, {len(todo)} "
          + ("in all -> texts-all.jsonl" if everything else f"without a {HEADLINE} score -> {GPU / 'texts.jsonl'}")
          + (f" ({stale} stale outputs left out)" if stale else ""))


def merge(scores: Path, fast_dir: Path | None) -> None:
    """Add new scores to the caches. A text that already has a score keeps it: nothing is overwritten."""
    from humanbench.harness.report_v01 import CACHE, DETECTORS, HEADLINE, rows

    def add(cache: Path, source: Path) -> None:
        have = {r["sha"] for r in rows(cache)}
        new = [r for r in rows(source) if r["sha"] not in have]
        with cache.open("a") as f:
            for r in new:
                f.write(json.dumps({"sha": r["sha"], "score": r["score"]}) + "\n")
        print(f"{len(new)} new scores -> {cache.name} ({len(have) + len(new)} in all)")

    add(CACHE / f"{HEADLINE}.jsonl", scores)
    man = scores.with_suffix(".manifest.json")
    if man.is_file():
        n = len(list(CACHE.glob(f"{HEADLINE}.manifest*.json")))
        shutil.copy(man, CACHE / f"{HEADLINE}.manifest-{n + 1}.json")
    if fast_dir:  # the three smaller detectors, scored with judge_run.py --device cuda --cache <dir> on the GPU machine
        for d in DETECTORS:
            if d != HEADLINE and (fast_dir / f"{d}.jsonl").is_file():
                add(CACHE / f"{d}.jsonl", fast_dir / f"{d}.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--run-id", action="append", required=True)
    e.add_argument("--all", action="store_true", help="every text, scored or not (to rescore a detector from scratch)")
    m = sub.add_parser("merge")
    m.add_argument("--scores", type=Path, default=GPU / "scores.jsonl", help="gpu_score.py output")
    m.add_argument("--fast-dir", type=Path, help="directory holding the smaller detectors' <detector>.jsonl files")
    a = ap.parse_args()
    export(a.run_id, a.all) if a.cmd == "export" else merge(a.scores, a.fast_dir)


if __name__ == "__main__":
    main()
