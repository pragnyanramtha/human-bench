"""Score every text in pairs.jsonl with one detector; one process per detector.

  uv run python score.py <detector-name> [--view fold|light] [--fp32] [--limit N]
  uv run python score.py list

Output (cached, resumable): private/judge_eval/scores/<detector>__<view>.jsonl
with one row per text: pair_id, side (human|ai), score(s), tokens, wall and CPU seconds.
A summary row with load time and peak RSS is written to <...>.meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import torch

from common import PAIRS_PATH, SCORES_DIR, fold, light, read_jsonl, word_count
from detectors import CLASSIFIERS, LM_PAIRS, Classifier, LMStats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("detector")
    ap.add_argument("--view", default="fold", choices=["fold", "light"])
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("JUDGE_THREADS", "2")))
    a = ap.parse_args()
    if a.detector == "list":
        print("\n".join(list(CLASSIFIERS) + [f"lm:{k}" for k in LM_PAIRS]))
        return
    torch.set_num_threads(a.threads)
    dtype = torch.float32 if a.fp32 else torch.bfloat16
    tag = f"{a.detector.replace(':', '_')}__{a.view}{'__fp32' if a.fp32 else ''}"
    out_path = SCORES_DIR / f"{tag}.jsonl"
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    done = {(r["pair_id"], r["side"]) for r in read_jsonl(out_path)}

    t0 = time.time()
    if a.detector.startswith("lm:"):
        det = LMStats(a.detector[3:], dtype=dtype)
    else:
        det = Classifier(CLASSIFIERS[a.detector], dtype=dtype)
    load_s = time.time() - t0

    view = fold if a.view == "fold" else light
    items = [(p["pair_id"], side, p[side]) for p in read_jsonl(PAIRS_PATH) for side in ("human", "ai")]
    items = [x for x in items if (x[0], x[1]) not in done]
    if a.limit:
        items = items[: a.limit]
    print(f"{tag}: {len(done)} cached, {len(items)} to score (load {load_s:.1f}s)", flush=True)
    with out_path.open("a") as f:
        for i, (pid, side, text) in enumerate(items, 1):
            t = view(text)
            w0, c0 = time.perf_counter(), time.process_time()
            s, ntok = det.score(t)
            row = dict(pair_id=pid, side=side, tokens=ntok, words=word_count(t),
                       wall_s=round(time.perf_counter() - w0, 3), cpu_s=round(time.process_time() - c0, 3))
            row.update(s if isinstance(s, dict) else {"score": s})
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(items):
                print(f"  {i}/{len(items)} last={row['wall_s']}s/{ntok}tok", flush=True)
    meta = dict(detector=a.detector, view=a.view, dtype=str(dtype), threads=a.threads, load_s=round(load_s, 1),
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                torch=torch.__version__, finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    prev = SCORES_DIR / f"{tag}.meta.json"
    if prev.exists():  # keep the max RSS across resumed runs
        old = json.loads(prev.read_text())
        meta["peak_rss_mb"] = max(meta["peak_rss_mb"], old.get("peak_rss_mb", 0)) if items else old["peak_rss_mb"]
    prev.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), file=sys.stderr)


if __name__ == "__main__":
    main()
