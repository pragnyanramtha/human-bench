"""Clean CPU timing per detector on a fixed 16-text subset (run when the machine is otherwise idle).

  uv run python timing.py <detector> [<detector> ...]     # each in its own process via run_timing.sh
Writes private/judge_eval/results/timing/<detector>.json:
wall and CPU seconds per 500 words, load time, peak RSS.
"""

from __future__ import annotations

import json
import resource
import sys
import time

import torch

from common import PAIRS_PATH, RESULTS_DIR, fold, read_jsonl, word_count
from detectors import CLASSIFIERS, Classifier, LMStats


def subset() -> list[str]:
    pairs = read_jsonl(PAIRS_PATH)
    texts = [fold(p[s]) for p in pairs for s in ("human", "ai")]
    mid = [t for t in texts if 400 <= word_count(t) <= 700]
    step = max(1, len(mid) // 16)
    return mid[::step][:16]


def main() -> None:
    name = sys.argv[1]
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    torch.set_num_threads(threads)
    t0 = time.time()
    det = LMStats(name[3:]) if name.startswith("lm:") else Classifier(CLASSIFIERS[name])
    load_s = time.time() - t0
    texts = subset()
    det.score(texts[0])  # warm-up
    w0, c0 = time.perf_counter(), time.process_time()
    for t in texts:
        det.score(t)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    words = sum(word_count(t) for t in texts)
    out = dict(detector=name, threads=threads, n_texts=len(texts), words=words, load_s=round(load_s, 1),
               wall_per_500w=round(wall / words * 500, 3), cpu_per_500w=round(cpu / words * 500, 3),
               peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    d = RESULTS_DIR / "timing"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name.replace(':', '_')}__t{threads}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
