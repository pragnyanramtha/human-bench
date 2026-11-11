"""Score texts with the three open detectors used as HumanBench judges, with one shared cache.

    uv run python judge_run.py --in ../../private/judge_cache/in/humans.jsonl --threads 2

Input rows: {"key": ..., "text": ...} (or {"sha", "text"}, as scripts/gpu_bundle.py writes them). Output (same name under private/judge_cache/out/):
{"key", "sha", "scores": {detector: score}}; higher = more AI. `sha` is the first 16 hex digits of
sha256(text), the same hash the trials report uses for scored_text, so scores are joined by content.

Detectors (pinned revisions in detectors.py and editlens.py):
  vanguard-modernbert-large, tropa-mini-deberta-large: off-the-shelf classifiers, scored on the
      folded view (common.fold: typography and whitespace normalised, both sides alike);
  editlens-roberta-large: Pangram's open-weights EditLens model (CC BY-NC-SA 4.0), scored with its
      own preprocessing; it reads the first 512 tokens.
  editlens-llama-3.2-3b: the larger EditLens model (a LoRA adapter on Llama-3.2-3B), the headline detector.
      It is scored as released (bfloat16) on a GPU by gpu_score.py, which writes to the same cache; it
      reads the first 1,024 tokens. A 4-bit CPU variant (editlens-llama-3.2-3b-nf4, about 8 s per 500
      words) can be run here with --detectors, but its scores differ slightly and are not used.
The cache (private/judge_cache/<detector>.jsonl) is keyed by text hash, so a text is never scored twice.
With --device cuda the three classifiers run on a GPU. bfloat16 results differ slightly between devices and
between CPU types (up to about 0.1 logit on the same text), so every text in one cache, the human
calibration pool included, must be scored on the same device: use --cache to keep them apart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from common import ROOT, fold

CACHE = ROOT / "private" / "judge_cache"
DETECTORS = ("vanguard-modernbert-large", "tropa-mini-deberta-large", "editlens-roberta-large")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()] if path.is_file() else []


def load(name: str, device: str = "cpu"):
    if name.startswith("editlens"):
        from editlens import EditLens
        m = EditLens("llama", quant=True) if "llama" in name else EditLens("roberta")
        if device != "cpu" and "llama" not in name:
            m.model.to(device)
        return lambda t: m.score(t)[0]
    from detectors import CLASSIFIERS, Classifier
    m = Classifier(CLASSIFIERS[name])
    m.model.to(device)
    return lambda t: m.score(fold(t))[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--device", default="cpu", help="cpu or cuda. One cache must hold scores from one device only: "
                                                    "bfloat16 results differ slightly between devices and CPU types")
    ap.add_argument("--cache", type=Path, default=CACHE, help="score cache directory")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cache = a.cache
    cache.mkdir(parents=True, exist_ok=True)
    items = [(r.get("key") or r["sha"], r["text"], sha(r["text"])) for r in _rows(a.inp) if r["text"].strip()]
    scores: dict[str, dict[str, float]] = {}
    for name in a.detectors.split(","):
        path = cache / f"{name}.jsonl"
        have = {r["sha"]: r["score"] for r in _rows(path)}
        todo = {h: t for _, t, h in items if h not in have}
        if todo:
            fn, t0 = load(name, a.device), time.time()
            with path.open("a") as f:
                for i, (h, t) in enumerate(todo.items(), 1):
                    have[h] = float(fn(t))
                    f.write(json.dumps({"sha": h, "score": have[h]}) + "\n")
                    f.flush()
                    if i % 50 == 0 or i == len(todo):
                        print(f"{name}: {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
            del fn
            import importlib.metadata as md

            (cache / f"{name}.manifest.json").write_text(json.dumps({
                "detector": name, "device": a.device, "dtype": "bfloat16 autocast",
                "gpu": torch.cuda.get_device_name(0) if a.device != "cpu" else None, "torch": torch.__version__,
                "versions": {p: md.version(p) for p in ("transformers", "tokenizers", "safetensors", "huggingface_hub")},
                "texts_in_cache": len(have), "scored_this_run": len(todo), "seconds": round(time.time() - t0, 1),
            }, indent=1))
        scores[name] = have
    out = cache / "out" / a.inp.name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for k, _, h in items:
            f.write(json.dumps({"key": k, "sha": h, "scores": {n: scores[n][h] for n in scores}}) + "\n")
    print(f"wrote {out} ({len(items)} rows)")


if __name__ == "__main__":
    main()
