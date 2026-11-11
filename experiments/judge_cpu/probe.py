"""CPU-trainable judge: frozen encoder features + logistic regression, grouped CV.

Extracts a mean-pooled final hidden state per text from a detector backbone
(one forward pass, cached), then fits L2 logistic regression with the same
grouped folds as the TF-IDF baseline (a text and its mirror never split).
  uv run python probe.py <classifier-name> [<classifier-name> ...]
Writes private/judge_eval/scores/probe-<name>__fold.jsonl so analyze.py picks it up
(scores there are out-of-fold decision values, not the off-the-shelf head).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import EVAL_DIR, PAIRS_PATH, SCORES_DIR, fold, read_jsonl, word_count
from detectors import CLASSIFIERS, Classifier

EMB_DIR = EVAL_DIR / "emb"


@torch.inference_mode()
def embed(det: Classifier, text: str) -> np.ndarray:
    wins = det.windows(text)
    backbone = getattr(det.model, "model", None) or getattr(det.model, "roberta", None)
    if backbone is None:  # HF *ForSequenceClassification
        backbone = det.model.base_model
    vecs, ws = [], []
    for w in wins:
        ids = torch.tensor([w])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            h = backbone(input_ids=ids, attention_mask=torch.ones_like(ids))[0].float()
        vecs.append(h[0].mean(0).numpy())
        ws.append(len(w))
    return np.average(np.stack(vecs), axis=0, weights=ws)


def main() -> None:
    torch.set_num_threads(2)
    pairs = read_jsonl(PAIRS_PATH)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:]:
        cache = EMB_DIR / f"{name}.npz"
        texts = [(p["pair_id"], s, fold(p[s])) for p in pairs for s in ("human", "ai")]
        have = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
        keys = [f"{pid}|{s}" for pid, s, _ in texts]
        if not all(k in have for k in keys):
            det = Classifier(CLASSIFIERS[name])
            t0 = time.time()
            for (pid, s, t), k in zip(texts, keys):
                if k not in have:
                    have[k] = embed(det, t)
            np.savez(cache, **have)
            print(f"{name}: embedded in {time.time() - t0:.0f}s", flush=True)
        X = np.stack([have[k] for k in keys])
        y = np.array([0 if s == "human" else 1 for _, s, _ in texts])
        g = np.array([pid for pid, _, _ in texts])
        oof = np.zeros(len(X))
        seeds = (0, 1, 2, 3, 4)
        for seed in seeds:
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
                clf.fit(X[tr], y[tr])
                oof[te] += clf.decision_function(X[te]) / len(seeds)
        out = SCORES_DIR / f"probe-{name}__fold.jsonl"
        with out.open("w") as f:
            for (pid, s, t), v in zip(texts, oof):
                f.write(json.dumps(dict(pair_id=pid, side=s, score=float(v), tokens=0, words=word_count(t),
                                        wall_s=0.0, cpu_s=0.0)) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
