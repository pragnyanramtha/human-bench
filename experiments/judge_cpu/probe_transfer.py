"""Transfer tests for the frozen-feature probes (uses embeddings cached by probe.py).

Train on every pair except one group, test on that group: leave-one-generator-out,
default->instructed, and leave-one-channel-out. Writes results/probe_transfer.json.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze import metrics
from common import PAIRS_PATH, RESULTS_DIR, read_jsonl
from probe import EMB_DIR

pairs = read_jsonl(PAIRS_PATH)
out = {}
for name in ("vanguard-modernbert-large", "tropa-mini-deberta-large"):
    emb = dict(np.load(EMB_DIR / f"{name}.npz"))
    H = np.stack([emb[f"{p['pair_id']}|human"] for p in pairs])
    A = np.stack([emb[f"{p['pair_id']}|ai"] for p in pairs])
    res = {}
    for key in ("generator", "mode", "channel"):
        for val in sorted({p[key] for p in pairs}):
            te = np.array([p[key] == val for p in pairs])
            if key == "mode" and val == "default":
                continue  # only report train-on-default -> test-on-instructed
            X = np.r_[H[~te], A[~te]]
            y = np.r_[np.zeros((~te).sum()), np.ones((~te).sum())]
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(X, y)
            m = metrics(clf.decision_function(H[te]), clf.decision_function(A[te]))
            res[f"held-out {key}={val}"] = m
            print(f"{name:28s} held-out {key}={val:10s} AUROC {m['auroc']:.3f} TPR@5% {m['tpr5']:.3f} "
                  f"pairwise {m['pairwise']:.3f} (n={m['n']})")
    out[name] = res
(RESULTS_DIR / "probe_transfer.json").write_text(json.dumps(out, indent=1))
