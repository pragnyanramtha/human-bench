"""Score model trial outputs with the CPU detectors, calibrated on real human text.

    uv run python score_trials.py --run-id R

For each detector (vanguard-modernbert-large, tropa-mini-deberta-large; pinned revisions
in detectors.py) this scores, with caching:
  - every trial output: the scored text of the record (all writes for agent tasks), folded
    the same way as the judge eval (common.fold);
  - the human calibration set: the 140 human texts of the judge eval (scores already cached
    in private/judge_eval/scores/) plus the 661 public-corpus texts in private/corpus/.
It also trains the CPU probe the judge report recommends: logistic regression on frozen
tropa-mini features, trained on the judge eval (140 human, 140 model mirrors) plus the
corpus humans. Its threshold comes from out-of-fold scores, so no human used to set the
threshold was seen in training.

Output: private/trials/<run>/judge/trials__<detector>.jsonl and trials__probe-*.jsonl; the human
calibration scores are cached once in private/judge_eval/calibration/<detector>.jsonl;
and thresholds.json (each detector's score above which at most 1% of the human calibration
texts fall). Higher scores mean "more AI".
"""

from __future__ import annotations

import os
RUN_ID = os.environ.get("HUMANBENCH_RUN_ID", "run")

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import EVAL_DIR, PAIRS_PATH, ROOT, SCORES_DIR, fold, read_jsonl
from detectors import CLASSIFIERS, Classifier
from probe import EMB_DIR, embed

DETECTORS = ["vanguard-modernbert-large", "tropa-mini-deberta-large"]
PROBE_ON = "tropa-mini-deberta-large"
FPR = 0.01
CAL_DIR = EVAL_DIR / "calibration"  # human calibration scores, shared by every run


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def trial_texts(run_dir: Path) -> list[tuple[str, str]]:
    out = []
    for p in sorted((run_dir / "raw").glob("*/*/s*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        text = (r.get("scored_text") or "").strip()
        if r.get("infra_failure") or not text:
            continue
        out.append((f"{r['model']}|{r['task']}|s{r['sample']}", fold(text)))
    return out


def corpus_texts() -> list[tuple[str, str]]:
    out = []
    for f in sorted((ROOT / "private" / "corpus").glob("*.jsonl")):
        for row in read_jsonl(f):
            out.append((f"corpus|{row['id']}", fold(row["text"])))
    return out


def _cached(path: Path) -> dict[str, dict]:
    return {r["key"]: r for r in read_jsonl(path)} if path.exists() else {}


def threshold(human_scores: np.ndarray, fpr: float = FPR) -> float:
    """Largest score that keeps at most `fpr` of human texts above it."""
    hs = np.sort(human_scores)[::-1]
    k = int(np.floor(fpr * len(hs)))  # humans allowed above the threshold
    return float(hs[k])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--view", choices=("fold", "prose-v1"), default="fold",
                    help="prose-v1 runs the grouped audit and writes versioned scores for both audit runs")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the probe (use for the archival pilot: the probe was trained on those human texts)")
    a = ap.parse_args()
    torch.set_num_threads(min(2, max(1, a.threads)))
    if a.view == "prose-v1":
        from audit import run
        run(run_id=a.run_id, no_probe=a.no_probe)
        return
    run_dir = ROOT / "private" / "trials" / a.run_id
    out_dir = run_dir / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = trial_texts(run_dir)
    corpus = corpus_texts()
    print(f"{len(trials)} trial outputs, {len(corpus)} corpus human texts", flush=True)

    thresholds: dict[str, dict] = {}
    for name in DETECTORS:
        det = None
        for kind, items in (("calibration", corpus), ("trials", trials)):
            path = (CAL_DIR / f"{name}.jsonl") if kind == "calibration" else out_dir / f"trials__{name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            have = _cached(path)
            todo = [(k, t) for k, t in items if have.get(k, {}).get("sha") != _sha(t)]
            if todo:
                det = det or Classifier(CLASSIFIERS[name])
                t0 = time.time()
                with path.open("a") as f:
                    for i, (k, t) in enumerate(todo, 1):
                        score, ntok = det.score(t)
                        row = {"key": k, "sha": _sha(t), "score": score, "tokens": ntok}
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        have[k] = row
                        if i % 50 == 0 or i == len(todo):
                            print(f"  {name} {kind}: {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
        eval_humans = [r["score"] for r in read_jsonl(SCORES_DIR / f"{name}__fold.jsonl") if r["side"] == "human"]
        cal = [r["score"] for r in _cached(CAL_DIR / f"{name}.jsonl").values()]
        hs = np.array(eval_humans + cal)
        thresholds[name] = {"threshold": threshold(hs), "n_humans": len(hs), "fpr": FPR}
        print(f"{name}: threshold {thresholds[name]['threshold']:.3f} from {len(hs)} human texts", flush=True)

    if a.no_probe:
        (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
        print(json.dumps(thresholds, indent=1))
        return
    # Probe: frozen tropa-mini features + logistic regression.
    cache = EMB_DIR / f"{PROBE_ON}.npz"
    emb = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    extra_path = out_dir / f"emb__{PROBE_ON}.npz"
    extra = dict(np.load(extra_path, allow_pickle=True)) if extra_path.exists() else {}
    need = [(k, t) for k, t in corpus + trials if f"{k}#{_sha(t)}" not in extra]
    if need:
        det = Classifier(CLASSIFIERS[PROBE_ON])
        t0 = time.time()
        for i, (k, t) in enumerate(need, 1):
            extra[f"{k}#{_sha(t)}"] = embed(det, t)
            if i % 100 == 0 or i == len(need):
                np.savez(extra_path, **extra)
                print(f"  probe features: {i}/{len(need)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
    pairs = read_jsonl(PAIRS_PATH)
    X_eval = np.stack([emb[f"{p['pair_id']}|{s}"] for p in pairs for s in ("human", "ai")])
    y_eval = np.array([0 if s == "human" else 1 for _ in pairs for s in ("human", "ai")])
    X_cor = np.stack([extra[f"{k}#{_sha(t)}"] for k, t in corpus])
    X = np.vstack([X_eval, X_cor])
    y = np.concatenate([y_eval, np.zeros(len(X_cor), dtype=int)])

    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))

    oof = np.zeros(len(X))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = model().fit(X[tr], y[tr]).decision_function(X[te])
    probe_thr = threshold(oof[y == 0])
    clf = model().fit(X, y)
    X_tr = np.stack([extra[f"{k}#{_sha(t)}"] for k, t in trials])
    scores = clf.decision_function(X_tr)
    with (out_dir / f"trials__probe-{PROBE_ON}.jsonl").open("w") as f:
        for (k, _), s in zip(trials, scores):
            f.write(json.dumps({"key": k, "score": float(s)}) + "\n")
    tpr_eval = float(np.mean(oof[: len(X_eval)][y_eval == 1] > probe_thr))
    thresholds[f"probe-{PROBE_ON}"] = {"threshold": probe_thr, "n_humans": int((y == 0).sum()), "fpr": FPR,
                                      "oof_tpr_on_eval_mirrors": tpr_eval}
    print(f"probe: threshold {probe_thr:.3f}; out-of-fold, it catches {tpr_eval:.0%} of eval mirrors", flush=True)
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    print(json.dumps(thresholds, indent=1))


if __name__ == "__main__":
    main()
