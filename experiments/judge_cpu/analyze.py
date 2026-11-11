"""Compute metrics for every scored detector plus the TF-IDF baseline.

  uv run python analyze.py            # writes private/judge_eval/results/{summary.json,tables.md}

Metrics (AI = positive class, higher score = more AI):
  AUROC; TPR at 1% / 5% FPR (threshold set on this set's human texts);
  pairwise accuracy = P(score(ai mirror) > score(its human source)), ties count 0.5.
95% CIs from a pair-level bootstrap (resample pairs, keep both sides together).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline, make_union

from common import PAIRS_PATH, RESULTS_DIR, SCORES_DIR, fold, light, read_jsonl

N_BOOT = 2000
RNG = np.random.default_rng(0)
LM_KEYS = {"binoculars": "Binoculars", "fdg_b2i": "Fast-DetectGPT (base->instruct)",
           "fdg_b2b": "Fast-DetectGPT (base only)", "loglik_inst": "log-likelihood (instruct)",
           "loglik_base": "log-likelihood (base)"}


# ---------------------------------------------------------------- metrics

def tpr_at(h: np.ndarray, a: np.ndarray, fpr: float) -> float:
    hs = np.sort(h)[::-1]
    k = int(np.floor(fpr * len(hs)))  # number of humans allowed above threshold
    thr = hs[k] if k < len(hs) else -np.inf
    return float(np.mean(a > thr))


def pairwise(h: np.ndarray, a: np.ndarray) -> float:
    return float(np.mean((a > h) + 0.5 * (a == h)))


def auroc(h: np.ndarray, a: np.ndarray) -> float:
    y = np.r_[np.zeros(len(h)), np.ones(len(a))]
    return float(roc_auc_score(y, np.r_[h, a]))


def metrics(h, a) -> dict:
    return dict(auroc=auroc(h, a), tpr1=tpr_at(h, a, 0.01), tpr5=tpr_at(h, a, 0.05), pairwise=pairwise(h, a),
                n=len(h))


def boot(h, a, n=N_BOOT) -> dict:
    idx = RNG.integers(0, len(h), size=(n, len(h)))
    out = defaultdict(list)
    for ix in idx:
        hh, aa = h[ix], a[ix]
        out["auroc"].append(auroc(hh, aa))
        out["tpr1"].append(tpr_at(hh, aa, 0.01))
        out["tpr5"].append(tpr_at(hh, aa, 0.05))
        out["pairwise"].append(pairwise(hh, aa))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


# ---------------------------------------------------------------- loading

def load_scores(pairs: list[dict]) -> dict[str, dict]:
    """name -> {'h': array, 'a': array, 'meta': {...}, 'speed': {...}} aligned to pairs order."""
    order = [p["pair_id"] for p in pairs]
    res = {}
    for f in sorted(SCORES_DIR.glob("*.jsonl")):
        tag = f.stem
        rows = read_jsonl(f)
        by = {(r["pair_id"], r["side"]): r for r in rows}
        if not all((pid, s) in by for pid in order for s in ("human", "ai")):
            print(f"skip {tag}: incomplete ({len(by)}/{2 * len(order)})", file=sys.stderr)
            continue
        meta_p = f.with_suffix(".meta.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        all_rows = [by[(pid, s)] for pid in order for s in ("human", "ai")]
        words = sum(r["words"] for r in all_rows)
        speed = dict(wall_per_500w=sum(r["wall_s"] for r in all_rows) / words * 500,
                     cpu_per_500w=sum(r["cpu_s"] for r in all_rows) / words * 500,
                     peak_rss_mb=meta.get("peak_rss_mb"), threads=meta.get("threads"), dtype=meta.get("dtype"))
        keys = [k for k in LM_KEYS if k in all_rows[0]] or ["score"]
        for k in keys:
            name = tag if k == "score" else f"{tag}::{k}"
            res[name] = dict(h=np.array([by[(pid, "human")][k] for pid in order], dtype=float),
                             a=np.array([by[(pid, "ai")][k] for pid in order], dtype=float),
                             speed=speed, meta=meta)
    return res


# ---------------------------------------------------------------- TF-IDF baseline

def tfidf_model():
    feats = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=False),
    )
    return make_pipeline(feats, LogisticRegression(C=10.0, max_iter=5000))


def tfidf_oof(pairs, view, seeds=(0, 1, 2, 3, 4), folds=5) -> tuple[np.ndarray, np.ndarray, list]:
    """Grouped CV (a human text and its mirror always share a fold); OOF scores averaged over seeds."""
    X = [view(p["human"]) for p in pairs] + [view(p["ai"]) for p in pairs]
    y = np.r_[np.zeros(len(pairs)), np.ones(len(pairs))]
    g = np.r_[np.arange(len(pairs)), np.arange(len(pairs))]
    oof_all, per_seed = np.zeros(len(X)), []
    for s in seeds:
        oof = np.zeros(len(X))
        for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=s).split(X, y, g):
            m = tfidf_model().fit([X[i] for i in tr], y[tr])
            oof[te] = m.decision_function([X[i] for i in te])
        per_seed.append(metrics(oof[: len(pairs)], oof[len(pairs):]))
        oof_all += oof / len(seeds)
    return oof_all[: len(pairs)], oof_all[len(pairs):], per_seed


def tfidf_transfer(pairs, key) -> dict:
    """Train on all pairs except one group value, test on that group (e.g. leave-one-generator-out)."""
    out = {}
    for val in sorted({p[key] for p in pairs}):
        tr = [p for p in pairs if p[key] != val]
        te = [p for p in pairs if p[key] == val]
        Xtr = [fold(p["human"]) for p in tr] + [fold(p["ai"]) for p in tr]
        ytr = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
        m = tfidf_model().fit(Xtr, ytr)
        h = m.decision_function([fold(p["human"]) for p in te])
        a = m.decision_function([fold(p["ai"]) for p in te])
        out[val] = metrics(h, a)
    return out


# ---------------------------------------------------------------- breakdowns

def breakdown(h, a, pairs, key) -> dict:
    """Per group: pairwise acc, within-group AUROC, and detection rate at the GLOBAL 5%-FPR threshold."""
    hs = np.sort(h)[::-1]
    thr5 = hs[int(np.floor(0.05 * len(hs)))]
    groups = defaultdict(list)
    for i, p in enumerate(pairs):
        groups[p[key]].append(i)
    out = {}
    for val, ix in sorted(groups.items()):
        ix = np.array(ix)
        out[val] = dict(n=len(ix), pairwise=pairwise(h[ix], a[ix]), auroc=auroc(h[ix], a[ix]),
                        det_at_global5=float(np.mean(a[ix] > thr5)), fpr_at_global5=float(np.mean(h[ix] > thr5)))
    return out


def fmt_ci(v, ci):
    return f"{v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def main() -> None:
    pairs = read_jsonl(PAIRS_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores(pairs)

    print("TF-IDF grouped CV ...", file=sys.stderr)
    for view_name, view in (("fold", fold), ("light", light)):
        h, a, per_seed = tfidf_oof(pairs, view)
        scores[f"tfidf-logreg__{view_name}"] = dict(h=h, a=a, speed={}, meta=dict(per_seed=per_seed))

    # Stack of the off-the-shelf detectors (their scores as features), same grouped CV.
    base = [k for k in scores if k.endswith("__fold") or "__fold::" in k]
    base = [k for k in base if not k.startswith(("tfidf", "probe-")) and not k.endswith(("::loglik_base",))]
    if len(base) >= 2:
        from sklearn.preprocessing import StandardScaler
        X = np.c_[[np.r_[scores[k]["h"], scores[k]["a"]] for k in base]].T
        y = np.r_[np.zeros(len(pairs)), np.ones(len(pairs))]
        g = np.r_[np.arange(len(pairs)), np.arange(len(pairs))]
        oof = np.zeros(len(X))
        for s in range(5):
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], y[tr])
                oof[te] += m.decision_function(X[te]) / 5
        scores["stack-offshelf-logreg__fold"] = dict(h=oof[: len(pairs)], a=oof[len(pairs):], speed={},
                                                     meta=dict(features=base))

    # Prefer clean, idle-machine timings (timing.py) over the in-run timings, which overlapped with
    # mirror generation for part of the set.
    for name, d in scores.items():
        tf = RESULTS_DIR / "timing" / f"{name.split('__')[0]}__t2.json"
        if tf.exists():
            t = json.loads(tf.read_text())
            d["speed"] = dict(d["speed"], wall_per_500w=t["wall_per_500w"], cpu_per_500w=t["cpu_per_500w"],
                              peak_rss_mb=t["peak_rss_mb"], source="timing.py")

    summary, rows = {}, []
    for name, d in scores.items():
        h, a = d["h"], d["a"]
        m = metrics(h, a)
        ci = boot(h, a)
        summary[name] = dict(metrics=m, ci=ci, speed=d["speed"], meta=d.get("meta", {}),
                             by_channel=breakdown(h, a, pairs, "channel"),
                             by_generator=breakdown(h, a, pairs, "generator"),
                             by_mode=breakdown(h, a, pairs, "mode"),
                             by_source=breakdown(h, a, pairs, "source"))
        rows.append((name, m, ci, d["speed"]))

    print("TF-IDF transfer ...", file=sys.stderr)
    summary["tfidf_transfer"] = {k: tfidf_transfer(pairs, k) for k in ("generator", "channel", "mode")}

    rows.sort(key=lambda r: -r[1]["pairwise"])
    lines = ["| detector | AUROC [95% CI] | TPR@1%FPR | TPR@5%FPR [CI] | pairwise acc [CI] | wall s/500w | "
             "CPU s/500w | peak RSS MB |", "|---|---|---|---|---|---|---|---|"]
    for name, m, ci, sp in rows:
        lines.append(
            f"| {name} | {fmt_ci(m['auroc'], ci['auroc'])} | {m['tpr1']:.3f} | {fmt_ci(m['tpr5'], ci['tpr5'])} | "
            f"{fmt_ci(m['pairwise'], ci['pairwise'])} | {sp.get('wall_per_500w', float('nan')):.2f} | "
            f"{sp.get('cpu_per_500w', float('nan')):.2f} | {sp.get('peak_rss_mb', '')} |")
    for key in ("channel", "generator", "mode", "source"):
        vals = sorted(next(iter(summary.values()))[f"by_{key}"].keys())
        lines += ["", f"### pairwise accuracy / detection@global-5%FPR by {key}", "",
                  "| detector | " + " | ".join(f"{v} (n={summary[rows[0][0]][f'by_{key}'][v]['n']})" for v in vals)
                  + " |", "|---" * (len(vals) + 1) + "|"]
        for name, *_ in rows:
            b = summary[name][f"by_{key}"]
            lines.append(f"| {name} | " + " | ".join(f"{b[v]['pairwise']:.2f} / {b[v]['det_at_global5']:.2f}"
                                                   for v in vals) + " |")
    lines += ["", "### TF-IDF transfer (train on the rest, test on the held-out group)", ""]
    for key, d in summary["tfidf_transfer"].items():
        for val, m in d.items():
            lines.append(f"- held-out {key}={val}: AUROC {m['auroc']:.3f}, pairwise {m['pairwise']:.3f}, "
                         f"TPR@5% {m['tpr5']:.3f} (n={m['n']})")
    (RESULTS_DIR / "tables.md").write_text("\n".join(lines) + "\n")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
