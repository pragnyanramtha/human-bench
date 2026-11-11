"""Cache-aware, grouped, three-way CPU judge audit; all data products are private."""

import os
RUN_ID = os.environ.get("HUMANBENCH_RUN_ID", "run")

import gc
import json
import re
from collections import defaultdict

import joblib
import numpy as np
import torch
from common import (
    EVAL_DIR,
    PAIRS_PATH,
    ROOT,
    SCORES_DIR,
    fold,
    prose_view,
    read_jsonl,
    word_count,
    write_jsonl,
)
from detectors import CLASSIFIERS, Classifier
from probe import EMB_DIR
from scipy.stats import spearmanr
from score_trials import DETECTORS, PROBE_ON, _sha, threshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = EVAL_DIR / "audit-prose-v1"
RUNS = (RUN_ID, "archival-pilot")


def density(t):
    return (
        100
        * len(
            re.findall(
                r"^\s{0,3}(?:#{1,6}\s|[-+*]\s|\d+[.)]\s|>|```|~~~)|\*\*|__|\[[^\]]+\]\(",
                t,
                re.M,
            )
        )
        / max(1, word_count(t))
    )


def edits(t):
    return dict(
        markdown=prose_view(t),
        lowercase=re.sub(r"(^|[.!?]\s+|\n)([A-Z])", lambda m: m[1] + m[2].lower(), t),
        typos=re.sub(
            r"\b[A-Za-z]{5,}\b", lambda m: m[0][1] + m[0][0] + m[0][2:], t, count=2
        ),
        dashes=t.replace("—", " "),
        linebreaks=re.sub(r"\s+", " ", t),
    )


def dataset():
    rows = []
    for p in read_jsonl(PAIRS_PATH):
        for side in ("human", "ai"):
            rows.append(
                dict(
                    key=p["pair_id"] + "|" + side,
                    group=p["pair_id"],
                    text=p[side],
                    side=side,
                    channel=p["channel"],
                    source=p["source"],
                    kind="eval",
                )
            )
    human_groups = {
        view(r["text"]): r["group"]
        for r in rows
        if r["side"] == "human"
        for view in (fold, prose_view)
    }
    for path in sorted((ROOT / "private/corpus").glob("*.jsonl")):
        for r in read_jsonl(path):
            rows.append(
                dict(
                    key="corpus|" + r["id"],
                    group=human_groups.get(
                        fold(r["text"]),
                        human_groups.get(prose_view(r["text"]), "corpus|" + r["id"]),
                    ),
                    text=r["text"],
                    side="human",
                    channel=r["channel"],
                    source=r["source"],
                    kind="corpus",
                )
            )
    for run in RUNS:
        for path in sorted((ROOT / "private/trials" / run / "raw").glob("*/*/s*.json")):
            r = json.loads(path.read_text())
            t = (r.get("scored_text") or "").strip()
            if not t or r.get("infra_failure"):
                continue
            rows.append(
                dict(
                    key=f"{r['model']}|{r['task']}|s{r['sample']}",
                    group=r["task"],
                    text=t,
                    side="ai",
                    channel=r.get("channel", "unknown"),
                    source="model",
                    kind=run,
                )
            )
    return rows


@torch.inference_mode()
def tropa_joint(det, text):
    """One backbone pass supplies both pooled features and the original head."""
    vecs, logits, weights = [], [], []
    for w in det.windows(text):
        ids = torch.tensor([w])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            h = det.model.model(input_ids=ids, attention_mask=torch.ones_like(ids))[0]
            z = det.model.classifier(
                h.sum(1) / torch.tensor(len(w), dtype=h.dtype)
            ).float()
        lp = torch.nn.functional.log_softmax(z, -1)[0, 0]
        value = torch.log1p(-lp.exp().clamp(max=1 - 1e-7)) - lp
        vecs.append(h.float().mean(1)[0].numpy())
        logits.append(float(value))
        weights.append(len(w))
    return float(np.average(logits, weights=weights)), np.average(
        vecs, axis=0, weights=weights
    )


def _save_features(path, features):
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **features)
    temporary.replace(path)


def caches(rows, samples):
    OUT.mkdir(parents=True, exist_ok=True)
    texts = {}
    for r in rows:
        for v in (fold, prose_view):
            texts[_sha(v(r["text"]))] = v(r["text"])
    for r in samples:
        for t in edits(r["text"]).values():
            for v in (fold, prose_view):
                texts[_sha(v(t))] = v(t)
    bykey = {r["key"]: r for r in rows}
    features = {}
    fp = OUT / "features.npz"
    for k, v in dict(np.load(EMB_DIR / f"{PROBE_ON}.npz")).items():
        if k in bykey:
            features[_sha(fold(bykey[k]["text"]))] = v
    for run in RUNS:
        p = ROOT / "private/trials" / run / "judge" / f"emb__{PROBE_ON}.npz"
        if p.exists():
            for k, v in dict(np.load(p)).items():
                features[k.rsplit("#", 1)[-1]] = v
    if fp.exists():
        features.update(dict(np.load(fp)))
    scores = {}
    for name in DETECTORS:
        cache = OUT / f"{name}.jsonl"
        sc = {r["sha"]: r["score"] for r in read_jsonl(cache)}
        for r in read_jsonl(SCORES_DIR / f"{name}__fold.jsonl"):
            k = r["pair_id"] + "|" + r["side"]
            if k in bykey:
                sc.setdefault(_sha(fold(bykey[k]["text"])), r["score"])
        for p in [
            EVAL_DIR / "calibration" / f"{name}.jsonl",
            *(
                ROOT / "private/trials" / run / "judge" / f"trials__{name}.jsonl"
                for run in RUNS
            ),
        ]:
            for r in read_jsonl(p):
                sc.setdefault(r["sha"], r["score"])
        todo = [
            h for h in texts if h not in sc or (name == PROBE_ON and h not in features)
        ]
        print(name, "need", len(todo), "of", len(texts), flush=True)
        if todo:
            det = Classifier(CLASSIFIERS[name])
            with cache.open("a") as f:
                for i, h in enumerate(todo):
                    if name == PROBE_ON and h not in features:
                        value, features[h] = tropa_joint(det, texts[h])
                        if h not in sc:
                            sc[h] = value
                            f.write(json.dumps(dict(sha=h, score=value)) + "\n")
                            f.flush()
                    if h not in sc:
                        sc[h] = det.score(texts[h])[0]
                        f.write(json.dumps(dict(sha=h, score=sc[h])) + "\n")
                        f.flush()
                    if (i + 1) % 25 == 0:
                        print(name, i + 1, "/", len(todo), flush=True)
                        if name == PROBE_ON:
                            _save_features(fp, features)
            del det
            gc.collect()
        write_jsonl(cache, [dict(sha=h, score=s) for h, s in sc.items()])
        if name == PROBE_ON:
            _save_features(fp, features)
        scores[name] = sc
    return scores, features


def model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"),
    )


def run(run_id=RUNS[0], no_probe=False):
    if run_id not in RUNS:
        raise ValueError(f"This audit supports {RUNS}; got {run_id!r}")
    torch.set_num_threads(2)
    rows = dataset()
    dev = [r for r in rows if r["kind"] == RUNS[0]]
    pilot = [r for r in rows if r["kind"] == RUNS[1]]
    samples = [
        dev[i]
        for i in np.random.default_rng(20261110).choice(len(dev), 100, replace=False)
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample-keys.json").write_text(json.dumps([r["key"] for r in samples]))
    sc, emb = caches(rows, samples)
    tr = [r for r in rows if r["kind"] in ("eval", "corpus")]
    y = np.array([r["side"] == "ai" for r in tr], int)
    groups = np.array([r["group"] for r in tr])
    folds = np.zeros(len(tr), int)
    for f, (_, te) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=20261110).split(
            np.zeros(len(tr)), y, groups
        )
    ):
        folds[te] = f
    gf = dict(zip(groups, folds))
    results = {}
    detail = {}
    print(
        "pilot source matches",
        sum(r["group"] in gf for r in pilot),
        "/",
        len(pilot),
        flush=True,
    )
    for variant in (
        "baseline",
        "a-prose",
        "a-prose-conformal",
        "b-drop-github",
        "c-channel",
        "d-prose-drop-github",
    ):
        view = (
            prose_view
            if variant in ("a-prose", "a-prose-conformal", "d-prose-drop-github")
            else fold
        )
        drop = "drop-github" in variant

        def h(r):
            return _sha(view(r["text"]))

        X = np.stack([emb[h(r)] for r in tr])
        XD = np.stack([emb[h(r)] for r in dev])
        for scorer in DETECTORS + ([] if no_probe else ["probe"]):
            margins = np.zeros(len(tr))
            raw_oof = np.zeros(len(tr))
            ds = []
            calibrations = []
            ps = {}
            pilot_scores = {}
            ms = defaultdict(list)
            for f in range(5):
                test = folds == f
                cal = (folds == (f + 1) % 5) & (y == 0)
                fit = (folds != f) & (folds != (f + 1) % 5)
                if drop:
                    keep = np.array([r["source"] != "github_threads" for r in tr])
                    cal &= keep
                    fit &= keep
                clf = None
                if scorer == "probe":
                    clf = model().fit(X[fit], y[fit])
                    s = clf.decision_function(X)
                    d = clf.decision_function(XD)
                    if f == 0:
                        joblib.dump(clf, OUT / f"head__{variant}.joblib")
                else:
                    s = np.array([sc[scorer][h(r)] for r in tr])
                    d = np.array([sc[scorer][h(r)] for r in dev])
                th = threshold(s[cal])
                if variant == "a-prose-conformal":
                    n_cal = int(cal.sum())
                    rank = int(np.ceil(0.99 * (n_cal + 1)))
                    if rank > n_cal:
                        raise ValueError(
                            "Too few calibration humans for a finite 99% conformal cutoff"
                        )
                    th = float(np.sort(s[cal])[rank - 1])
                ch = {}
                if variant == "c-channel":
                    for c in set(r["channel"] for r in tr):
                        ix = cal & np.array([r["channel"] == c for r in tr])
                        if ix.any():
                            ch[c] = threshold(s[ix])

                calibrations.append(
                    dict(
                        fold=f,
                        threshold=th,
                        channels=ch,
                        n_cal=int(cal.sum()),
                        n_fit=int(fit.sum()),
                    )
                )

                def thr(r):
                    return ch.get(r["channel"], th)

                raw_oof[test] = s[test]
                margins[test] = [s[i] - thr(tr[i]) for i in np.flatnonzero(test)]
                ds.append(d - np.array([thr(r) for r in dev]))
                for r in pilot:
                    if gf.get(r["group"]) != f:
                        continue
                    hr = next(
                        z
                        for z in tr
                        if z["group"] == r["group"] and z["side"] == "human"
                    )
                    delta = (
                        sc[scorer][h(r)] - sc[scorer][h(hr)]
                        if clf is None
                        else np.diff(
                            clf.decision_function(np.stack([emb[h(hr)], emb[h(r)]]))
                        )[0]
                    )
                    pilot_scores[r["key"]] = float(delta)
                    ps[r["key"]] = float(delta > 0) + 0.5 * float(delta == 0)
                for r in samples:
                    for edit, t in {"original": r["text"], **edits(r["text"])}.items():
                        sha = _sha(view(t))
                        v = (
                            sc[scorer][sha]
                            if clf is None
                            else clf.decision_function(emb[sha][None])[0]
                        )
                        ms[r["key"], edit].append(v - thr(r))
            dm = ds[0]  # Preselected final head; calibration remains independent.
            ev = np.array([r["kind"] == "eval" for r in tr])
            result = dict(
                auroc=float(roc_auc_score(y[ev], raw_oof[ev])),
                tpr_at_eval_1pct=float(
                    np.mean(raw_oof[ev & (y == 1)] > threshold(raw_oof[ev & (y == 0)]))
                ),
                eval_human_fpr=float(np.mean(margins[ev & (y == 0)] > 0)),
                tpr=float(np.mean(margins[ev & (y == 1)] > 0)),
                human_fpr=float(np.mean(margins[y == 0] > 0)),
                final_head_test_fpr=float(
                    np.mean(margins[(y == 0) & (folds == 0)] > 0)
                ),
                pilot_pairwise=float(np.mean(list(ps.values()))) if ps else None,
                pilot_n=len(ps),
                dev_human=int(sum(dm <= 0)),
                flips={
                    e: int(
                        sum(
                            (ms[r["key"], e][0] > 0)
                            != (ms[r["key"], "original"][0] > 0)
                            for r in samples
                        )
                    )
                    for e in edits("test")
                },
            )
            # Source-relative pilot margins use a head that never saw its reference.
            write_jsonl(
                ROOT
                / "private/trials"
                / RUNS[1]
                / "judge"
                / f"trials__{scorer}__audit-{variant}.jsonl",
                [
                    dict(key=k, score=v, score_kind="AI minus held-out human")
                    for k, v in pilot_scores.items()
                ],
            )
            results[variant + "/" + scorer] = result
            print(variant, scorer, result, flush=True)
            target = ROOT / "private/trials" / RUNS[0] / "judge"
            write_jsonl(
                target / f"trials__{scorer}__audit-{variant}.jsonl",
                [dict(key=r["key"], score=float(v)) for r, v in zip(dev, dm)],
            )
            (target / f"thresholds__{scorer}__audit-{variant}.json").write_text(
                json.dumps(
                    dict(
                        threshold=0,
                        calibrations=calibrations,
                        final_head_fold=0,
                        seed=20261110,
                        method="fixed head fold 0; folds 2/3/4 fit, fold 1 calibration, fold 0 test; exported score subtracts its calibration threshold",
                    )
                )
            )
            write_jsonl(
                ROOT
                / "private/trials"
                / RUNS[1]
                / "judge"
                / f"pairwise__{scorer}__audit-{variant}.jsonl",
                [dict(key=k, accuracy=v) for k, v in ps.items()],
            )
            detail[variant + "/" + scorer] = [
                dict(key=r["key"], score=float(v)) for r, v in zip(tr, margins)
            ]
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    (OUT / "oof.json").write_text(json.dumps(detail))
    audit_baseline(rows, emb)


def audit_baseline(rows, emb=None):
    dev = [r for r in rows if r["kind"] == RUNS[0]]
    hum = [r for r in rows if r["side"] == "human"]
    target = ROOT / "private/trials" / RUNS[0] / "judge"
    ths = json.loads((target / "thresholds.json").read_text())
    out = {}
    for name in [*DETECTORS, "probe-" + PROBE_ON]:
        s = {r["key"]: r["score"] for r in read_jsonl(target / f"trials__{name}.jsonl")}
        th = ths[name]["threshold"]
        passed = [r for r in dev if s[r["key"]] <= th]
        out[name] = dict(
            human=len(passed),
            heavy_human=sum(density(r["text"]) >= 1 for r in passed),
            heavy_total=sum(density(r["text"]) >= 1 for r in dev),
            density_rho=float(
                spearmanr(
                    [density(r["text"]) for r in dev], [s[r["key"]] for r in dev]
                ).statistic
            ),
        )
        if name not in DETECTORS:
            if emb is None:
                continue
            from sklearn.model_selection import StratifiedKFold

            train = [r for r in rows if r["kind"] in ("eval", "corpus")]
            X = np.stack([emb[_sha(fold(r["text"]))] for r in train])
            y = np.array([r["side"] == "ai" for r in train], int)
            old_oof = np.zeros(len(train))
            for fit, test in StratifiedKFold(5, shuffle=True, random_state=0).split(
                X, y
            ):
                old_oof[test] = model().fit(X[fit], y[fit]).decision_function(X[test])
            hs = {r["key"]: float(v) for r, v in zip(train, old_oof)}
            out[name]["reconstructed_threshold"] = threshold(old_oof[y == 0])
            write_jsonl(
                OUT / "original-probe-calibration.jsonl",
                [dict(key=r["key"], score=hs[r["key"]]) for r in hum],
            )
        else:
            hs = {
                r["pair_id"] + "|" + r["side"]: r["score"]
                for r in read_jsonl(SCORES_DIR / f"{name}__fold.jsonl")
            }
            hs.update(
                {
                    r["key"]: r["score"]
                    for r in read_jsonl(EVAL_DIR / "calibration" / f"{name}.jsonl")
                }
            )
        boundary = sorted(hum, key=lambda r: hs[r["key"]], reverse=True)[8]
        out[name]["threshold_source"] = boundary["source"]
        out[name]["threshold_channel"] = boundary["channel"]
        out[name]["groups"] = {}
        for field in ("source", "channel"):
            for val in sorted(set(r[field] for r in hum)):
                a = [hs[r["key"]] for r in hum if r[field] == val]
                out[name]["groups"][field + ":" + val] = dict(
                    n=len(a),
                    q=np.quantile(a, [0.5, 0.9, 0.99]).tolist(),
                    above=sum(v > th for v in a),
                )

    def surface(rs):
        return dict(
            n=len(rs),
            words=float(np.median([word_count(r["text"]) for r in rs])),
            **{
                k: float(np.mean([bool(re.search(p, r["text"], re.M)) for r in rs]))
                for k, p in dict(
                    linebreak="\n",
                    nonascii=r"[^\x00-\x7f]",
                    url=r"https?://",
                    quote=r"^\s*>",
                    lowercase=r"\A[a-z]",
                    split_contractions="\\b\\w+\\s+['’](?:s|m|d|re|ve|ll)\\b|\\b\\w+\\s+n['’]t\\b",
                    html="</?[A-Za-z][^>]*>|&(?:amp|lt|gt|quot|apos|#\\d+);",
                    escapes="\\\\[nrt]",
                    encoding_damage="�|Ã.|Â.|â€",
                    spelling_proxy="(?i)\\b(?:teh|recieve|seperate|definately)\\b",
                    boilerplate=r"(?i)^(?:from:|sent:|subject:|signed-off-by:|on .+ wrote:)|<br\s*/?>|\\n",
                ).items()
            },
        )

    out["surface"] = {
        k: surface(
            [
                r
                for r in rows
                if r["kind"] == k and (k != "eval" or r["side"] == "human")
            ]
        )
        for k in ("eval", "corpus", *RUNS)
    }
    out["surface"]["eval_ai"] = surface(
        [r for r in rows if r["kind"] == "eval" and r["side"] == "ai"]
    )
    (OUT / "baseline.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
