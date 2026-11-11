"""Publish aggregate audit measurements only; never include private text."""

import os
RUN_ID = os.environ.get("HUMANBENCH_RUN_ID", "run")

import json

import numpy as np
from audit import DETECTORS, OUT, PROBE_ON, ROOT, dataset, density
from common import read_jsonl


def pct(v):
    return f"{100 * v:.1f}%"


def main():
    result = json.loads((OUT / "results.json").read_text())
    base = json.loads((OUT / "baseline.json").read_text())
    rows = dataset()
    out = [
        "# CPU judge audit — 2026-11-10",
        "",
        "All text, per-item results and features remain private. These are archival proxy measurements, not HumanScores or validation on commissioned writers.",
        "",
        "## Separation and definitions",
        "",
        "Five grouped outer folds, seed 20261110. Each human/mirror pair stays together; corpus humans have independent groups except normalized duplicate human texts, which share the original pair group. For each test fold, the next fold supplies calibration humans and the other three folds train the probe (StandardScaler and balanced LR, C=0.05). Neither the test fold nor calibration fold enters training. The archival pilot uses the head whose test fold contains its human source; all 12 outputs for that source use that head. Dev outputs and edit flips use the predetermined fold-0 head (fit on folds 2/3/4, calibrated on fold 1), with its cutoff subtracted; threshold zero. The serialized final probe heads are private head__<variant>.joblib artifacts. The outer-fold metrics summarize the training/calibration procedure over all five rotations; pilot sources likewise use their held-out outer head. No refit-to-all-data head is compared with out-of-fold thresholds. One eval/corpus duplicate appears after folding; it is kept in the same group. There are no exact raw duplicates or within-corpus folded duplicates.",
        "",
        "AUROC uses held-out raw eval scores (out-of-fold decision values for the probe); changing only the cutoff leaves off-the-shelf AUROC unchanged. TPR@eval1% is descriptive: the threshold is computed from 140 held-out eval-human scores (one allowed false positive, 0.71%). Cal-TPR uses only the separate calibration fold and is accompanied by the actual test-human FPR. A nominal 1% calibration target does not guarantee 1% on unseen data. Pairwise accuracy is the proportion with AI above the matching human; ties count half. Channel thresholds use channel calibration maxima when fewer than 100 humans are available and fall back to the global threshold for unsupported chat/social channels.",
        "",
        "Variants were fixed before examining new scores: a = prose view; b = remove all github_threads humans from both training and calibration; c = channel thresholds; d = a+b. The additional a-prose-conformal uses the ceil(0.99*(n_cal+1))-th calibration score instead of the empirical quantile, predeclared before inspecting the new metrics. Its finite-sample error control requires exchangeability and does not protect against archival-to-contemporary drift. The off-the-shelf weights are unchanged. This is a development comparison, not an untouched final test for the selected configuration. All 140 briefs were used to develop the original benchmark; source/prompt/generator holdouts and commissioned validation remain necessary.",
        "",
        "## Held-out candidate results",
        "",
        "| configuration | scorer | AUROC | TPR@eval1% | cal-TPR | test FPR (801 humans) | pilot accuracy (1,680) | dev reads human (953) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    short = {
        "vanguard-modernbert-large": "vanguard",
        "tropa-mini-deberta-large": "tropa",
        "probe": "probe",
    }
    for k, r in result.items():
        v, s = k.split("/")
        out.append(
            f"| {v} | {short[s]} | {r['auroc']:.4f} | {pct(r['tpr_at_eval_1pct'])} | {pct(r['tpr'])} | {pct(r['human_fpr'])} | {pct(r['pilot_pairwise'])} | {r['dev_human']} |"
        )
    out += [
        "",
        "There are 960 dev records but seven have empty text or infrastructure failures. They are excluded from judge denominators, not counted as human. The 953 scored outputs and all 1,680 pilot outputs are evaluated. All thresholds use strict `score > threshold` for AI. The old report’s 960 denominator includes failed records.",
        "",
        "## Cached baseline: markdown artefact",
        "",
        "Markdown density = detected block markers, emphasis delimiters and inline-link openers per 100 words. Heavy means density ≥1; below that is labelled light/plain, not guaranteed syntax-free. These are the existing full-data trial scores and thresholds, distinct from the grouped baseline above.",
        "",
        "| scorer | reads human | heavy among human verdicts | heavy among all outputs | density vs score (Spearman) |",
        "|---|---:|---:|---:|---:|",
    ]
    for n in [*DETECTORS, "probe-" + PROBE_ON]:
        b = base[n]
        out.append(
            f"| {n} | {b['human']} | {b['heavy_human']}/{b['human']} ({pct(b['heavy_human'] / b['human'])}) | {b['heavy_total']}/953 | {b['density_rho']:.3f} |"
        )
    out += [
        "",
        "A density association is not a causal attribution. The paired markdown-stripping intervention and robustness tests below provide the causal check. Six of the original probe’s 12 human verdicts contain no detected markdown marker; the artefact cannot explain every miss. Github is just 45 of 801 humans; it does not necessarily set the upper-tail threshold itself.",
        "",
        "## Human score distributions",
        "",
        "All values describe the original calibration: off-the-shelf fold logits and the reconstructed original (ungrouped) probe OOF decisions. These diagnose the old threshold; they are not held-out evidence for the repaired judge. Each cell is median / p90 / p99; the final number is the count above the corresponding threshold.",
        "",
        "| group | n | vanguard | tropa | original probe OOF |",
        "|---|---:|---|---|---|",
    ]
    boundary_note = (
        "The ninth-highest human sets the original cutoff: "
        + "; ".join(
            f"{n}: {base[n]['threshold_source']} ({base[n]['threshold_channel']})"
            for n in [*DETECTORS, "probe-" + PROBE_ON]
        )
        + "."
    )
    out[len(out) - 2 : len(out) - 2] = [boundary_note, ""]
    humans = [r for r in rows if r["side"] == "human"]
    for field in ["source", "channel"]:
        for val in sorted(set(r[field] for r in humans)):
            group = [r for r in humans if r[field] == val]
            cells = []
            for n in DETECTORS:
                b = base[n]["groups"][field + ":" + val]
                cells.append(
                    " / ".join(f"{v:.2f}" for v in b["q"]) + f"; {b['above']} flags"
                )
            b = base["probe-" + PROBE_ON]["groups"][field + ":" + val]
            cells.append(
                " / ".join(f"{v:.2f}" for v in b["q"]) + f"; {b['above']} flags"
            )
            out.append(f"| {field}: {val} | {len(group)} | " + " | ".join(cells) + " |")
    out += [
        "",
        "## Surface inventory",
        "",
        "| pool | n | median words | line breaks | non-ASCII | URLs | blockquotes | lowercase opening | boilerplate proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, r in base["surface"].items():
        out.append(
            f"| {k} | {r['n']} | {r['words']:.0f} | "
            + " | ".join(
                pct(r[x])
                for x in [
                    "linebreak",
                    "nonascii",
                    "url",
                    "quote",
                    "lowercase",
                    "boilerplate",
                ]
            )
            + " |"
        )
    out += [
        "",
        "| pool | split contractions | HTML/entities | literal escapes | encoding damage | spelling proxy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pool, values in base["surface"].items():
        out.append(
            f"| {pool} | "
            + " | ".join(
                pct(values[key])
                for key in [
                    "split_contractions",
                    "html",
                    "escapes",
                    "encoding_damage",
                    "spelling_proxy",
                ]
            )
            + " |"
        )
    out += [
        "",
        "The boilerplate proxy detects mail headers, Signed-off-by, HTML br tags and literal backslash-n escapes; it is not manual adjudication. Encoding, hard wraps, HTML remnants, Enron vocabulary, review sentiment, dialogue punctuation, unequal length distributions and missing channels are dataset/source cues. Fold removes whitespace and common typographic differences but leaves URLs, literal escape sequences, HTML and lexical boilerplate. Quoting and casing can also be legitimate writing traits; correlations alone cannot separate them from source effects. The spelling proxy is a sparse, predefined inventory (teh, recieve, seperate, definately), not a general typo-rate estimate; transposition sensitivity is measured directly below. Removing these lexical cues requires matched, fresh human data, not an ever-growing regex.",
        "",
        "## Trivial-edit robustness",
        "",
        "The same seeded 100 dev outputs are used for every scorer/configuration. Entries are flip counts out of 100 (also percentages). Edits: strip markdown with prose_view; lowercase sentence/line initials; transpose the first two letters of the first two words of at least five ASCII letters; replace em dashes with spaces (preserving word boundaries); collapse whitespace. Zero flips in 100 only bounds the underlying rate to roughly <3%, and a verdict far from threshold can hide score sensitivity.",
        "",
        "| configuration | scorer | markdown | lowercase initials | two typos | em dashes | line breaks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for k, r in result.items():
        v, s = k.split("/")
        out.append(
            f"| {v} | {short[s]} | "
            + " | ".join(
                str(r["flips"][e])
                for e in ["markdown", "lowercase", "typos", "dashes", "linebreaks"]
            )
            + " |"
        )
    out += [
        "",
        "## Original score vs markdown density",
        "",
        "| scorer | density bin | n | median score | reads human |",
        "|---|---|---:|---:|---:|",
    ]
    dev = [r for r in rows if r["kind"] == RUN_ID]
    target = ROOT / "private" / "trials" / RUN_ID / "judge"
    thresholds = json.loads((target / "thresholds.json").read_text())
    for name in [*DETECTORS, "probe-" + PROBE_ON]:
        values = {
            r["key"]: r["score"] for r in read_jsonl(target / f"trials__{name}.jsonl")
        }
        for label, keep in [
            ("zero", lambda d: d == 0),
            ("0 < density < 1", lambda d: 0 < d < 1),
            ("density >= 1", lambda d: d >= 1),
        ]:
            vals = [values[r["key"]] for r in dev if keep(density(r["text"]))]
            rate = np.mean(np.array(vals) <= thresholds[name]["threshold"])
            out.append(
                f"| {name} | {label} | {len(vals)} | {np.median(vals):.3f} | {pct(rate)} |"
            )
    out += [
        "",
        "## Artifacts and reproduction",
        "",
        "`cd experiments/judge_cpu && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python score_trials.py --view prose-v1` runs this audit for both named trial runs. It imports unchanged fold scores/features by normalized-text SHA, caches new scores and features under `private/judge_eval/audit-prose-v1/`, and writes `trials__<scorer>__audit-<variant>.jsonl` and distinct threshold files beside the old dev scores. Pilot files contain AI-minus-held-out-human score margins, with a separate pairwise accuracy file. Existing files and thresholds are untouched. `uv run python audit_report.py` regenerates this aggregate report. At most two Torch threads; no text is sent to APIs. Detector revisions are now pinned in detectors.py to the cached baseline snapshots (vanguard 823061be…; tropa 10fecd08…). Previously fetch() followed main despite the baseline documentation calling these pinned.",
        "",
        "Selection caveats: only two mirror generators, pre-2022 human text, no commissioned or new final holdout, small channel calibration cells and no reliable guarantee of 1% deployment FPR. This audit repairs a normalization shortcut and measurement leakage; it cannot establish that the judge measures natural human writing.",
    ]
    (ROOT / "reports/judge-cpu-audit-2026-11-10.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
