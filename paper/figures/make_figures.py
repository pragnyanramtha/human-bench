#!/usr/bin/env python3
"""The data figures of paper/humanbench.tex, drawn from the public scoring files only.

    uv run --with matplotlib python paper/figures/make_figures.py

Reads   reports/ledger-v0.1.jsonl       one row per output: split (ref = public human-reference tasks,
                                        dev = public synthetic tasks, test = private), task, model, sample,
                                        scripted, floor, detectors, empty
        reports/calibration-v0.1.json   per detector: the threshold and the 801 human scores
Writes  paper/figures/fig-leaderboard.pdf   the score per model with its interval, public and held-out beside it
        paper/figures/fig-outcomes.pdf      where each model's outputs end (four outcomes, 100% per model)
        paper/figures/fig-detector.pdf      headline detector scores: human texts against model outputs
        paper/figures/fig-kinds.pdf         the score by kind of task
        paper/figures/figure_data.json      every plotted number and a fingerprint of the two files read,
                                            compared with the report by paper/check_numbers.py

An output passes when `scripted` is 1, `floor` is 1.0 and the headline detector's score is at or below its
threshold; an empty output fails. A task's score is the mean over its outputs and a model's score the mean over
tasks. Intervals use `ci` from humanbench.harness.report_v01 (bootstrap over tasks, 2,000 resamples, seed 0),
with tasks in ledger order, so they equal the intervals of reports/results-v0.1.md.

The protocol diagram (Figure 1) and the example task are set in LaTeX inside the paper, not here.
Figures are drawn at their printed size (one column: 3.42 in, full width: 7.1 in) with 9 pt text, the size of
the paper's captions, so nothing is scaled when the paper includes them.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from humanbench.harness.report_v01 import HEADLINE, ci  # noqa: E402  (the report's own bootstrap)

LEDGER = REPO / "reports" / "ledger-v0.1.jsonl"
CALIBRATION = REPO / "reports" / "calibration-v0.1.json"

# --- one palette for every figure -------------------------------------------------------------------------
# Checked with the dataviz palette validator against a white page: passed and flagged as a two-colour pair
# (lightness band, chroma, colour-blind separation 27.5, contrast above 3:1), the two greys as an ordinal ramp
# (monotone lightness, light end 2:1 against the page). Printed in greyscale the four sit at L* 74, 36, 56
# and 35 in stacking order, so neighbours in a stacked bar differ by 20 or more; the two dark ones never touch.
PASSED = "#274fa3"      # ink blue
FLAGGED = "#c4710f"     # amber
QUALITY = "#50555c"     # dark grey: failed quality review
CHECKS = "#b3b7bd"      # light grey: failed task checks
INK, MUTED, GRID = "#111111", "#4a4a4a", "#d9d9d9"
OUTCOMES = (("failed task checks", CHECKS, INK), ("failed quality review", QUALITY, "white"),
            ("flagged as AI", FLAGGED, INK), ("passed", PASSED, "white"))

DISPLAY = {
    "claude-fable-5-2": "Claude Fable 5.2", "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5", "claude-haiku-5-5": "Claude Haiku 5.5",
    "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6.1-terra": "GPT-6.1 Terra",
    "gpt-6-astra": "GPT-6 Astra", "gpt-6-luna": "GPT-6 Luna",
    "deepseek-v4p1-flash-1017": "DeepSeek V4.1 Flash", "kimi-k3p1-pro": "Kimi K3.1 Pro",
    "minimax-m3p1-flash": "MiniMax M3.1 Flash",
}
KINDS = (("ref", "Human-reference tasks", "public"), ("dev", "Synthetic tasks", "public"),
         ("test", "Synthetic tasks", "held-out"))
COLUMN, PAGE, PT = 3.42, 7.1, 9.0


def setup_fonts() -> None:
    """Latin Modern Roman 9, the face of the paper's captions; Computer Modern from matplotlib if it is absent."""
    found = sorted((HERE / "fonts").glob("lmroman9-*.otf"))
    for f in found:
        font_manager.fontManager.addfont(str(f))
    family = "Latin Modern Roman" if found else "cmr10"
    family = next((f.name for f in font_manager.fontManager.ttflist if f.name.startswith("Latin Modern Roman")), family)
    plt.rcParams.update({
        "font.family": family, "font.size": PT, "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.labelsize": PT, "xtick.labelsize": PT, "ytick.labelsize": PT, "legend.fontsize": PT,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.5, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 0,
        "pdf.fonttype": 42, "savefig.pad_inches": 0.01, "figure.dpi": 200,
    })


def load() -> tuple[list[dict], dict]:
    rows = [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").split("\n") if x.strip()]
    return rows, json.loads(CALIBRATION.read_text(encoding="utf-8"))


def outcome(r: dict, cut: float) -> str:
    """The first gate an output failed, or passed. An empty output fails AI-text detector if it got that far."""
    if not r["scripted"]:
        return "failed task checks"
    if r["floor"] != 1.0:
        if r["floor"] is None:
            raise SystemExit(f"unscored output in the ledger: {r['model']} {r['task']} sample {r['sample']}")
        return "failed quality review"
    score = r["detectors"].get(HEADLINE)
    return "passed" if (not r["empty"]) and score is not None and score <= cut else "flagged as AI"


def by_card(rows: list[dict], model: str, splits: tuple[str, ...], cut: float) -> dict[str, list[float]]:
    """task -> [1.0 if the output passed], tasks in the order the report pools them (ref, dev, test; ledger order)."""
    out: dict[str, list[float]] = defaultdict(list)
    for split in splits:
        for r in rows:
            if r["model"] == model and r["split"] == split:
                out[r["task"]].append(float(outcome(r, cut) == "passed"))
    return dict(out)


def compute(rows: list[dict], cal: dict) -> dict:
    cut = cal[HEADLINE]["cutoff"]
    models = sorted({r["model"] for r in rows})
    parts = {"overall": ("ref", "dev", "test"), "public": ("ref", "dev"), "private": ("test",),
             "ref": ("ref",), "dev": ("dev",), "test": ("test",)}
    data = {"sources": {str(f.relative_to(REPO)): hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                        for f in (LEDGER, CALIBRATION)},
            "cutoff": cut, "models": {}, "cards": {k: len({r["task"] for r in rows if r["split"] in v})
                                                   for k, v in parts.items()}}
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        counts = Counter(outcome(r, cut) for r in mine)
        data["models"][m] = {
            "name": DISPLAY[m],
            **{k: [100 * x for x in ci(by_card(rows, m, v, cut))] for k, v in parts.items()},
            "outcomes": {name: 100 * counts[name] / len(mine) for name, _, _ in OUTCOMES},
            "answers": len(mine),
        }
    data["order"] = sorted(models, key=lambda m: (-data["models"][m]["overall"][0], m))
    reached = [r for r in rows if r["scripted"] and r["floor"] == 1.0 and not r["empty"]]
    scores = [r["detectors"][HEADLINE] for r in reached]
    humans = cal[HEADLINE]["human_scores"]
    data["detector"] = {
        "label": cal[HEADLINE]["label"], "humans": len(humans),
        "humans_at_or_below": sum(s <= cut for s in humans),
        "answers_reaching_gate": len(scores), "answers_at_or_below": sum(s <= cut for s in scores),
        "human_median": sorted(humans)[len(humans) // 2], "answer_median": sorted(scores)[len(scores) // 2],
    }
    return data, humans, scores


def axes_in(fig, left: float, bottom: float, width: float, height: float):
    """An axes placed in inches, so a figure keeps its printed size whatever its labels are."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


LABELS = 1.29   # inches kept at the left for the model names
FOOT = 0.43     # inches kept under the plot for the tick labels and the axis label


def rows_axis(ax, data: dict, labels: bool = True) -> list[float]:
    """Models top to bottom in leaderboard order; returns the y position of each."""
    order = data["order"]
    ys = list(range(len(order)))
    ax.set_ylim(len(order) - 0.45, -0.55)
    ax.set_yticks(ys)
    ax.set_yticklabels([data["models"][m]["name"] for m in order] if labels else [])
    ax.tick_params(axis="y", pad=4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    return ys


def percent_axis(ax, top: float, step: int, low: float = 0.0) -> None:
    ax.set_xlim(low, top)
    ax.set_xticks(list(range(0, int(top) + 1, step)))
    ax.spines["bottom"].set_bounds(0, max(range(0, int(top) + 1, step)))


def interval_dot(ax, y: float, mean: float, lo: float, hi: float, size: float = 6.2) -> None:
    ax.plot([lo, hi], [y, y], color=PASSED, linewidth=1.5, solid_capstyle="butt", zorder=3)
    ax.plot([mean], [y], marker="o", markersize=size, color=PASSED, markeredgecolor="white",
            markeredgewidth=1.0, zorder=5, clip_on=False)


def fig_leaderboard(data: dict) -> None:
    height, key = 2.74, 0.27
    fig = plt.figure(figsize=(COLUMN, height))
    ax = axes_in(fig, LABELS, FOOT, COLUMN - LABELS - 0.03, height - FOOT - key)
    ys = rows_axis(ax, data)
    percent_axis(ax, 33.5, 10, low=-0.8)
    off = 0.27
    small = dict(markersize=4.6, markeredgecolor=INK, markeredgewidth=0.8)
    for y, m in zip(ys, data["order"]):
        d = data["models"][m]
        interval_dot(ax, y, *d["overall"])
        ax.plot([d["public"][0]], [y - off], marker="v", color="white", zorder=4, **small)
        ax.plot([d["private"][0]], [y + off], marker="^", color=INK, zorder=4, **small)
        ax.text(max(d["overall"][2], d["public"][0], d["private"][0]) + 0.9, y, f"{d['overall'][0]:.1f}",
                va="center", ha="left")
    c = data["cards"]
    handles = [Line2D([], [], color=PASSED, linewidth=1.5, marker="o", markersize=6.2, markeredgecolor="white",
                      markeredgewidth=1.0, label=f"all {c['overall']} tasks, 95% interval"),
               Line2D([], [], linestyle="none", marker="v", color="white", label="public", **small),
               Line2D([], [], linestyle="none", marker="^", color=INK, label="held-out", **small)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
               handlelength=1.5, handletextpad=0.4, columnspacing=1.1, borderaxespad=0.0, borderpad=0.1)
    ax.set_xlabel("HumanBench score (% of outputs)")
    fig.savefig(HERE / "fig-leaderboard.pdf")
    plt.close(fig)


def fig_outcomes(data: dict) -> None:
    height, key = 2.89, 0.42
    fig = plt.figure(figsize=(COLUMN, height))
    ax = axes_in(fig, LABELS, FOOT, COLUMN - LABELS - 0.09, height - FOOT - key)
    ys = rows_axis(ax, data)
    ax.grid(False)
    percent_axis(ax, 100, 25)
    for y, m in zip(ys, data["order"]):
        left = 0.0
        for name, colour, text in OUTCOMES:
            w = data["models"][m]["outcomes"][name]
            ax.barh(y, w, left=left, height=0.68, color=colour, edgecolor="white", linewidth=1.1, zorder=3)
            if w >= 9.5:  # a two-digit label fits with room on both sides; thinner segments carry none
                ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", color=text, zorder=4)
            left += w
    fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label={"failed task checks": "failed checks", "failed quality review": "rejected by judge", "passed": "passed all three"}.get(n, n)) for n, c, _ in OUTCOMES], ncol=2,
               loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, handlelength=1.1, handleheight=0.8,
               handletextpad=0.45, columnspacing=1.6, labelspacing=0.25, borderaxespad=0.0, borderpad=0.1)
    ax.set_xlabel("Share of each model’s outputs (%)")
    fig.savefig(HERE / "fig-outcomes.pdf")
    plt.close(fig)


def fig_detector(data: dict, humans: list[float], scores: list[float]) -> None:
    cut, d = data["cutoff"], data["detector"]
    width = math.log10(1 / cut) / 6          # six bins from the cutoff to 1, so the cutoff is a bin edge
    edges = [cut * 10 ** (k * width) for k in range(-20, 7)]
    height, left, panel, title, gap = 2.62, 0.47, 0.84, 0.2, 0.09
    fig = plt.figure(figsize=(COLUMN, height))
    bottom = axes_in(fig, left, FOOT, COLUMN - left - 0.06, panel)
    top = axes_in(fig, left, FOOT + panel + title + gap, COLUMN - left - 0.06, panel)

    def shares(values: list[float]) -> list[float]:
        counts = [0] * (len(edges) - 1)
        for v in values:
            k = sum(v > e for e in edges[1:-1])   # (lower, upper] bins: a score at the cutoff is below it
            counts[k] += 1
        return [100 * c / len(values) for c in counts]

    below_h = 100 * d["humans_at_or_below"] / d["humans"]
    below_a = 100 * d["answers_at_or_below"] / d["answers_reaching_gate"]
    for ax, values, name in ((top, humans, f"{d['humans']} human texts"),
                             (bottom, scores, f"{d['answers_reaching_gate']:,} model outputs that reached this gate")):
        h = shares(values)
        colours = [MUTED if ax is top else (PASSED if edges[k + 1] <= cut * (1 + 1e-9) else FLAGGED)
                   for k in range(len(h))]
        ax.bar(edges[:-1], h, width=[edges[k + 1] - edges[k] for k in range(len(h))], align="edge",
               color=colours, edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(edges[0], 1.0)
        ax.set_ylim(0, 33)
        ax.set_yticks([0, 10, 20, 30])
        ax.tick_params(axis="y", pad=3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.plot([cut, cut], [0, 31], color=INK, linewidth=0.9, zorder=4)
        ax.set_xticks([0.001, 0.01, 0.1, 1.0])
        ax.set_xticklabels(["0.001", "0.01", "0.1", "1"] if ax is bottom else [])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.text(0.0, 1.03, name, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")
    top.text(cut * 0.88, 24, f"{below_h:.0f}% at or below\nthe threshold", ha="right", va="center", linespacing=1.15)
    top.text(cut * 1.13, 24, f"threshold\n{cut:.3f}", ha="left", va="center", linespacing=1.15)
    bottom.legend(handles=[Patch(facecolor=FLAGGED, edgecolor="none", label=f"{100 - below_a:.0f}% above: flagged as AI"),
                           Patch(facecolor=PASSED, edgecolor="none", label=f"{below_a:.0f}% at or below: passed")],
                  loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False, handlelength=1.0, handleheight=0.8,
                  handletextpad=0.45, labelspacing=0.3, borderaxespad=0.0, borderpad=0.0)
    bottom.set_xlabel(f"{d['label']} score (log scale)")
    fig.text(0.012, (FOOT + panel + (title + gap) / 2) / height, "Share of texts (%)", rotation=90, ha="left",
             va="center")
    fig.savefig(HERE / "fig-detector.pdf")
    plt.close(fig)


def fig_kinds(data: dict) -> None:
    height, head, gap = 2.44, 0.36, 0.2
    panel = (PAGE - LABELS - 0.05 - 2 * gap) / 3
    fig = plt.figure(figsize=(PAGE, height))
    for i, (key, kind, part) in enumerate(KINDS):
        ax = axes_in(fig, LABELS + i * (panel + gap), FOOT, panel, height - FOOT - head)
        ys = rows_axis(ax, data, labels=i == 0)
        percent_axis(ax, 40, 10, low=-1.2)
        for y, m in zip(ys, data["order"]):
            interval_dot(ax, y, *data["models"][m][key], size=5.6)
        ax.text(0.03, 1.125, kind, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")
        ax.text(0.03, 1.025, f"{data['cards'][key]} {part} tasks", transform=ax.transAxes, ha="left", va="bottom")
        if i == 1:
            ax.set_xlabel("HumanBench score on these tasks (% of outputs), with 95% interval")
    fig.savefig(HERE / "fig-kinds.pdf")
    plt.close(fig)


def main() -> None:
    setup_fonts()
    rows, cal = load()
    data, humans, scores = compute(rows, cal)
    fig_leaderboard(data)
    fig_outcomes(data)
    fig_detector(data, humans, scores)
    fig_kinds(data)
    (HERE / "figure_data.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    for m in data["order"]:
        d = data["models"][m]
        print(f"{d['name']:22s} {d['overall'][0]:5.1f} [{d['overall'][1]:.1f}, {d['overall'][2]:.1f}]  "
              + "  ".join(f"{d['outcomes'][n]:5.1f}" for n, _, _ in OUTCOMES))
    print("wrote", ", ".join(sorted(p.name for p in HERE.glob("fig-*.pdf"))), "and figure_data.json")


if __name__ == "__main__":
    main()
