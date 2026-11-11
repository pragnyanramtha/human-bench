"""HumanBench v0.1 results: one leaderboard (one score over all tasks, its public and held-out parts), then every gate by kind of task.

    uv run python -m humanbench.harness.report_v01 --run-id RUN [--run-id RUN2] --out reports/results-v0.1.md

Nothing stored in a trial record is trusted except the raw model output:
  programmatic checks   recomputed now, against the task as it is (score_output);
  LLM judge     private/trials/<run>/floor.jsonl, used only if it was made for this exact output and
                    this exact LLM-judge question (task, hidden lists, judge prompt version);
  detectors         private/judge_cache/<detector>.jsonl, joined by the hash of the scored text.
A record made from an older version of its task's prompt is stale and left out (and counted). A missing or
invalid score is "unscored": never a pass, never a fail, counted and shown, and a score with unscored
outputs is printed as incomplete.

Every task is scored the same way:
  an output reads as human to a detector when its score is at or below the threshold that leaves 1% of the
  human calibration texts above it.   an output counts = programmatic checks x LLM judge x reads as human
The leaderboard pools task sets: the HumanBench score is taken over every task, public and held-out together;
beside it are the scores on the public tasks (tasks/ref and tasks/dev) and on the held-out tasks
(leaderboard, pooled).
Two kinds of task:
  synthetic task (tasks/dev and the held-out tasks): the reference solution is model-written, so there is no
    human-reference claim, and no task is dropped because the detector flags its reference solution (selecting
    on that would keep only the tasks where model text slips past the detector); the count is reported.
  human-reference tasks (tasks/ref): the human reference reads as human to the headline detector (a task whose
    human reference the headline detector flags is left out and listed), so the model is asked to clear a bar a
    person cleared on the same content. A task is used only if it passed the memorisation probe for every
    launch model (humanbench.refs). Also shown for these tasks: how often the detector scores the model's
    piece as more human than the human reference (tie = half). That asks the model to beat the human reference, not to
    match it; another human text in the model's seat wins about half the time (the human baseline section).

The headline detector is fixed (HEADLINE). Other detectors are shown beside it and never substitute for it.
Intervals: 95% bootstrap over tasks, 2,000 resamples. Rates are means of task means, except the LLM-judge
column, which is the share of outputs passing the programmatic checks that also are accepted by the LLM judge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from humanbench import refs
from humanbench.harness.cards import fingerprint, load_tasks, score_output, split_dir
from humanbench.scoring import floor as floor_mod

REPO = Path(__file__).resolve().parents[3]
PRIV = REPO / "private"
CACHE = PRIV / "judge_cache"
DETECTORS = {  # name -> (label, human calibration file)
    "editlens-llama-3.2-3b": ("Open Pangram large", "humans"),  # as released (bf16), scored on a GPU: gpu_score.py
    "editlens-roberta-large": ("Open Pangram small", "humans"),
    "vanguard-modernbert-large": ("vanguard", "humans"),
    "tropa-mini-deberta-large": ("tropa-mini", "humans"),
}
HEADLINE = "editlens-llama-3.2-3b"
FPR = 0.01
CALIBRATION = REPO / "reports" / "calibration-v0.1.json"  # written by scripts/ledger.py: the published cutoffs


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()] if path.is_file() else []


def cutoff(scores: list[float]) -> float | None:
    """The score that leaves at most 1% of the human texts above it (None with under 100 humans)."""
    if len(scores) < 100:
        return None
    return sorted(scores, reverse=True)[math.floor(FPR * len(scores))]


def ci(by_task: dict[str, list[float]], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of task means, with a bootstrap interval over tasks."""
    means = [sum(v) / len(v) for v in by_task.values() if v]
    if not means:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(n))
    return sum(means) / len(means), boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


def ratio_ci(by_task: dict[str, tuple[int, int]], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """passes / eligible over all outputs, with a bootstrap interval over tasks."""
    items = list(by_task.values())
    den = sum(d for _, d in items)
    if not den:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        pick = [rng.choice(items) for _ in items]
        d = sum(x[1] for x in pick)
        boots.append(sum(x[0] for x in pick) / d if d else 0.0)
    boots.sort()
    return sum(p for p, _ in items) / den, boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


def pct(m: float, lo: float, hi: float) -> str:
    return "–" if math.isnan(m) else f"{100 * m:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


class Judge:
    def __init__(self) -> None:
        self.scores = {d: {r["sha"]: r["score"] for r in rows(CACHE / f"{d}.jsonl")
                           if isinstance(r.get("score"), (int, float)) and math.isfinite(r["score"])}
                       for d in DETECTORS}
        self.humans = {name: rows(CACHE / "in" / f"{name}.jsonl") for name in {v[1] for v in DETECTORS.values()}}
        self.cut, self.n_h, self.n_pool = {}, {}, {}
        # The human calibration texts cannot be redistributed. A checkout without them scores against the
        # published cutoffs, which hold only for detector scores made the same way (see the README).
        published = json.loads(CALIBRATION.read_text()) if CALIBRATION.is_file() else {}
        self.published = []  # detectors whose cutoff is the published one
        for d, (_, pool) in DETECTORS.items():
            if not self.humans[pool] and d in published:
                self.cut[d] = published[d]["cutoff"]
                self.n_h[d] = self.n_pool[d] = published[d]["humans_scored"]
                self.published.append(d)
                continue
            hs = [self.scores[d][sha(h["text"])] for h in self.humans[pool] if sha(h["text"]) in self.scores[d]]
            self.n_h[d], self.n_pool[d] = len(hs), len(self.humans[pool])
            # A cutoff from part of the pool would move as scoring continues: wait for the whole pool.
            self.cut[d] = cutoff(hs) if len(hs) == len(self.humans[pool]) else None

    def score(self, d: str, text: str) -> float | None:
        return self.scores[d].get(sha(text))

    def human(self, d: str, text: str) -> bool | None:
        s = self.score(d, text)
        return None if s is None or self.cut[d] is None else s <= self.cut[d]


def reference_judge_text(pkg) -> str | None:
    """The reference in the form the detectors read it (same preparation as a model output)."""
    from humanbench.envs import OracleRun

    od = PRIV / "oracle" / pkg.card.id
    if (od / "actions.json").is_file():
        return pkg.scoring_text(run=OracleRun.model_validate_json((od / "actions.json").read_text()))
    if (od / "draft.md").is_file():
        return pkg.scoring_text(text=(od / "draft.md").read_text())
    return None


def human_baseline(tasks: dict, judge: Judge) -> list[str]:
    """Other real human texts in the model's seat: how often do they win the pair against the reference?

    For each eligible reference task, every human calibration text from the same channel (other than the
    reference itself) is paired with the reference. These are different pieces on different subjects, not
    a second person answering the same request: the control shows that human text is not penalised for
    being "not the reference", and whether one source of text is systematically favoured over another."""
    mf = REPO / "tasks" / "ref" / "MANIFEST.json"
    source = {c["id"]: c.get("source") for c in json.loads(mf.read_text())["cards"]} if mf.is_file() else {}
    humans = rows(CACHE / "in" / "humans.jsonl")
    cards = {s: p for s, p in tasks.items() if refs.eligible(s)[0]}
    lines = ["| detector | other human text, same source as the reference | other human text, other sources in the channel |",
             "|---|---|---|"]
    for d, (label, _) in DETECTORS.items():
        same, other = {}, {}
        for slot, pkg in cards.items():
            ref = reference_judge_text(pkg)
            rs = judge.score(d, ref) if ref else None
            if rs is None:
                continue
            for h in humans:
                if h["channel"] != pkg.card.channel.value or sha(h["text"]) == sha(ref):
                    continue
                hs = judge.score(d, h["text"])
                if hs is not None:
                    bucket = same if h.get("source") == source.get(slot) else other
                    bucket.setdefault(slot, []).append(1.0 if hs < rs else 0.5 if hs == rs else 0.0)
        lines.append(f"| {label} | {pct(*ci(same))} ({sum(map(len, same.values()))} pairs) | "
                     f"{pct(*ci(other))} ({sum(map(len, other.values()))} pairs) |")
    return lines


FAILURE_KINDS = (  # column -> prefixes of the check names that count towards it
    ("formatting the venue does not show", ("verifier:format",)),
    ("assistant talk or a placeholder", ("verifier:wrapper",)),
    ("length", ("verifier:length",)),
    ("copied text", ("verifier:copy", "verifier:material_copy")),
    ("workspace routing", tuple(f"agent:{c}" for c in ("required_writes", "targets", "write_count", "reply_item",
                                                       "delivery", "unparseable"))),
    ("a private detail or an invented amount", ("agent:private_leak", "agent:invented_amounts")),
    ("required or barred words", ("verifier:must_include", "verifier:must_not_include")),
)
ROUTING_CHECKS = tuple(x.split(":")[1] for x in dict(FAILURE_KINDS)["workspace routing"])  # shared with other reports


WAIVABLE = ("formatting the venue does not show", "workspace routing")  # kinds the report bounds the cost of


def failure_kinds(res: dict) -> set[str]:
    """Which kinds of programmatic check an output failed (an output can fail several)."""
    out = set()
    for f in [*res["failures"], *res.get("task_failures", [])]:
        kind = next((name for name, pre in FAILURE_KINDS if f.startswith(pre)), "other")
        out.add(kind)
    return out


def floor_validation(tasks: dict[str, dict]) -> list[str]:
    """Per LLM judge: how many current references it passes, and what junk it rejected.

    Evidence from `floor_run validate`, kept per judge. A reference counts only if the stored verdict is for
    the floor question as it is now (task, hidden lists, prompt and the reference itself)."""
    from humanbench.harness.floor_run import _oracle_text

    lines = ["", "## LLM-judge validation", "",
             "Each LLM judge was given every task's reference (which must pass) and junk (which must fail): filler, "
             "a refusal and another task's output on 20 tasks of each task set, and the reference with a leak added on "
             "every task that has private items or a factual keep-out statement. A reference is counted only if the "
             "verdict is for the task and the reference as they are now.", "",
             "| judge | references passing, of the current ones | junk rejected | junk passed or unscored |",
             "|---|---|---|---|"]
    cards = {s: p for d in tasks.values() for s, p in d.items()}
    # One row per judge with a validation on record (private/floor/validate-<judge>.json and its verdict file).
    on_record = sorted(f for f in (PRIV / "floor").glob("validate-*.json")
                       if (f.parent / f.name.replace("validate-", "validated-", 1)).is_file())
    if not on_record:
        lines.append("| – | no validation on record | – | – |")
    for sf in on_record:
        vf = sf.parent / sf.name.replace("validate-", "validated-", 1)
        j = (json.loads(sf.read_text()).get("judges") or [sf.stem])[0]
        passed, stats = json.loads(vf.read_text()), json.loads(sf.read_text())["stats"]
        n = ok = 0
        for slot, pkg in cards.items():
            ref = _oracle_text(pkg, slot)
            if ref:
                n += 1
                ok += passed.get(slot) == floor_mod.prompt_id(floor_mod.build_prompt(pkg, ref))
        junk_ok = sum(v for k, v in stats.items() if k.startswith("junk") and k.endswith(":ok"))
        junk_bad = sum(v for k, v in stats.items() if k.startswith("junk") and not k.endswith(":ok"))
        lines.append(f"| {j.split(':')[-1]} | {ok} of {n} | {junk_ok} | {junk_bad} |")
    return lines


LOSSES = ("failed checks", "rejected by judge", "flagged as AI")


PUBLIC_SETS, PRIVATE_SETS = ("ref", "dev"), ("test",)
ALL_SETS = PUBLIC_SETS + PRIVATE_SETS


def pooled(summary: dict[str, dict], sets: tuple[str, ...], model: str) -> dict | None:
    """One model's results over several task sets taken as one set of tasks (task ids never repeat across
    sets). None if the model has no outputs there; `score` is None if any of its outputs is missing or unscored."""
    parts = [summary[k][model] for k in sets if model in summary.get(k, {})]
    if not parts:
        return None
    by_card = {t: v for x in parts for t, v in x["by_card"].items()}
    good = {t: v for x in parts for t, v in x["good_by_card"].items()}
    lost = Counter()
    for x in parts:
        lost.update(x["lost"])
    whole = all(x["score"] is not None for x in parts) and len(parts) == sum(bool(summary.get(k)) for k in sets)
    return {"score": ci(by_card) if whole and by_card else None, "quality": ci(good) if whole and good else None,
            "lost": dict(lost), "outputs": sum(lost.values())}


def leaderboard(summary: dict[str, dict]) -> dict:
    """The one leaderboard: every model's HumanBench score over all tasks, with its public and held-out parts
    beside it. Used by the results report, the ledger tables and the viewer."""
    cards = {k: summary[k]["_cards"] for k in summary}
    models = sorted({m for k in summary for m in summary[k] if not m.startswith("_")})
    rows_ = [{"model": m, "all": pooled(summary, ALL_SETS, m), "public": pooled(summary, PUBLIC_SETS, m),
              "private": pooled(summary, PRIVATE_SETS, m)} for m in models]

    def rank(r: dict) -> tuple:
        top = r["all"] and r["all"]["score"]
        return (-(top[0] if top else -1.0), r["model"])

    return {"rows": sorted(rows_, key=rank), "cards": cards,
            "public_cards": sum(cards.get(k, 0) for k in PUBLIC_SETS),
            "private_cards": sum(cards.get(k, 0) for k in PRIVATE_SETS)}


def score_table(summary: dict[str, dict]) -> list[str]:
    """The leaderboard, with what makes each score up. `summary` is filled by section() for each task set."""
    def cell(x) -> str:
        return "–" if x is None else "incomplete" if x.get("score") is None else pct(*x["score"])

    def quality(x) -> str:
        return "–" if x is None else "incomplete" if x.get("quality") is None else pct(*x["quality"])

    def stops(x) -> str:
        if x is None:
            return " | ".join("–" for _ in LOSSES)
        n = x["outputs"] or 1
        return " | ".join(f"{100 * x['lost'].get(k, 0) / n:.1f}%" for k in LOSSES)

    board = leaderboard(summary)
    cards = board["cards"]
    total = board["public_cards"] + board["private_cards"]
    lines = ["## The HumanBench score", "",
             "One benchmark, one score: **the share of a model's outputs that pass all three gates while "
             "doing the job**. An output counts only if it passes the programmatic checks, is accepted by an LLM judge, and is "
             "not flagged by the AI-text detector at the threshold that 99% of human-written texts stay "
             "under. Three samples are produced per task; a task's score is the share of its three outputs that "
             f"count, and a model's score is the average over all {total} tasks. Scores are low because nearly every "
             "output is flagged.", "",
             f"**{board['public_cards']} tasks are public**: {cards.get('ref', 0)} built backwards from a real person's "
             f"text (human-reference tasks) and {cards.get('dev', 0)} synthetic tasks. **{board['private_cards']} are held out**: "
             "synthetic tasks of the same kinds, held back so that no model can be trained or tuned on them. The "
             "score on each part is shown beside the overall score: the public part can be reproduced by anyone, and "
             "the held-out part shows whether it holds on tasks nobody has seen. The two parts are not expected to be "
             "equal, because the public part also contains the human-reference tasks, which are harder.", "",
             "**Answer quality** is the share of outputs that passed the "
             "programmatic checks and the LLM judge, over all tasks.", "",
             "| model | HumanBench score | public tasks | held-out tasks | answer quality |",
             "|---|---|---|---|---|"]
    for r in board["rows"]:
        lines.append(f"| {r['model']} | **{cell(r['all'])}** | {cell(r['public'])} | {cell(r['private'])} | "
                     f"{quality(r['all'])} |")
    lines += ["", "Where the other outputs stopped, over all tasks. With the score, each row adds up to 100%.", "",
              "| model | " + " | ".join(LOSSES) + " |", "|---|" + "---|" * len(LOSSES)]
    for r in board["rows"]:
        lines.append(f"| {r['model']} | {stops(r['all'])} |")
    kinds = [(k, label) for k, label in (("ref", "public, human-reference tasks"), ("dev", "public, synthetic tasks"),
                                         ("test", "held-out, synthetic tasks")) if summary.get(k)]
    if len(kinds) > 1:
        lines += ["", "The score by kind of task. A human-reference task has a real person's piece behind "
                  "it, which passes the same detector threshold; a synthetic task has a model-written reference "
                  "solution and covers more kinds of task (replies in a thread, workspace tasks, ghostwriting).", "",
                  "| model | " + " | ".join(f"{label} ({cards[k]} tasks)" for k, label in kinds) + " |",
                  "|---|" + "---|" * len(kinds)]
        for r in board["rows"]:
            m = r["model"]
            lines.append(f"| {m} | " + " | ".join(
                "–" if m not in summary[k] else "incomplete" if summary[k][m]["score"] is None
                else pct(*summary[k][m]["score"]) for k, _ in kinds) + " |")
    return [*lines, ""]


def judge_agreement(runs: list[str]) -> list[str]:
    """Outputs that two LLM judges both read (floor.jsonl and floor-second.jsonl): how often they agree."""
    pairs = []
    for run in runs:
        second = {(x["key"], x["sha"], x["prompt_id"]): x for x in rows(PRIV / "trials" / run / "floor-second.jsonl")}
        for x in rows(PRIV / "trials" / run / "floor.jsonl"):
            y = second.get((x["key"], x["sha"], x["prompt_id"]))
            if y and x["floor"] in (0.0, 1.0) and y["floor"] in (0.0, 1.0) and x["judges"] and y["judges"]:
                pairs.append((x["judges"][0], y["judges"][0], x["floor"], y["floor"], x["key"].split("|")[0]))
    if not pairs:
        return []
    lines = ["", "## The two LLM judges on the same outputs", "",
             "A judge never reads its own company's model, so Anthropic's models are judged by a different judge "
             "from everyone else. The open-weight models' outputs can be read by both, which shows how far the two "
             "judges differ on identical text.", "",
             "| judges | outputs read by both | first passes | second passes | both pass | both fail | "
             "only the first passes | only the second passes | agree |", "|---|---|---|---|---|---|---|---|---|"]
    for a, b in sorted({(p[0], p[1]) for p in pairs}):
        mine = [p for p in pairs if (p[0], p[1]) == (a, b)]
        n = len(mine)
        pp = sum(p[2] == 1.0 and p[3] == 1.0 for p in mine)
        ff = sum(p[2] == 0.0 and p[3] == 0.0 for p in mine)
        pf = sum(p[2] == 1.0 and p[3] == 0.0 for p in mine)
        fp = sum(p[2] == 0.0 and p[3] == 1.0 for p in mine)
        lines.append(f"| {a.split(':')[1]}, then {b.split(':')[1]} | {n} | {100 * (pp + pf) / n:.1f}% | "
                     f"{100 * (pp + fp) / n:.1f}% | {pp} | {ff} | {pf} | {fp} | {100 * (pp + ff) / n:.1f}% |")
    return lines


def reference_cards(tasks: dict, judge: Judge) -> tuple[dict, list[str]]:
    """The reference tasks a result may use, and a note for each group left out.

    One rule, used by this report and by scripts/ledger.py. A task is usable when it passed the memorisation
    probe for every model in the launch panel, and the headline detector has scored its human reference and
    reads it as human. A reference that is missing or not scored yet leaves the task out: the track's claim
    is that the human reference cleared the bar, so "unknown" cannot count as cleared."""
    note, active = [], dict(tasks)
    out = {s: refs.eligible(s)[1] for s in active if not refs.eligible(s)[0]}
    if out:
        note.append(f"{len(out)} tasks left out by the memorisation rule (" + ", ".join(
            f"{k} {n}" for k, n in sorted(Counter(v.split(': ')[1] for v in out.values()).items())) + ").")
    active = {s: p for s, p in active.items() if s not in out}
    verdict = {s: (judge.human(HEADLINE, t) if t else None) for s, t in
               ((s, reference_judge_text(p)) for s, p in active.items())}
    ai = sorted(s for s, v in verdict.items() if v is False)
    unknown = sorted(s for s, v in verdict.items() if v is None)
    if ai:
        note.append(f"{len(ai)} tasks left out because the headline detector reads their human reference as AI: "
                    + ", ".join(ai) + ".")
    if unknown:
        note.append(f"{len(unknown)} tasks left out because their human reference is missing or the headline detector "
                    "has not scored it: " + ", ".join(unknown) + ".")
    return {s: p for s, p in active.items() if verdict[s] is True}, note


def section(title: str, recs: list[dict], tasks: dict, judge: Judge, track: str,
            expect: dict[tuple[str, str], dict[str, int] | int] | None = None,
            collect: dict | None = None) -> list[str]:
    """One table for a task set, over every run given. Each record carries the run it came from (`_run`).

    `expect` maps (run, model) to the samples that run was asked for, task by task (from its invocation
    manifests, after any --max-sample cap; a bare number means that many for every usable task). A model
    is complete only when it has that many outputs for every usable task, whether or not the run produced
    anything for this task set; where a run has no manifest, the largest sample number seen stands in."""
    ref_track = track == "ref"
    expect = expect or {}
    floor = {(run, r["key"], r["sha"]): r for run in sorted({r["_run"] for r in recs})
             for r in rows(PRIV / "trials" / run / "floor.jsonl")}
    note, lines = [], [f"## {title}", ""]
    # Which cards count.
    active = dict(tasks)
    if ref_track:
        active, note = reference_cards(tasks, judge)
        panel = set(refs.launch_models())
        unscreened = sorted({r["model"] for r in recs} - panel)
        if unscreened:  # a model outside the panel was never asked to continue the human texts
            note.append("Left out, because they are not in the memorisation panel (private/refcards/launch_models.txt) "
                        "and so were not screened: " + ", ".join(unscreened) + ".")
            recs = [r for r in recs if r["model"] in panel]
    ref_text = {s: reference_judge_text(p) for s, p in active.items()}
    if not ref_track:
        ai_ref = sorted(s for s, t in ref_text.items() if t and judge.human(HEADLINE, t) is False)
        unk_ref = sorted(s for s, t in ref_text.items() if not t or judge.human(HEADLINE, t) is None)
        if not unk_ref:
            # Here every reference solution is model-written. Dropping the cards whose reference is read as AI
            # would keep exactly the tasks where model text slips past the detector and inflate every rate,
            # so all cards are kept and the count is only reported.
            note.append(f"The headline detector reads {len(ai_ref)} of the {len(ref_text)} model-written reference "
                        "solutions as AI. All tasks are kept: selecting on that would keep only the tasks where model "
                        "text slips past the detector.")
    stale = Counter()
    cur = []
    for r in recs:
        pkg = active.get(r["task"])
        if pkg is None:
            continue
        if r.get("card_fingerprint") != fingerprint(pkg):
            stale[r["model"]] += 1
            continue
        cur.append((r, pkg, score_output(pkg, r["output"])))
    seen = Counter()  # (run, model) -> largest sample number among its records, current or not
    for r in recs:
        seen[(r["_run"], r["model"])] = max(seen[(r["_run"], r["model"])], r["sample"])
    # How many outputs each (run, model) owes this card set: for every usable card the run's manifests name,
    # the number of samples they asked for. A run with records here but no manifest owes the largest sample
    # number seen, for every usable card.
    owed: dict[tuple[str, str], int] = {}
    for key, per_task in expect.items():
        if ref_track and key[1] not in refs.launch_models():
            continue
        per_task = {t: per_task for t in active} if isinstance(per_task, int) else per_task
        n = sum(k for t, k in per_task.items() if t in active)
        if n:
            owed[key] = n
    for key, k in seen.items():
        owed.setdefault(key, k * len(active))
    models = sorted({m for _, m in owed})
    in_run = sorted({r["task"] for r, _, _ in cur})
    lines += [f"{len(in_run)} of {len(active)} usable tasks have outputs; {len(cur)} outputs from "
              f"{len(models)} models." + (f" {sum(stale.values())} stale outputs (made from an older version of a "
                                           "task) were left out." if stale else ""), *note, ""]
    comp_head = ["| model | samples per task | outputs (missing) | pass programmatic checks | accepted by the LLM judge, of those | "
                 + " | ".join(f"reads as human: {DETECTORS[d][0]}" for d in DETECTORS) + " |",
                 "|---|---|---|---|---|" + "---|" * len(DETECTORS)]
    comp = []
    head = ("| model | " + " | ".join(("**" + DETECTORS[d][0] + " (headline)**") if d == HEADLINE else DETECTORS[d][0]
                                      for d in DETECTORS) + " | unscored |")
    allg, pair, why, waived, waived_open, short = [], [], {}, {}, Counter(), {}
    lost, good, scores, finals = defaultdict(Counter), {}, {}, {}
    for m in models:
        det_ok, fl = defaultdict(list), defaultdict(lambda: [0, 0])
        jd = {d: defaultdict(list) for d in DETECTORS}
        final = {d: defaultdict(list) for d in DETECTORS}   # all gates, with "reads as human"
        beat = {d: defaultdict(list) for d in DETECTORS}    # all gates, with "more human than the real piece"
        incomplete, unscored, pair_open = Counter(), Counter(), Counter()
        mine = [(r, p, s) for r, p, s in cur if r["model"] == m]
        for r, pkg, res in mine:
            task, text = r["task"], res["scored_text"]
            ok = res["deterministic"] == 1.0
            det_ok[task].append(float(ok))
            if not ok:
                for kind in failure_kinds(res):
                    why.setdefault(m, Counter())[kind] += 1
            why.setdefault(m, Counter())["outputs"] += 1
            f = None
            if ok:
                row = floor.get((r["_run"], f"{m}|{task}|s{r['sample']}", sha(text)))
                want = floor_mod.prompt_id(floor_mod.build_prompt(pkg, res["authored_text"]))
                if row is not None and row.get("prompt_id") == want and row.get("floor") in (0.0, 1.0):
                    f = row["floor"]
                    fl[task][0] += int(f == 1.0)
                    fl[task][1] += 1
                else:
                    unscored["floor"] += 1
            for d in DETECTORS:
                if not text.strip():
                    v = w = 0.0
                else:
                    h = judge.human(d, text)
                    v = None if h is None else float(h)
                    w = None
                    if ref_track:
                        ms, hs = judge.score(d, text), judge.score(d, ref_text[task] or "")
                        w = None if ms is None or hs is None else 1.0 if ms < hs else 0.5 if ms == hs else 0.0
                if v is None:
                    unscored[DETECTORS[d][0]] += 1
                else:
                    jd[d][task].append(v)
                if d == HEADLINE:
                    # The headline again, counting an output that failed nothing but one kind of check as if it
                    # had passed the checks and the floor (the floor is not asked about outputs that fail a check).
                    for kind in WAIVABLE:
                        only = not ok and failure_kinds(res) == {kind}
                        if v is None or (ok and f is None):
                            waived_open[(m, kind)] += 1  # not scored or not judged: the column is incomplete too
                        else:
                            waived.setdefault((m, kind), defaultdict(list))[task].append(
                                float(v == 1.0 and ((ok and f == 1.0) or only)))
                if d == HEADLINE:  # where this answer stopped, and whether it is a good answer whatever it reads like
                    lost[m]["failed checks" if not ok else "rejected by judge" if f == 0.0 else
                            "flagged as AI" if v == 0.0 else "not scored yet" if f is None or v is None else "passes"] += 1
                    if not ok or f is not None:
                        good.setdefault(m, defaultdict(list))[task].append(float(ok and f == 1.0))
                for gate, store, opened in ((v, final, incomplete), (w, beat, pair_open)):
                    if not ok or f == 0.0 or gate == 0.0:
                        store[d][task].append(0.0)  # one failed gate settles it
                    elif f is None or gate is None:
                        opened[d] += 1
                    else:
                        store[d][task].append(gate)
        expected = sum(v for (_, mm), v in owed.items() if mm == m)  # over every run the model is in
        samples = f"{expected / len(active):g}" if active else "0"
        missing = short[m] = expected - len(mine)  # against every usable card, with or without outputs
        comp.append((m, f"| {m} | {samples} | {len(mine)} ({missing}) | {pct(*ci(det_ok))} | "
                        f"{pct(*ratio_ci({t: (a, b) for t, (a, b) in fl.items()}))} | "
                        + " | ".join("unscored" if not jd[d] else f"incomplete ({unscored[DETECTORS[d][0]]} unscored)"
                                     if unscored[DETECTORS[d][0]] else pct(*ci(jd[d])) for d in DETECTORS) + " |"))
        for store, opened, table in ((final, incomplete, allg), (beat, pair_open, pair)):
            cells = []
            for d in DETECTORS:
                c = (f"incomplete ({missing} outputs missing)" if missing else
                     f"incomplete ({opened[d]} unscored)" if opened[d] else pct(*ci(store[d])))
                cells.append(f"**{c}**" if d == HEADLINE else c)
            rank = -1.0 if opened[HEADLINE] or missing else ci(store[HEADLINE])[0]
            if table is allg:
                scores[m] = None if rank < 0 else ci(store[HEADLINE])
                finals[m] = dict(store[HEADLINE])
            table.append((rank, m, f"| {m} | " + " | ".join(cells) + " | "
                          + (", ".join(f"{k} {v}" for k, v in sorted(unscored.items())) or "0") + " |"))
    order = [m for _, m, _ in sorted(allg, key=lambda x: (-x[0], x[1]))]  # best headline rate first
    if collect is not None:  # for the one-score table at the top of the report
        collect.update({m: {"score": scores.get(m), "quality": ci(good[m]) if m in good else None,
                            "lost": dict(lost[m]), "outputs": sum(lost[m].values()), "missing": short.get(m, 0),
                            "by_card": finals.get(m, {}), "good_by_card": dict(good.get(m, {}))}
                        for m in order})
        collect["_cards"] = len(active)
    sep = "|---|" + "---|" * len(DETECTORS) + "---|"
    lines += ["**Pass all gates** (programmatic checks x LLM judge x not flagged by the detector), by detector:", "", head, sep,
              *(row for _, _, row in sorted(allg, key=lambda x: order.index(x[1]))), "",
              "Each gate on its own:", "", *comp_head,
              *(row for _, row in sorted(comp, key=lambda x: order.index(x[0])))]
    kinds = [name for name, _ in FAILURE_KINDS] + ["other"]
    lines += ["", "Why outputs fail the programmatic checks (share of each model's outputs; one output can fail several). "
                  "Formatting means markdown the venue would show as typed, such as a bold title on a review; the "
                  "prompt names the venue but does not say that it is plain text. Copied text on human-reference tasks "
                  "means the output shares a 13-word run with the human reference, which the model never saw in the task. "
                  "The last columns are the score if an output that failed nothing but that kind of check "
                  "were counted as passing the programmatic checks and the LLM judge: an upper bound on what that kind costs. Workspace "
                  "routing is bounded this way because the output format for workspace tasks (a short To / Replying "
                  "to header) was introduced late and routing failures rose with it:", "",
              "| model | " + " | ".join(kinds) + " | " + " | ".join(f"score, {k} waived" for k in WAIVABLE) + " |",
              "|---|" + "---|" * (len(kinds) + len(WAIVABLE)),
              *("| " + m + " | " + " | ".join(f"{100 * why[m][k] / why[m]['outputs']:.1f}%" for k in kinds) + " | "
                + " | ".join("incomplete" if waived_open[(m, k)] or short.get(m) else
                             f"{100 * ci(waived[(m, k)])[0]:.1f}%" if (m, k) in waived else "–" for k in WAIVABLE)
                + " |" for m in order if m in why)]
    if ref_track:
        lines += ["", "**Against the human reference** (programmatic checks x LLM judge x the detector scores the model's "
                      "piece as more human than the human reference; tie = half). This asks the model to beat the "
                      "reference, not to match it: another human text in the model's seat wins about half the "
                      "time (see the human baseline), so 50% is parity.", "", head, sep,
                  *(row for _, _, row in sorted(pair, key=lambda x: order.index(x[1])))]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", action="append", required=True, help="repeat for several runs")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "results-v0.1.md")
    ap.add_argument("--max-sample", action="append", default=[], metavar="RUN[/MODEL-PREFIX]=K",
                    help="report only samples 1..K of that run, or of the models in it whose names start with the "
                         "prefix (while a later round is still running)")
    a = ap.parse_args()
    judge = Judge()
    tasks = {s: load_tasks([split_dir(s)]) for s in ("dev", "test", "ref")}
    lines = ["# HumanBench v0.1 research preview: results", "",
             "Generated by `python -m humanbench.harness.report_v01`. Read `README.md` for what these numbers do and "
             "do not mean: they measure task compliance, a minimum quality bar, and detectability under the named "
             "open AI-text detectors. They are not a measure of whether people can tell the text from human writing.", "",
             "<<SCORE>>",
             "## Detectors and their thresholds", "",
             "| detector | human texts scored / in the pool | threshold (higher = more AI) |", "|---|---|---|"]
    for d, (label, _) in DETECTORS.items():
        c = judge.cut[d]
        lines.append(f"| {label} (`{d}`){' (headline)' if d == HEADLINE else ''} | {judge.n_h[d]} / {judge.n_pool[d]} | "
                     f"{'not set yet' if c is None else f'{c:.4f}'} |")
    lines += ["", f"A threshold targets {FPR:.0%} of the human calibration texts flagged. It is a calibration on archival "
                  "text, not a validated false-positive rate. vanguard and tropa-mini read the whole text with "
                  "typography and whitespace normalised; Open Pangram small reads the first 512 tokens and Open Pangram "
                  "large the first 1,024, with Pangram's own preprocessing.",
              *(["", "The human calibration texts are not in this checkout, so the thresholds above are the published "
                 "v0.1 ones (`reports/calibration-v0.1.json`). They apply only to detector scores made the way the "
                 "published ones were: the same model revisions, on a GPU, in bfloat16."] if judge.published else []), "",
              "How to read the intervals: they are bootstrap intervals over tasks. An interval of [0.0, 0.0] means no "
              "win was observed in this sample, not that the true rate is zero.", "",
              "## Do the references read as human?", "",
              "The human references behind the human-reference tasks are part of the human calibration pool that sets the "
              "thresholds, so their row is not an independent validation of the detectors.", "",
              "| tasks | n | " + " | ".join(f"{DETECTORS[d][0]}" for d in DETECTORS) + " |",
              "|---|---|" + "---|" * len(DETECTORS)]
    for split, label in (("dev", "public, synthetic tasks (model-written reference solutions)"),
                         ("test", "held-out, synthetic tasks (model-written reference solutions)"),
                         ("ref", "public, human-reference tasks (the human references)")):
        texts = [t for t in (reference_judge_text(p) for p in tasks[split].values()) if t]
        cells = []
        for d in DETECTORS:
            v = [judge.human(d, t) for t in texts]
            v = [x for x in v if x is not None]
            cells.append(f"{sum(v)}/{len(v)} read as human" if v else "unscored")
        lines.append(f"| {label} | {len(texts)} | " + " | ".join(cells) + " |")
    lines += ["", "## Human baseline for the comparison with the human reference", "",
              "For the human-reference tasks, the report also gives how often a model's piece is rated as more human than "
              "the human reference. To show what a human scores on that scale, other real human texts are put in the model's "
              "seat: about half the time they win, so 50% is parity there. They are "
              "different pieces, not outputs to the same request, and they come from the same archival pool as the "
              "references, so this does not test old text against new text.", "",
              *human_baseline(tasks["ref"], judge)]
    score_at = lines.index("<<SCORE>>")
    del lines[score_at]
    recs, tracks, system_lines = [], set(), set()
    lines += ["", "## Runs", "", "Outputs were collected through each provider's hosted client or a hosted inference service, at each model's default reasoning effort unless a run says otherwise, with tools off and the fixed system line below; the client software may add framing text of its own around a request.", ""]
    caps = [(k.partition("/")[0], k.partition("/")[2], int(v)) for k, v in (x.split("=") for x in a.max_sample)]

    def limit(run: str, model: str) -> int:
        return min([k for r, prefix, k in caps if r == run and model.startswith(prefix)], default=10**6)

    expect: dict[tuple[str, str], dict[str, int]] = {}  # (run, model) -> {card: samples asked for}, after any cap
    for run in a.run_id:
        held = Counter()
        df = PRIV / "trials" / run / "dropped_models.txt"  # "model<TAB>reason": withdrawn from the run on purpose
        dropped = dict(x.split("\t", 1) for x in df.read_text().split("\n") if "\t" in x) if df.is_file() else {}
        for p in sorted((PRIV / "trials" / run / "raw").glob("*/*/s*.json")):
            r = json.loads(p.read_text())
            if r["sample"] > limit(run, r["model"]):
                held[r["model"]] += 1  # a later sample that is still being generated or scored
            elif not r.get("infra_failure"):
                recs.append({**r, "_run": run})
        tf = PRIV / "trials" / run / "track.txt"
        tracks.add(tf.read_text().strip() if tf.is_file() else "default")
        invs = [json.loads(x.read_text()) for x in sorted((PRIV / "trials" / run / "invocations").glob("*.json"))]
        for m in invs:
            for name in m.get("models") or list(m.get("routes") or {}):
                if name not in dropped:
                    k = min(int(m.get("samples") or 0), limit(run, name))
                    per_task = expect.setdefault((run, name), {})
                    for t in m.get("tasks") or []:
                        per_task[t] = max(per_task.get(t, 0), k)
        if invs:
            present = {r["model"] for r in recs if r["_run"] == run}
            effort = {k: v for m in invs for k, v in m.get("effort", {}).items() if k in present}
            system_lines.add(invs[-1].get("system_line", ""))
            lines += [f"- `{run}`: " + "Reasoning effort: "
                      + ", ".join(f"{k} {v}" for k, v in sorted(effort.items())) + "."
                      + (" Later samples of " + ", ".join(f"{m} (after sample {limit(run, m)})" for m in sorted(held))
                         + f" are left out: that round is not finished ({sum(held.values())} outputs so far)."
                         if held else "")
                      + ("".join(f" Withdrawn: {m}." for m, why in
                                 sorted(dropped.items())))]
    lines += ["", "System line, the same for every model: " + " / ".join(f'"{x}"' for x in sorted(system_lines))]
    track = "/".join(sorted(tracks))
    summary, body = {}, []
    for split, name in (("ref", "Public human-reference tasks"), ("dev", "Public tasks, synthetic tasks"),
                        ("test", "Held-out tasks, synthetic tasks")):
        sub = [r for r in recs if r["task"] in tasks[split]]
        if sub:
            summary[split] = {}
            body += ["", *section(f"{name} ({track} system line)", sub, tasks[split], judge, split, expect,
                                  summary[split])]
    lines[score_at:score_at] = score_table(summary)
    lines += ["", "# Details", "", "Everything below breaks the scores above down: each gate on its own, the other "
              "detectors, and checks on the scoring itself.", *body]
    lines += floor_validation(tasks)
    lines += judge_agreement(a.run_id)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
