"""Run the quality floor on trial outputs, or validate it on oracles and junk.

    uv run python -m humanbench.harness.floor_run validate          # oracles must pass, junk must fail
    uv run python -m humanbench.harness.floor_run run --run-id R [--when det_pass|all]

Judges use an API endpoint ("api:<model>") or OpenRouter
("openrouter:<vendor/model>", key in private/keys.txt). Votes are
cached in private/floor/votes.jsonl by a hash of (prompt version, judge, full prompt), so a
changed card or output is judged again and an unchanged one never is. Results for a run go to
private/trials/<run>/floor.jsonl: one row per output with its text hash, votes and verdict.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from humanbench.envs import OracleRun
from humanbench.scoring import floor
from humanbench.taskpkg import load_package

REPO = Path(__file__).resolve().parents[3]
PRIV = REPO / "private"
SPLITS = {"dev": REPO / "tasks" / "dev", "test": PRIV / "tasks" / "test", "ref": REPO / "tasks" / "ref"}
VOTES = PRIV / "floor" / "votes.jsonl"
# One judge per company; a judge never scores its own company's model (floor.eligible).
# Add "openrouter:<vendor/model>" for a third company when the OpenRouter account has credit.
def _private_routes():
    import importlib
    try:
        return importlib.import_module("humanbench.harness.cli_routes")
    except ModuleNotFoundError as exc:
        raise SystemExit("name a judge with --judges api:<model>; the working-repository routes are unavailable") from exc


def _default_judges():
    import importlib.util
    if importlib.util.find_spec("humanbench.harness.cli_routes") is not None:
        return _private_routes().DEFAULT_JUDGES
    return None


DEFAULT_JUDGES = None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Votes:
    def __init__(self) -> None:
        VOTES.parent.mkdir(parents=True, exist_ok=True)
        self.have: dict[str, dict] = {}
        if VOTES.is_file():
            for line in VOTES.read_text().split("\n"):
                if line.strip():
                    r = json.loads(line)
                    self.have[r["key"]] = r

    def put(self, key: str, judge: str, vote: dict) -> None:
        row = {"key": key, "judge": judge, **vote}
        self.have[key] = row
        with VOTES.open("a") as f:
            f.write(json.dumps(row) + "\n")


def _openrouter_key() -> str:
    for line in (PRIV / "keys.txt").read_text().split("\n"):
        if line.strip().startswith("sk-or-"):
            return line.strip()
    raise SystemExit("no OpenRouter key in private/keys.txt")


def _openrouter(model: str, prompt: str, timeout: float) -> str:
    import urllib.request

    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 400,
                       "messages": [{"role": "system", "content": floor.SYSTEM}, {"role": "user", "content": prompt}]})
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body.encode(),
                                 headers={"Authorization": f"Bearer {_openrouter_key()}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return str(d["choices"][0]["message"].get("content") or "")
    except Exception:  # noqa: BLE001  (a failed call is retried by the caller and never counted as a vote)
        return ""


async def _call(judge: str, prompt: str, timeout: float = 600) -> str:
    kind, model = judge.split(":", 1)
    if kind == "openrouter":
        return await asyncio.to_thread(_openrouter, model, prompt, min(timeout, 180))
    if kind == "api":
        from humanbench.harness.run_model import configured_call
        res = await asyncio.to_thread(configured_call, prompt, model, floor.SYSTEM, timeout)
        return res["text"] if res["ok"] else ""
    routes = _private_routes()
    return await routes.judge_call(judge, prompt, timeout)


def _sems(n: int) -> dict[str, asyncio.Semaphore]:
    """One queue per judge route, so a slow judge does not hold up the outputs another judge reads."""
    from collections import defaultdict
    return defaultdict(lambda: asyncio.Semaphore(n))


async def judge_piece(pkg, piece: str, judges: list[str], votes: Votes, sem: dict[str, asyncio.Semaphore]) -> tuple[float | None, list[str], list[dict]]:
    prompt = floor.build_prompt(pkg, piece)

    async def one(j: str) -> dict | None:
        key = floor.cache_key(prompt, j)
        if key in votes.have:
            return votes.have[key]
        async with sem[j.split(":", 1)[0]]:
            for attempt in range(3):
                v = floor.parse(await _call(j, prompt), floor.asks_privacy(pkg))
                if v is not None:
                    votes.put(key, j, v)
                    return votes.have[key]
                await asyncio.sleep(10 * (attempt + 1))
        return None

    judges = list(dict.fromkeys(judges))
    got = [v for v in await asyncio.gather(*(one(j) for j in judges)) if v]
    score, failed = floor.verdict(got, electorate=len(judges))
    return score, failed, got


def _packages(splits: list[str]) -> dict[str, object]:
    out = {}
    for s in splits:
        for d in sorted(SPLITS[s].iterdir()) if SPLITS[s].is_dir() else []:
            if d.is_dir() and not d.name.startswith("."):
                out[d.name] = load_package(d, require_oracle=False)
    return out


def _oracle_text(pkg, slot: str) -> str | None:
    """Everything the reference authored (subjects included), as the floor reads a model output."""
    od = PRIV / "oracle" / slot
    if (od / "actions.json").is_file():
        return pkg.authored_text(run=OracleRun.model_validate_json((od / "actions.json").read_text()))
    if (od / "draft.md").is_file():
        return pkg.authored_text(text=(od / "draft.md").read_text())
    return None


def judges_slug(judges: list[str]) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", ",".join(judges).lower()).strip("-")


async def validate(a: argparse.Namespace) -> None:
    """Every reference must pass; junk must fail. Junk: repeated filler, a refusal and another task's
    reference on `--junk` cards of each card set, and for every card with private items or keep-out
    statements, the reference plus a leak. Exits non-zero if any judgment is wrong or missing.

    The evidence is kept per judge, so one judge's pass is never read as another's. Two files in
    private/floor/, named after the judges given:
      validate-<judges>.json    counts by card set and kind, and every wrong or missing judgment;
      validated-<judges>.json   for each card whose reference passed, the identity of that floor question
                                (card, hidden lists, floor prompt version and the reference itself), so a
                                later change shows up as "not validated". A card judged in this run and
                                not passed is removed from it.
    The results report reads both and prints, per judge, how many current references have a passing
    verdict."""
    judges = list(dict.fromkeys(a.judges.split(",")))
    pkgs = _packages(a.splits.split(","))
    if a.skip_file:  # cards not in use yet (for example reference cards still in audit)
        skip = set(re.split(r"[\s,]+", Path(a.skip_file).read_text()))
        pkgs = {k: v for k, v in pkgs.items() if k not in skip}
    votes, sem = Votes(), _sems(a.concurrency)
    rng = random.Random(0)
    slots = sorted(pkgs)
    oracles = {s: _oracle_text(pkgs[s], s) for s in slots}
    missing = [s for s in slots if not oracles[s]]
    jobs = [(s, "oracle", oracles[s], 1.0) for s in slots if oracles[s]]
    have = [s for s in slots if oracles[s]]
    for split in sorted({pkgs[s].card.split for s in have}):  # the same number of cards from each card set
        mine = [s for s in have if pkgs[s].card.split == split]
        for s in rng.sample(mine, min(a.junk, len(mine))):
            lo = pkgs[s].card.words[0]
            other = rng.choice([o for o in have if o != s and pkgs[o].card.channel != pkgs[s].card.channel])
            jobs += [(s, "junk:filler", "banana " * lo, 0.0),
                     (s, "junk:refusal", "I'm sorry, but I can't help with this request. " * (lo // 9), 0.0),
                     (s, f"junk:other-task({other})", oracles[other], 0.0)]
    for s in have:
        env = pkgs[s].env
        priv = [i for i in env.items if i.private and i.body.strip()] if env is not None else []
        if priv:  # a workspace item marked private, quoted
            leak = " ".join(priv[0].body.split()[:60])
            jobs.append((s, "junk:leak-private-item", oracles[s] + "\n\nAlso, for what it's worth, this is what I was told privately: " + leak, 0.0))
        # Something the user asked to keep out, stated outright. Only statements that are themselves the fact
        # ("That Raff's father was in hospital") can be turned into a leak; a category ("the woman's identity")
        # reveals nothing when repeated, so it is not a test.
        facts = [x for x in pkgs[s].card.must_not_reveal if x.lower().startswith("that ")]
        for stmt in facts[:1]:
            jobs.append((s, "junk:leak-keep-out", oracles[s] + "\n\nOne more thing I should mention: "
                         + stmt[5:6].upper() + stmt[6:], 0.0))
    stats: dict[str, Counter] = {}
    wrong, validated, judged = [], {}, set()

    async def one(slot: str, kind: str, piece: str, want: float) -> None:
        score, failed, got = await judge_piece(pkgs[slot], piece, judges, votes, sem)
        ok = score == want
        name = kind.split("(")[0]
        stats.setdefault(pkgs[slot].card.split, Counter())[
            f"{name}:{'ok' if ok else 'unscored' if score is None else 'WRONG'}"] += 1
        if kind == "oracle":
            judged.add(slot)
            if ok:
                validated[slot] = floor.prompt_id(floor.build_prompt(pkgs[slot], piece))
        if not ok:
            wrong.append({"slot": slot, "kind": kind, "score": score, "failed": failed,
                          "reasons": [f"{v['judge']}: {v['reason']}" for v in got]})
        print(f"{slot} {kind}: {score} {failed}", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))
    slug = judges_slug(judges)
    out = PRIV / "floor" / f"validate-{slug}.json"
    total = Counter()
    for c in stats.values():
        total.update(c)
    out.write_text(json.dumps({"judges": judges, "splits": a.splits, "junk_cards_per_split": a.junk,
                               "floor_prompt_version": floor.PROMPT_VERSION, "stats": dict(total),
                               "stats_by_split": {k: dict(v) for k, v in sorted(stats.items())}, "wrong": wrong,
                               "no_reference": missing}, indent=1))
    vfile = PRIV / "floor" / f"validated-{slug}.json"
    old = json.loads(vfile.read_text()) if vfile.is_file() else {}
    kept = {k: v for k, v in old.items() if k not in judged}  # a card judged now keeps only what it earned now
    vfile.write_text(json.dumps({**kept, **validated}, indent=1, sort_keys=True))
    print(json.dumps(dict(total), indent=1), f"\nwrong or unscored: {len(wrong)}; cards without a reference: {len(missing)} "
                                             f"(details in {out})")
    if wrong or missing:
        raise SystemExit(1)


async def run(a: argparse.Namespace) -> None:
    """Judge every current output that passes the scripted checks (rescored now, against the card as it is).
    The floor reads everything the model authored, subjects included (`authored_text`)."""
    from humanbench.harness.cards import fingerprint, score_output

    run_dir = PRIV / "trials" / a.run_id
    pkgs = _packages(["dev", "test", "ref"])
    votes, sem = Votes(), _sems(a.concurrency)
    recs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("raw/*/*/s*.json"))]
    recs = [r for r in recs if not r.get("infra_failure") and r["task"] in pkgs
            and r.get("card_fingerprint") == fingerprint(pkgs[r["task"]])]
    if a.models:
        recs = [r for r in recs if r["model"] in set(a.models.split(","))]
    if a.max_sample:
        recs = [r for r in recs if r["sample"] <= a.max_sample]
    rows, stats = [], Counter()

    async def one(r: dict) -> None:
        pkg = pkgs[r["task"]]
        res = score_output(pkg, r["output"])
        if a.when != "all" and res["deterministic"] != 1.0:
            return
        piece = res["authored_text"]
        judges = floor.eligible(a.judges.split(","), r["model"])
        if not a.panel:
            judges = judges[:1]  # one judge per output: the first listed that is not from the model's company
        if piece.strip():
            score, failed, got = await judge_piece(pkg, piece, judges, votes, sem)
        else:
            score, failed, got = 0.0, ["floor:coherent"], []
        rows.append({"key": f"{r['model']}|{r['task']}|s{r['sample']}", "sha": _sha(res["scored_text"]),
                     "prompt_id": floor.prompt_id(floor.build_prompt(pkg, piece)), "floor": score, "failed": failed,
                     "judges": [v["judge"] for v in got], "reasons": [v["reason"] for v in got if v["reason"] != "ok"]})
        stats[f"{r['model']}:{score}"] += 1
        if sum(stats.values()) % 25 == 0:
            print(sum(stats.values()), dict(stats), flush=True)

    await asyncio.gather(*(one(r) for r in recs))
    # Merge into what is there: a run over some of the models must not drop the others' rows. A row for an
    # output this run looked at and did not judge again (it was replaced, or no longer passes the checks) is
    # removed, so the file never holds a verdict on text that is gone. Locked: several runs can finish together.
    import fcntl

    out = run_dir / a.out
    looked_at = {f"{r['model']}|{r['task']}|s{r['sample']}" for r in recs}
    with open(out.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = [json.loads(x) for x in out.read_text().split("\n") if x.strip()] if out.is_file() else []
        merged = {r["key"]: r for r in old if r["key"] not in looked_at} | {r["key"]: r for r in rows}
        tmp = out.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(merged[k]) + "\n" for k in sorted(merged)))
        tmp.replace(out)
    print(json.dumps(dict(sorted(stats.items())), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--splits", default="dev,test,ref")
    v.add_argument("--junk", type=int, default=20, help="cards of each card set that get the three junk pieces")
    v.add_argument("--skip-file", help="file of card ids to leave out (comma or whitespace separated)")
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True)
    r.add_argument("--when", choices=["det_pass", "all"], default="det_pass",
                   help="det_pass: only outputs that pass the scripted checks (the floor cannot change the others)")
    r.add_argument("--models", help="comma-separated model names: judge only their outputs")
    r.add_argument("--max-sample", type=int, help="judge only samples 1..K")
    r.add_argument("--out", default="floor.jsonl",
                   help="file in the run directory. The report reads floor.jsonl; a second judge's verdicts on the "
                        "same outputs go to floor-second.jsonl (with --judges naming that judge alone), and the "
                        "report then shows how often the two agree")
    r.add_argument("--panel", action="store_true",
                   help="ask every eligible judge and take the majority (a tie fails), instead of one judge per output")
    for s in (v, r):
        s.add_argument("--judges", default=DEFAULT_JUDGES)
        s.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    if not a.judges:
        a.judges = _default_judges()
    if not a.judges:
        ap.error("name a judge with --judges api:<model>")
    asyncio.run(validate(a) if a.cmd == "validate" else run(a))


if __name__ == "__main__":
    main()
