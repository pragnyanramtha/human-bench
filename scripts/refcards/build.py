"""Build reference-matched cards backwards from real human texts.

    uv run python scripts/refcards/build.py --per-channel 20 --concurrency 4

For each chosen human text (private/judge_eval/human.jsonl first, then private/corpus/*.jsonl) the runner
derives the request and notes a person could have given an assistant before writing it. The human text
itself becomes the card's oracle (private/oracle/<id>/draft.md, never published or shown to models).

A card is accepted only if, with no model in the loop:
  - the schema loads and the slot is allowed (channel, band, audience);
  - the human text scores 1.0 on the card's scripted checks, and empty / wrapper / echo outputs score 0.0;
  - the prompt and notes share no run of 8+ words with the human text (no pre-written prose leaks).
Accepted cards go to tasks/ref/<id>/, rejected attempts and reasons to private/refcards/.
The harsh audit, the judge check on the oracle and the quality floor run afterwards.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from humanbench.axes import ALLOWED_AUDIENCES, ALLOWED_BANDS, BAND_WORDS, Band, Channel, Mode
from humanbench.scoring.verifiers import _FORMAT_PATTERNS, FORMAT_POLICY, copied_share
from humanbench.taskpkg import load_package, nop_outputs

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRIV = REPO / "private"
OUT = REPO / "tasks" / "ref"
WORK = PRIV / "refcards"



def _working_generator():
    import importlib.util
    path = Path(__file__).with_name("build_private.py")
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("working_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def pool() -> list[dict]:
    rows = []
    for h in (json.loads(x) for x in (PRIV / "judge_eval" / "human.jsonl").read_text().split("\n") if x.strip()):
        rows.append({"src": f"judge_eval:{h['id']}", "channel": h["channel"], "text": h["text"], "source": h.get("source"),
                     "dataset": h.get("dataset"), "licence": h.get("licence"), "date": h.get("date"), "pilot_id": h["id"]})
    for f in sorted((PRIV / "corpus").glob("*.jsonl")):
        for r in (json.loads(x) for x in f.read_text().split("\n") if x.strip()):
            rows.append({"src": f"corpus:{r['id']}", "channel": r["channel"], "text": r["text"], "source": r.get("source"),
                         "dataset": r.get("repo"), "licence": r.get("licence"), "date": r.get("date"), "pilot_id": None})
    seen, out = set(), []
    for r in rows:  # the corpus repeats a few judge-eval texts
        key = " ".join(r["text"].split())[:300]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def band_for(words: int, channel: Channel) -> Band | None:
    for band, (lo, hi) in BAND_WORDS.items():
        if lo <= words <= hi and band in ALLOWED_BANDS[(channel, Mode.DIRECT)]:
            return band
    return None


def choose(per_channel: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in pool():
        ch = Channel(r["channel"])
        r["words"] = len(r["text"].split())
        r["band"] = band_for(r["words"], ch)
        if r["band"] is not None:
            by.setdefault(r["channel"], []).append(r)
    picked = []
    for ch, rows in sorted(by.items()):
        first = [r for r in rows if r["pilot_id"]]  # these already have pilot outputs
        rest = [r for r in rows if not r["pilot_id"]]
        rng.shuffle(first)
        rng.shuffle(rest)
        # Spread sources and bands: round-robin over (source, band) groups.
        groups: dict[tuple, list[dict]] = {}
        for r in first + rest:
            groups.setdefault((r["source"], r["band"]), []).append(r)
        order = []
        while any(groups.values()):
            for k in sorted(groups, key=str):
                if groups[k]:
                    order.append(groups[k].pop(0))
        for i, r in enumerate(order[:per_channel], 1):
            r["id"] = f"ref-{ch}-{i:02d}"
            picked.append(r)
    return picked


async def draft(prompt: str, model: str, effort: str, out: Path) -> dict | None:
    from humanbench.harness.run_model import configured_call
    legacy = _working_generator()
    if legacy is not None:
        return await legacy.working_draft(prompt, model, effort, out)
    res = await asyncio.to_thread(configured_call, prompt, model, "You are a helpful assistant.", 1500)
    try:
        return json.loads(res["text"]) if res["ok"] else None
    except json.JSONDecodeError:
        return None


def slug(role: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", role.lower()).strip("-")[:40] or "writer"


def assemble(r: dict, d: dict) -> dict:
    p = d["persona"]
    l2 = p["english"] == "L2"
    pid = f"{r['id']}-{slug(p['role'])}"
    overrides = {name: True for name, pat in _FORMAT_PATTERNS.items()
                 if not FORMAT_POLICY[Channel(r["channel"])][name] and pat.search(r["text"])}
    card = {
        "id": r["id"], "split": "ref", "channel": r["channel"], "mode": "direct", "tone": d["tone"],
        "audience": d["audience"], "stakes": d["stakes"], "band": r["band"].value, "persona_id": pid,
        "persona": {"id": pid, "role": p["role"], "age": max(13, min(100, p["age"])), "english": p["english"],
                    "first_language": (p["first_language"] or "unknown") if l2 else None, "lives": p["lives"],
                    "writing_level": p["writing_level"], "excluded_channels": []},
        "prompt": d["prompt"].strip(),
        "notes_pack": {k: [n.strip() for n in d["notes"][k] if n.strip()] for k in ("facts", "anecdotes", "opinions", "wants")},
        "must_include": [alts for alts in d["must_include"] if alts][:3],
        "must_not_include": [],
        "must_cover": [c.strip() for c in d["must_cover"] if c.strip()][:5],
        "format_overrides": overrides,
        "drafted_by": "gpt-6.1-sol, derived from a real human text (the oracle)",
    }
    return card


def validate(card: dict, text: str, tmp: Path) -> list[str]:
    problems = []
    if card["audience"] not in [a.value for a in ALLOWED_AUDIENCES[Channel(card["channel"])]]:
        problems.append(f"audience {card['audience']} not allowed for {card['channel']}")
    if not 2 <= len(card["must_cover"]) <= 5:
        problems.append("must_cover needs 2-5 points")
    notes = [n for v in card["notes_pack"].values() for n in v]
    if not 6 <= len(notes) <= 30:
        problems.append(f"{len(notes)} notes; need 6-30")
    if any(len(n.split()) > 40 for n in notes):
        problems.append("a note is over 40 words")
    if not 40 <= len(card["prompt"].split()) <= 260:
        problems.append("prompt length outside 40-260 words")
    share = copied_share(" \n ".join([card["prompt"], *notes]), [text], n=8)
    if share > 0:
        problems.append(f"prompt or notes copy runs of 8+ words from the human text (share {share:.2f})")
    if problems:
        return problems
    d = tmp / card["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True, width=110))
    try:
        pkg = load_package(d, require_oracle=False)
    except Exception as e:  # noqa: BLE001
        return [f"does not load: {str(e)[:300]}"]
    from humanbench import refs

    problems += refs.leakage(pkg, text)
    score, why = pkg.score(text=text)
    if score != 1.0:
        problems.append(f"the human text fails the card's own checks: {why}")
    for name, kwargs in nop_outputs(pkg):
        s, _ = pkg.score(**kwargs)
        if s != 0.0:
            problems.append(f"no-op output {name!r} passes the checks")
    return problems


async def build_one(r: dict, a: argparse.Namespace, sem: asyncio.Semaphore, stats: Counter) -> None:
    gone = any((WORK / d / r["id"]).is_dir() for d in ("dropped", "held_back"))  # already tried and removed
    gone = gone or (REPO / "private" / "tasks" / "ref_pending" / r["id"]).is_dir()  # built, waiting for its audit
    if gone or (OUT / r["id"] / "card.yaml").is_file() or (WORK / "rejected" / f"{r['id']}.json").is_file():
        stats["cached"] += 1
        return
    ch = Channel(r["channel"])
    base = ((HERE / "build_prompt.md").read_text()
            .replace("{CHANNEL}", r["channel"]).replace("{AUDIENCES}", ", ".join(x.value for x in ALLOWED_AUDIENCES[ch]))
            .replace("{WORDS}", str(r["words"])).replace("{TEXT}", r["text"].strip()))
    feedback, last = "", None
    async with sem:
        for attempt in range(3):
            d = await draft(base + feedback, a.model, a.effort, WORK / "raw" / f"{r['id']}.{attempt}.json")
            if d is None:
                continue
            last = d
            if not d["usable"]:
                break
            card = assemble(r, d)
            with tempfile.TemporaryDirectory() as tmp:
                problems = validate(card, r["text"], Path(tmp))
            if not problems:
                (OUT / r["id"]).mkdir(parents=True, exist_ok=True)
                (OUT / r["id"] / "card.yaml").write_text(
                    yaml.safe_dump(card, sort_keys=False, allow_unicode=True, width=110))
                od = PRIV / "oracle" / r["id"]
                od.mkdir(parents=True, exist_ok=True)
                (od / "draft.md").write_text(r["text"].strip() + "\n")
                (od / "provenance.json").write_text(json.dumps(
                    {k: r[k] for k in ("src", "source", "dataset", "licence", "date", "pilot_id", "words")}
                    | {"kind": "real human text (archival); the card was derived from it", "transformations": "none"}, indent=1))
                stats["accepted"] += 1
                print(f"{r['id']}: accepted ({r['words']}w, {r['source']})", flush=True)
                return
            feedback = ("\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED BY THE AUTOMATIC CHECKS:\n- " + "\n- ".join(problems)
                        + "\nFix exactly these and return the full JSON again.")
            last = {**d, "problems": problems}
    (WORK / "rejected").mkdir(parents=True, exist_ok=True)
    (WORK / "rejected" / f"{r['id']}.json").write_text(json.dumps({"src": r["src"], "last": last}, indent=1))
    stats["rejected" if last and last.get("usable") else "unusable" if last else "infra"] += 1
    print(f"{r['id']}: rejected ({(last or {}).get('reason') or (last or {}).get('problems')})"[:300], flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-channel", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    (WORK / "raw").mkdir(parents=True, exist_ok=True)
    rows = [r for r in choose(a.per_channel) if not a.only or r["id"] in a.only.split(",")]
    (WORK / "selection.json").write_text(json.dumps(
        [{k: r[k] for k in ("id", "src", "channel", "source", "words")} | {"band": r["band"].value} for r in rows], indent=1))
    print(f"{len(rows)} texts chosen:", dict(Counter(r["channel"] for r in rows)), flush=True)
    sem, stats = asyncio.Semaphore(a.concurrency), Counter()
    await asyncio.gather(*(build_one(r, a, sem, stats) for r in rows))
    print(dict(stats))


if __name__ == "__main__":
    asyncio.run(main())
