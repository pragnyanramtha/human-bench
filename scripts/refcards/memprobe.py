"""Memorisation probe for reference cards: can a model continue the real human text?

    uv run python scripts/refcards/memprobe.py --models api:gpt-6.1-sol

Each launch model is shown the first 40% of a reference text and asked to continue it word for word. The
continuation is compared with the hidden remainder, and a card is flagged for a model when either holds:
  - exact: they share a run of 12+ words, or three or more separate runs of 8+ words (one 8-word run can
    be a stock phrase such as "I look forward to hearing from you");
  - near-verbatim: at least 25% of the continuation's 5-word sequences occur in the remainder (catches
    reproduction with small substitutions that leaves no long exact run).
The continuation and the matched runs are stored (private/refcards/memprobe.json) so a flag can be read
and judged. The launch panel (private/refcards/launch_models.txt) is the set of models a card must be
clean for: humanbench.refs.eligible() excludes a card that any panel model flagged or was not probed on,
and the results report leaves a model that is not in the panel out of the cards built from real writing. Probing does
not change the panel; --panel add or --panel set does.

A clean probe means "no reproduction on request", not "absent from the model's training data".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PRIV = REPO / "private"
OUT = PRIV / "refcards" / "memprobe.json"
LAUNCH = PRIV / "refcards" / "launch_models.txt"
ASK = ("Below is the beginning of a piece of writing. If you recognise it, continue it exactly, word for word, "
       "from where it stops, for about {n} words. If you do not recognise it, continue it as best you can. "
       "Output only the continuation.\n\n{head}")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def shared_runs(a: list[str], b: list[str], n: int = 8) -> list[str]:
    """Maximal runs of n+ words that appear in both, longest first."""
    index: dict[tuple, list[int]] = {}
    for j in range(len(b) - n + 1):
        index.setdefault(tuple(b[j:j + n]), []).append(j)
    runs, i = [], 0
    while i <= len(a) - n:
        best = 0
        for j in index.get(tuple(a[i:i + n]), []):
            k = n
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
        if best:
            runs.append(" ".join(a[i:i + best]))
            i += best
        else:
            i += 1
    return sorted(runs, key=lambda r: -len(r.split()))


def compare(continuation: str, remainder: str) -> dict:
    a, b = tokens(continuation), tokens(remainder)
    runs = shared_runs(a, b)
    five = {tuple(b[i:i + 5]) for i in range(len(b) - 4)}
    grams = [tuple(a[i:i + 5]) for i in range(len(a) - 4)]
    near = sum(g in five for g in grams) / len(grams) if grams else 0.0
    longest = len(runs[0].split()) if runs else 0
    return {"longest_run": longest, "runs": runs[:5], "five_gram_share": round(near, 3),
            "flag": longest >= 12 or len(runs) >= 3 or near >= 0.25}


MIN_SHARE = 0.3  # a reply shorter than this share of the words asked for is not a continuation


def model_name(spec: str) -> str:
    """'claude:claude-sonnet-5-5' -> 'claude-sonnet-5-5'; a route prefix is optional in the working repository."""
    return spec.partition(":")[2].split("/")[-1] if ":" in spec else spec


async def main() -> None:

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True,
                    help="comma-separated api:<model> names (additional routes in the working repository)")
    ap.add_argument("--only", help="comma-separated card ids (default: every card in tasks/ref)")
    ap.add_argument("--panel", choices=["keep", "add", "set"], default="keep",
                    help="what to do with the launch panel (private/refcards/launch_models.txt), the set of models "
                         "a reference card must be clean for. keep: probe only, leave the panel alone (default). "
                         "add: add these models to it. set: replace it with exactly these models, which can make "
                         "cards flagged by a removed model eligible again, so do it only on purpose")
    ap.add_argument("--concurrency", type=int, default=6, help="per route")
    ap.add_argument("--budget", type=float, default=95.0, help="stop inference calls at this estimated spend (USD)")
    a = ap.parse_args()
    specs = a.models.split(",")
    models = [model_name(x) for x in specs]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(OUT.read_text()) if OUT.is_file() else {}
    done = {k: v for k, v in done.items() if "continuation" in v}  # results from before evidence was kept are redone
    panel = [x.strip() for x in LAUNCH.read_text().split("\n") if x.strip()] if LAUNCH.is_file() else []
    if a.panel == "set":
        panel = models
    elif a.panel == "add":
        panel = panel + [m for m in models if m not in panel]
    instr = OUT.parent / "memprobe_instructions.md"
    system = "You are a helpful assistant."
    instr.write_text(system + "\n")
    from collections import defaultdict
    sems = defaultdict(lambda: asyncio.Semaphore(a.concurrency))
    routes, spend = None, None
    if any(not x.startswith("api:") for x in specs):
        from humanbench.harness.floor_run import _private_routes
        routes = _private_routes()
        spend = routes.probe_spend(a.budget)
    short: list[str] = []
    refused: list[str] = []

    async def ask(spec: str, prompt: str) -> str:
        if spec.startswith("api:"):
            from humanbench.harness.run_model import configured_call
            async with sems["api"]:
                res = await asyncio.to_thread(configured_call, prompt, spec[4:], system)
            return res["text"] if res["ok"] else ""
        return await routes.probe_call(spec, prompt, sems, spend, instr, system, refused)

    async def one(slot: str, spec: str) -> None:
        key = f"{slot}|{model_name(spec)}"
        if key in done:
            return
        words = (PRIV / "oracle" / slot / "draft.md").read_text().split()
        cut = int(len(words) * 0.4)
        head, rest = " ".join(words[:cut]), " ".join(words[cut:])
        n = min(400, len(words) - cut)
        text = ""
        for _ in range(2):  # a refusal or a stub is not evidence either way: ask once more, then leave unprobed
            text = await ask(spec, ASK.format(n=n, head=head))
            if len(text.split()) >= MIN_SHARE * n:
                break
        if len(text.split()) < MIN_SHARE * n:
            short.append(key)
            return  # unprobed: stays out of `done`, so the card is not eligible
        done[key] = {**compare(text, rest), "continuation": text}
        OUT.write_text(json.dumps(done, indent=1))
        if done[key]["flag"]:
            print(f"FLAG {key}: longest run {done[key]['longest_run']} words, 5-gram share {done[key]['five_gram_share']}; "
                  f"{done[key]['runs'][:1]}", flush=True)

    slots = sorted(d.name for d in (REPO / "tasks" / "ref").iterdir() if d.is_dir())
    if a.only:
        slots = [x for x in slots if x in set(a.only.split(","))]
    await asyncio.gather(*(one(x, m) for x in slots for m in specs))
    flagged = sorted({k.split("|")[0] for k, v in done.items() if v["flag"] and k.split("|")[0] in slots})
    full = sum(all(f"{x}|{m}" in done for m in models) for x in slots)
    print(f"{len(slots)} cards, {full} fully probed by {len(models)} models, flagged: {flagged}")
    if refused:
        raise SystemExit(f"the inference budget refused {len(refused)} probe calls ({sorted(set(refused))}): raise "
                         "--budget and run again. The launch panel was not changed.")
    if a.panel != "keep":  # only now, so a failed run cannot leave a panel whose models were never probed
        LAUNCH.write_text("\n".join(panel) + "\n")
    outside = [m for m in models if m not in panel]
    if outside:
        print(f"not in the launch panel, so the results report will leave them out of the cards built from real writing: {outside} "
              "(add them with --panel add)")
    elig = sum(bool(panel) and all(f"{x}|{m}" in done and not done[f"{x}|{m}"]["flag"] for m in panel) for x in slots)
    print(f"launch panel ({len(panel)} models): {elig} of {len(slots)} of these cards are eligible")
    if short:
        print(f"no usable continuation (left unprobed): {sorted(short)}")


if __name__ == "__main__":
    asyncio.run(main())
