"""Build synthetic mirrors for the human texts.

  uv run python generate.py briefs     # ~14 batched calls: human text -> brief (no text copied)
  uv run python generate.py mirrors    # 1 call per human text: brief -> AI piece
  uv run python generate.py pairs      # assemble pairs.jsonl, check copying

Every call is cached under private/judge_eval/ and skipped on re-run.
Generators use the provider's hosted client in the working repository or a configured API endpoint,
both with a plain "helpful assistant" system prompt and no tools.
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import BRIEFS_PATH, EVAL_DIR, GEN_DIR, HUMAN_PATH, PAIRS_PATH, read_jsonl, word_count, write_jsonl

CLAUDE_MODEL = "claude-sonnet-5-5"
MODEL_B = "gpt-6.1-sol"
EFFORT_B = "low"  # model default for this experiment
SYSTEM = "You are a helpful assistant."
BATCH = 10
WORKERS = 4
WORKDIR = EVAL_DIR / "cli_workdir"  # empty dir the CLIs run in

BRIEF_PROMPT = """You are helping build a research dataset. For each numbered text below, write a short
brief that a different writer could use to write a NEW piece for the same situation. The brief must NOT
contain the text itself: paraphrase everything in your own plain words, and never copy sentences or
distinctive phrases. Keep concrete facts (names, places, numbers, products, events) because the new writer
needs them.

Return ONLY a JSON array, one object per text, with exactly these keys:
- "id": the text's id
- "genre": what kind of piece it is and where it appears, e.g. "work email to a colleague", "Yelp review of a
  tyre shop", "answer on a gardening Q&A site", "post on a Reddit relationships forum", "personal blog post",
  "Medium article", "short story written for a Reddit writing-prompt thread", "IMDb user review"
- "writer": who is writing (role, rough age, anything inferable), one sentence; say "unknown" if unclear
- "audience": who it is for
- "situation": what prompted it, 1-2 sentences
- "key_points": 6-12 short bullet strings with the content to cover, in the original order
- "tone": a few words
- "writing_level": one of "polished", "everyday", "rough but clear"
- "format_notes": one short sentence on format a writer would be told (e.g. "reply to the question above",
  "plain email, no subject line needed"), without describing the original's style

TEXTS:
"""

DEFAULT_TEMPLATE = """Write a {genre}. Aim for about {words} words.

Who is writing: {writer}
Audience: {audience}
Situation: {situation}
Points to cover:
{points}
Tone: {tone}
Writer's usual writing level: {writing_level}
{format_notes}
{extra}
Output only the text of the piece itself, with no title or commentary."""

INSTRUCTED_EXTRA = """
Important: write this exactly as the real person described would write it, so it reads as genuinely written by
them and not by an AI. Match their writing level, including the casualness, rough edges and quirks a real
person at that level would have. Avoid AI tells: no em dashes, no markdown or headings, no neat summary or
moral at the end, no "it's not X, it's Y" constructions, no lists of three, no stock words like "delve",
"testament", "tapestry", "navigate", "crucial". Vary sentence length naturally, don't give every point equal
weight, and don't over-explain.
"""



def _working_generator():
    import importlib.util
    path = Path(__file__).with_name("generate_private.py")
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("working_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_a(prompt: str) -> str:
    from humanbench.harness.run_model import configured_call
    legacy = _working_generator()
    if legacy is not None:
        return legacy.working_a(prompt)
    res = configured_call(prompt, CLAUDE_MODEL, SYSTEM, 900)
    if not res["ok"]:
        raise RuntimeError(res["error"])
    return res["text"]


def run_b(prompt: str) -> str:
    from humanbench.harness.run_model import configured_call
    legacy = _working_generator()
    if legacy is not None:
        return legacy.working_b(prompt)
    res = configured_call(prompt, MODEL_B, SYSTEM, 900)
    if not res["ok"]:
        raise RuntimeError(res["error"])
    return res["text"]


def call(gen: str, prompt: str, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            return run_a(prompt) if gen == "claude" else run_b(prompt)
        except Exception as e:  # noqa: BLE001 - surface after retries
            if attempt == retries:
                raise
            print(f"  retry {gen} after error: {str(e)[:200]}", file=sys.stderr)
            time.sleep(10 * (attempt + 1))
    raise AssertionError


def parse_json_array(s: str) -> list[dict]:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    start, end = s.find("["), s.rfind("]")
    return json.loads(s[start : end + 1])


def cmd_briefs() -> None:
    human = read_jsonl(HUMAN_PATH)
    done = {b["id"]: b for b in read_jsonl(BRIEFS_PATH)}
    todo = [h for h in human if h["id"] not in done]
    batches = [todo[i : i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(done)} briefs cached, {len(todo)} to do in {len(batches)} calls")

    def work(batch):
        body = "\n\n".join(f"### id: {h['id']}\n{h['text']}" for h in batch)
        res = parse_json_array(call("claude", BRIEF_PROMPT + body))
        ids = {h["id"] for h in batch}
        return [r for r in res if r.get("id") in ids]

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(work, b) for b in batches]
        for f in as_completed(futs):
            try:
                for b in f.result():
                    done[b["id"]] = b
            except Exception as e:  # noqa: BLE001
                print("batch failed:", str(e)[:300], file=sys.stderr)
            order = {h["id"]: i for i, h in enumerate(human)}
            write_jsonl(BRIEFS_PATH, sorted(done.values(), key=lambda b: order[b["id"]]))
    print(f"{len(done)}/{len(human)} briefs -> {BRIEFS_PATH}")


def assign(human: list[dict]) -> dict[str, tuple[str, str]]:
    """Balanced generator x mode per channel: alternate generators; every 3rd item per generator instructed."""
    out, counters = {}, {}
    for h in human:
        c = counters.setdefault(h["channel"], {"n": 0, "claude": 0, "model_b": 0})
        gen = "claude" if c["n"] % 2 == 0 else "model_b"
        c["n"] += 1
        mode = "instructed" if c[gen] % 3 == 2 else "default"
        c[gen] += 1
        out[h["id"]] = (gen, mode)
    return out


def build_prompt(brief: dict, words: int, mode: str) -> str:
    target = max(250, int(round(words / 10.0) * 10))
    return DEFAULT_TEMPLATE.format(
        genre=brief["genre"], words=target, writer=brief.get("writer", "unknown"),
        audience=brief.get("audience", ""), situation=brief.get("situation", ""),
        points="\n".join(f"- {p}" for p in brief.get("key_points", [])), tone=brief.get("tone", ""),
        writing_level=brief.get("writing_level", "everyday"),
        format_notes=f"Format: {brief['format_notes']}" if brief.get("format_notes") else "",
        extra=INSTRUCTED_EXTRA if mode == "instructed" else "",
    ).replace("\n\n\n", "\n\n")


def cmd_mirrors(limit: int | None = None) -> None:
    human = read_jsonl(HUMAN_PATH)
    briefs = {b["id"]: b for b in read_jsonl(BRIEFS_PATH)}
    plan = assign(human)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    todo = [h for h in human if h["id"] in briefs and not (GEN_DIR / f"{h['id']}.json").exists()]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} mirrors to generate")

    def work(h):
        gen, mode = plan[h["id"]]
        prompt = build_prompt(briefs[h["id"]], h["words"], mode)
        t0 = time.time()
        text = call(gen, prompt).strip()
        rec = dict(id=h["id"], generator=gen, model=CLAUDE_MODEL if gen == "claude" else MODEL_B,
                   mode=mode, prompt=prompt, text=text, words=word_count(text), seconds=round(time.time() - t0, 1),
                   created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (GEN_DIR / f"{h['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        return rec

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(work, h): h["id"] for h in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                print(f"[{i}/{len(todo)}] {r['id']} {r['generator']}/{r['mode']} {r['words']}w {r['seconds']}s")
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(todo)}] {futs[f]} FAILED: {str(e)[:300]}", file=sys.stderr)


def ngram_overlap(a: str, b: str, n: int = 8) -> int:
    wa, wb = a.lower().split(), b.lower().split()
    ga = {tuple(wa[i : i + n]) for i in range(len(wa) - n + 1)}
    return sum(tuple(wb[i : i + n]) in ga for i in range(len(wb) - n + 1))


def strip_wrapper(text: str) -> str:
    """Drop a leading 'Here's ...:' line or a title line if a model added one despite instructions."""
    lines = text.strip().split("\n")
    if lines and re.match(r"(?i)^(here'?s|here is|sure|certainly)\b.*:\s*$", lines[0].strip()):
        lines = lines[1:]
    return "\n".join(lines).strip()


def cmd_pairs() -> None:
    human = read_jsonl(HUMAN_PATH)
    pairs, dropped = [], []
    for h in human:
        p = GEN_DIR / f"{h['id']}.json"
        if not p.exists():
            continue
        g = json.loads(p.read_text())
        ai = strip_wrapper(g["text"])
        ov = ngram_overlap(h["text"], ai)
        share = ov / max(1, len(ai.split()) - 7)
        if share > 0.15:  # names and titles carried through the brief are fine; reproduced prose is not
            dropped.append((h["id"], ov, round(share, 3)))
            continue
        pairs.append(dict(pair_id=h["id"], channel=h["channel"], source=h["source"], generator=g["generator"],
                          model=g["model"], mode=g["mode"], human=h["text"], ai=ai, human_words=h["words"],
                          ai_words=word_count(ai), overlap8=ov, overlap8_share=round(share, 4)))
    write_jsonl(PAIRS_PATH, pairs)
    print(f"{len(pairs)} pairs -> {PAIRS_PATH}; dropped for copying: {dropped}")


if __name__ == "__main__":
    WORKDIR.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "briefs":
        cmd_briefs()
    elif what == "mirrors":
        cmd_mirrors(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif what == "pairs":
        cmd_pairs()
    else:
        raise SystemExit(__doc__)
