"""Deterministic checks on a drafted task package, plus its oracle and no-op scores.

Used three ways: by the builder to check its own work, by the task-card
pipeline's blocking gates, and before tasks are frozen.

    uv run python -m humanbench.cardcheck <task_dir> [--slot slot.json] [--accepted DIR ...]

Exit code 0 means every check passed. Every failure names the exact miss.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from humanbench.axes import Band, Mode
from humanbench.schema import Persona, load_personas
from humanbench.scoring.normalize import word_count
from humanbench.scoring.verifiers import _canon
from humanbench.similarity import most_similar_pairs, shared_openings
from humanbench.taskpkg import AGENTIC, PackageError, TaskPackage, load_package, nop_outputs

PROMPT_WORDS = (12, 220)
NOTE_MAX_WORDS = 60
MIN_NOTES = {Band.SHORT: 5, Band.MEDIUM: 7, Band.LONG: 10, Band.EXTENDED: 14}
MIN_NOTES_AGENTIC = 2  # agent tasks get their facts from the environment
# Lexical anchors only for what the piece can't do its job without, and none is fine:
# whether the piece covers the right content is the quality floor's call (DESIGN.md 4.4).
MUST_INCLUDE_RANGE = (0, 4)
MUST_INCLUDE_AGENTIC = (0, 4)
ALT_MAX_WORDS = 4
MIN_ENV_ITEMS = 8
MIN_VOICE_WORDS = 400
MAX_THREAD_WORDS = 1500
MAX_PROMPT_SIMILARITY = 0.5

SLOT_FIELDS = ("id", "split", "channel", "mode", "tone", "audience", "stakes", "band", "persona_id")

# Default-track prompts are what real users type. They never coach the model on
# sounding human; that is what the benchmark measures.
_STYLE_COACHING = re.compile(
    r"\b(sound(s|ing)? (more )?(human|natural|real|authentic)|human[- ]?like|not (sound )?like (an? )?(ai|bot|robot|chatgpt)"
    r"|ai[- ]?(generated|detector|detection|tells?)|undetectable|bypass|chatgpt|llm|language model"
    r"|avoid (clich[eé]s|em ?dashes|buzzwords))\b",
    re.IGNORECASE,
)


def _words(s: str) -> int:
    return word_count(s)


def _present(phrase: str, haystack: str) -> bool:
    return f" {_canon(phrase)} " in haystack


def check_card(pkg: TaskPackage, slot: dict | None = None, personas: Sequence[Persona] | None = None) -> list[str]:
    card = pkg.card
    problems: list[str] = []

    if slot is not None:
        for f in SLOT_FIELDS:
            got = getattr(card, f)
            got = got.value if hasattr(got, "value") else got
            if got != slot[f]:
                problems.append(f"card.{f} is {got!r} but the slot says {slot[f]!r}; slot fields are fixed")
    personas = personas if personas is not None else load_personas()
    pool = {p.id: p for p in personas}
    if card.persona_id not in pool:
        problems.append(f"persona_id {card.persona_id!r} is not in the persona pool")
    elif card.persona != pool[card.persona_id]:
        problems.append("card.persona differs from the persona pool entry; copy it unchanged")

    pw = _words(card.prompt)
    if not PROMPT_WORDS[0] <= pw <= PROMPT_WORDS[1]:
        problems.append(f"prompt has {pw} words; keep it {PROMPT_WORDS[0]}-{PROMPT_WORDS[1]}")
    for m in _STYLE_COACHING.finditer(card.prompt + " " + " ".join(card.notes_pack.wants)):
        problems.append(f"style coaching in prompt or wants: {m.group(0)!r} (Default-track prompts never mention AI or sounding human)")

    notes = card.notes_pack
    items = [*notes.facts, *notes.anecdotes, *notes.opinions, *notes.wants]
    need = MIN_NOTES_AGENTIC if card.mode in AGENTIC else MIN_NOTES[card.band]
    if len(items) < need:
        problems.append(f"notes pack has {len(items)} items; a {card.band.value} task needs at least {need}")
    for i in items:
        if _words(i) > NOTE_MAX_WORDS:
            problems.append(f"note over {NOTE_MAX_WORDS} words: {i[:60]!r}...")
    if len({_canon(i) for i in items}) != len(items):
        problems.append("notes pack has duplicate items")

    public_source = f" {_canon(' '.join(pkg.source_texts(include_private=False)))} "
    lo, hi = MUST_INCLUDE_AGENTIC if card.mode in AGENTIC else MUST_INCLUDE_RANGE
    if not lo <= len(card.must_include) <= hi:
        problems.append(f"must_include has {len(card.must_include)} entries; use {lo}-{hi}")
    for pt in card.must_cover:
        if not 3 <= _words(pt) <= 30:
            problems.append(f"must_cover point should be one plain statement of 3-30 words: {pt[:60]!r}")
    anchors = list(card.must_include)
    if card.agent_checks is not None:
        anchors += [r for exp in card.agent_checks.required for r in exp.must_include]
    for req in anchors:
        options = [req] if isinstance(req, str) else list(req)
        empty = [o for o in options if not _canon(o)]
        if empty:
            problems.append(f"must_include alternative {empty} is empty once punctuation is ignored; checks match words, not punctuation")
        long_alts = [o for o in options if _words(o) > ALT_MAX_WORDS]
        if long_alts:
            problems.append(
                f"must_include alternative over {ALT_MAX_WORDS} words: {long_alts}; use short anchors a human would also write"
            )
        if not any(_present(o, public_source) for o in options):
            problems.append(f"must_include {options} appears nowhere in the prompt, notes, thread or public items")
    for p in card.must_not_include:
        if _words(p) > 8:
            problems.append(f"must_not_include entry over 8 words: {p!r}")
        if not _canon(p):
            problems.append(f"must_not_include entry {p!r} is empty once punctuation is ignored; it would never match")
        elif _present(p, f" {_canon(card.prompt)} "):
            problems.append(f"must_not_include {p!r} is the user's own word in the prompt; a writer could reasonably echo it")

    if pkg.thread is not None:
        t = pkg.thread
        answered = next(m for m in t.messages if m.id == t.reply_to)
        if answered.author == t.user:
            problems.append(f"context.json: the user ({t.user}) cannot be replying to their own message {t.reply_to}")
        tw = sum(_words(m.body) for m in t.messages)
        if tw > MAX_THREAD_WORDS:
            problems.append(f"context.json: thread has {tw} words; keep it under {MAX_THREAD_WORDS}")

    if card.mode in AGENTIC:
        env, checks = pkg.env, card.agent_checks
        assert env is not None and checks is not None
        if len(env.items) < MIN_ENV_ITEMS:
            problems.append(f"env/fixture.json has {len(env.items)} items; agent tasks need {MIN_ENV_ITEMS}+ (most are distractors)")
        if not any(i.private for i in env.items):
            problems.append("env/fixture.json has no private item; add one whose details must not leak")
        if not checks.private_phrases:
            problems.append("agent_checks.private_phrases is empty; list distinctive strings from the private items")
        private_text = f" {_canon(' '.join(t for i in env.items if i.private for t in [i.title or '', i.body]))} "
        for p in checks.private_phrases:
            if not _canon(p):
                problems.append(f"private phrase {p!r} is empty once punctuation is ignored")
                continue
            if not _present(p, private_text):
                problems.append(f"private phrase {p!r} is not in any private item")
            if _present(p, public_source):
                problems.append(f"private phrase {p!r} also appears in public material, so leaking it is unavoidable")
        known_targets = {t.lower() for t in env.targets}
        ids = {i.id for i in env.items}
        for exp in checks.required:
            for t in [*exp.targets(), *exp.cc, *exp.all_recipients]:
                if t.lower() not in known_targets:
                    problems.append(f"agent_checks target {t!r} is not in env.targets")
            if exp.in_reply_to is not None and exp.in_reply_to not in ids:
                problems.append(f"agent_checks in_reply_to {exp.in_reply_to!r} is not an item id")
        floor = sum(exp.min_words for exp in checks.required)
        if floor > card.words[0]:
            problems.append(
                f"required writes demand {floor} words in total (min_words), more than the band's lower bound "
                f"{card.words[0]}; a valid answer at the short end would fail. Lower min_words."
            )
        wanted = {"create_draft" if checks.delivery == "draft" else "send"}
        if not wanted <= set(card.tools):
            problems.append(f"tools must include {sorted(wanted)} for delivery={checks.delivery}")
        if checks.delivery == "draft" and "send" in card.tools:
            problems.append("delivery is draft, so do not give the agent the send tool")
        if not {"search", "read_item"} & set(card.tools):
            problems.append("tools need at least one of search/read_item so the agent can find things")
    if pkg.voice is not None:
        vw = sum(_words(s.body) for s in pkg.voice.samples)
        if vw < MIN_VOICE_WORDS:
            problems.append(f"env/user_voice.json has {vw} words of samples; need {MIN_VOICE_WORDS}+")
    return problems


def check_variety(pkg: TaskPackage, others: Sequence[TaskPackage | tuple[str, str]]) -> list[str]:
    """Prompt must not resemble another card's prompt (DESIGN.md 2.4, checks 3-4).

    `others` are accepted packages or (card id, prompt) pairs of cards still in progress."""
    pairs = [(o.card.id, o.card.prompt) if isinstance(o, TaskPackage) else o for o in others]
    pairs = [(i, p) for i, p in pairs if i != pkg.card.id]
    if not pairs:
        return []
    items = [(pkg.card.id, pkg.card.prompt), *pairs]
    problems = []
    for a, b, sim in most_similar_pairs(items, top_fraction=1.0):
        if pkg.card.id in (a, b) and sim > MAX_PROMPT_SIMILARITY:
            other = b if a == pkg.card.id else a
            problems.append(f"prompt is too similar to accepted card {other} (cosine {sim:.2f} > {MAX_PROMPT_SIMILARITY})")
    for opening, ids in shared_openings(items).items():
        if pkg.card.id in ids:
            others = [i for i in ids if i != pkg.card.id]
            problems.append(f"prompt opens with the same four words as {others}: {opening!r}")
    return problems


def oracle_report(pkg: TaskPackage) -> list[str]:
    """Golden must score 1.0 and every no-op 0.0."""
    problems = []
    if pkg.card.mode in AGENTIC:
        golden, why = pkg.score(run=pkg.oracle_run)
    else:
        golden, why = pkg.score(text=pkg.oracle_text)
    if golden != 1.0:
        problems.append(f"oracle scores {golden}, must be 1.0: {why}")
    for name, kwargs in nop_outputs(pkg):
        score, _ = pkg.score(**kwargs)
        if score != 0.0:
            problems.append(f"no-op output {name!r} scores {score}, must be 0.0 (the checks are too lax)")
    return problems


def load_accepted(dirs: Sequence[Path]) -> list[TaskPackage]:
    out = []
    for d in dirs:
        for card in sorted(Path(d).glob("*/card.yaml")):
            if card.parent.name.startswith("."):
                continue  # publication staging, not an accepted package
            try:
                out.append(load_package(card.parent, require_oracle=False))
            except PackageError as exc:
                raise PackageError([f"accepted package {card.parent} is invalid: {p}" for p in exc.problems]) from exc
    return out


def check_all(task_dir: Path, slot: dict | None = None, accepted_dirs: Sequence[Path] = ()) -> list[str]:
    try:
        pkg = load_package(task_dir)
    except PackageError as exc:
        return exc.problems
    problems = check_card(pkg, slot)
    problems += check_variety(pkg, load_accepted(accepted_dirs))
    problems += oracle_report(pkg)
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--slot", type=Path, help="slot.json the card must match (default: <task_dir>/slot.json if present)")
    ap.add_argument("--accepted", type=Path, action="append", default=[], help="directory of accepted task packages")
    args = ap.parse_args(argv)
    slot_path = args.slot or (args.task_dir / "slot.json")
    slot = json.loads(slot_path.read_text(encoding="utf-8")) if slot_path.is_file() else None
    problems = check_all(args.task_dir, slot, args.accepted)
    if problems:
        print(f"FAIL ({len(problems)} problems)")
        for p in problems:
            print(f"- {p}")
        return 1
    print("PASS: structure, variety, oracle 1.0, no-ops 0.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
