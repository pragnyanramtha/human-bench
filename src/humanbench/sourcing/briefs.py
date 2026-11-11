"""Writer briefs: a TaskCard rendered as instructions for a human writer.

The writer gets the same raw material as the model (the request, the notes
pack, the thread or environment, the persona facts) plus channel and length
instructions. Two things are deliberately left out:

* Nothing about AI, detectors or "sounding human". Telling writers their text
  will be compared with machine text makes them perform humanness (extra typos,
  forced slang), which ruins the reference. ``lint_brief`` enforces this.
* Recruiting labels such as ``writing_level``. They describe the pool, not
  instructions; showing "rough" to a writer would push them to write down.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from humanbench.axes import Channel, Mode
from humanbench.schema import TaskCard
from humanbench.taskpkg import render_notes
from humanbench.scoring.verifiers import FORMAT_POLICY, MAX_WORDS, MIN_WORDS

# Words that must never appear in the brief template. Card content (the
# request and notes) may legitimately mention AI as a topic, so it is checked
# only against _PERFORMANCE_TERMS.
_TEMPLATE_BANNED = re.compile(
    r"\b(?:AI|A\.I\.|artificial intelligence|chat ?gpt|gpt-?\d*|llms?|language models?|chatbots?|"
    r"detect(?:or|ors|ion|ed|able)?|human-?like|humanness|sound(?:s|ing)? (?:more )?human|robotic|"
    r"machine[- ](?:written|generated)|generated|benchmark|turing|authentic(?:ity)?|natural-sounding|"
    r"pass(?:es)? as)\b",
    re.IGNORECASE,
)
_PERFORMANCE_TERMS = re.compile(
    r"\b(?:ai[- ]detect\w*|detector|sound(?:s|ing)? (?:more )?human|human-?like|humanness|"
    r"not (?:sound|read) like (?:an? )?(?:ai|bot|robot|chatgpt)|pass(?:es)? as (?:a )?human|"
    r"(?:fool|beat|evade) (?:the )?(?:detector|classifier|judge))\b",
    re.IGNORECASE,
)


class BriefLintError(ValueError):
    pass


def lint_brief(template_text: str, card_text: str = "") -> None:
    """Raise if the brief mentions AI detection or asks writers to sound human."""
    if m := _TEMPLATE_BANNED.search(template_text):
        raise BriefLintError(f"brief template mentions {m.group(0)!r}")
    if m := _PERFORMANCE_TERMS.search(card_text):
        raise BriefLintError(f"task card content asks for performed humanness: {m.group(0)!r}")


_CHANNEL_WHAT: dict[Channel, str] = {
    Channel.EMAIL: "An email. Write it as you'd send it, greeting and sign-off included if you'd use them.",
    Channel.CHAT: "A chat message (Slack, WhatsApp, DM or similar). Write it as you'd type and send it.",
    Channel.FORUM: "A forum post or reply. Write it as you'd post it.",
    Channel.SOCIAL: "A social media post. Write it as you'd post it from your own account.",
    Channel.REVIEW: "A review. Write it as you'd post it on the site.",
    Channel.BLOG: "A blog post or newsletter. Write it as you'd publish it.",
    Channel.WORK_DOCS: "A work document. Write it as you'd share it with the people it's for.",
    Channel.PERSONAL: "A personal piece. Write it as you'd actually give, read out or send it.",
}

_FORMAT_NAMES = {"headings": "headings", "tables": "tables", "bold": "bold text", "bullets": "bullet points"}

_THREAD_WORDS = {Channel.EMAIL: "email thread", Channel.CHAT: "chat", Channel.FORUM: "forum thread",
                 Channel.WORK_DOCS: "thread"}
_ENV_WORDS = {Channel.EMAIL: "inbox", Channel.CHAT: "workspace chat", Channel.FORUM: "forum",
              Channel.SOCIAL: "account", Channel.REVIEW: "order history", Channel.BLOG: "blog dashboard",
              Channel.WORK_DOCS: "workspace", Channel.PERSONAL: "files"}


def _join(items: list[str], word: str) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} {word} {items[-1]}"


def _format_line(card: TaskCard) -> str:
    policy = {**FORMAT_POLICY[card.channel], **card.format_overrides}
    allowed = [_FORMAT_NAMES[k] for k, ok in policy.items() if ok]
    banned = [_FORMAT_NAMES[k] for k, ok in policy.items() if not ok]
    parts = []
    if allowed:
        text = _join(allowed, "and")
        parts.append(f"{text[0].upper()}{text[1:]} {'are' if len(allowed) > 1 or text.endswith('s') else 'is'} fine if you'd use them.")
    if banned:
        parts.append(f"Please don't use {_join(banned, 'or')}.")
    return " ".join(parts) or "Any formatting you'd normally use is fine."


def _requirement(req: str | list[str]) -> str:
    if isinstance(req, str):
        return req
    return req[0] + (f" (or {_join(list(req[1:]), 'or')})" if len(req) > 1 else "")


def requirements_shown(card: TaskCard, include: bool | None) -> bool:
    """Briefs hide must_include / must_not_include by default, in every mode.

    Models never see these lists either. Showing them only to writers would give
    the human side an answer key and steer references toward a checklist; the
    anchors are designed to be things anyone doing the task would mention
    (cardcheck and review atom R7), and a writer who misses one is a signal that
    the anchor is unfair, so QA routes the miss to review.
    """
    if include is not None:
        return include
    return False


def render_thread(context: Any) -> str:
    """Render a thread (humanbench.envs.Thread JSON, a list of messages, or text) as markdown."""
    if isinstance(context, str):
        return context.strip()
    meta = context if isinstance(context, dict) else {}
    messages = meta.get("messages", context)
    me, reply_to = meta.get("user"), meta.get("reply_to")
    out = [f"**Thread: {meta['title']}**"] if meta.get("title") else []
    for m in messages:
        if isinstance(m, str):
            out.append(m)
            continue
        who = m.get("from") or m.get("author") or m.get("user") or "someone"
        when = m.get("date") or m.get("time") or ""
        head = f"**{who}**" + (" (you)" if me and who == me else "") + (f" ({when})" if when else "")
        if reply_to and m.get("id") == reply_to:
            head += ", the message you're replying to"
        body = (m.get("body") or m.get("text") or "").strip()
        out.append(f"{head}\n\n{body}")
    return "\n\n---\n\n".join(out)


def load_context(card: TaskCard, root: Path) -> str | None:
    if not card.context:
        return None
    path = root / card.context
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    try:
        return render_thread(json.loads(raw))
    except json.JSONDecodeError:
        return raw.strip()


def render_brief(card: TaskCard, *, context_text: str | None = None, include_requirements: bool | None = None) -> str:
    """Markdown brief for the writer of ``card``'s human reference.

    ``include_requirements``: None (default) shows must_include / must_not_include
    for direct and thread tasks and hides them for agent and ghostwrite tasks.
    """
    include_requirements = requirements_shown(card, include_requirements)
    p = card.persona
    lo, hi = card.words
    # Same accepted range the scorer uses (10% either side, never past 250 / 4,000).
    lo_ok, hi_ok = max(round(lo * 0.9), MIN_WORDS), min(round(hi * 1.1), MAX_WORDS)
    tpl: list[str] = []  # fixed template text, linted strictly
    content: list[str] = []  # text taken from the card, linted for performance terms only
    out: list[str] = []

    def t(line: str = "") -> None:
        tpl.append(line)
        out.append(line)

    def c(line: str) -> None:
        content.append(line)
        out.append(line)

    t(f"# Writing task {card.id}")
    t()
    t("You're writing this piece as yourself. The situation is set up for you; the words are yours.")
    t("If a detail below doesn't fit your life closely enough to write from, tell us before you start.")
    t()
    t("## About you, for this piece")
    t()
    c(f"- Work: {p.role}")
    c(f"- Age: {p.age}")
    c(f"- Lives in: {p.lives}")
    if p.native:
        c(f"- Your English: {p.english}")
    else:
        c(f"- First language: {p.first_language}; you're writing in English")
    t()
    t("## The situation")
    t()
    t("Here's what's going on, in the words of the person in it (you):")
    t()
    c("> " + card.prompt.strip().replace("\n", "\n> "))
    t()

    notes = render_notes(card)  # the same list, in the same order, that the model gets
    if notes:
        t("## Your notes")
        t()
        t("Use what fits. You don't have to use everything.")
        t()
        for line in notes.splitlines():
            c(line)
        t()

    if card.mode is Mode.THREAD:
        t(f"## The {_THREAD_WORDS.get(card.channel, 'thread')} you're replying to")
        t()
        if context_text:
            c(context_text)
        else:
            t("The writing tool shows the full thread above the editor.")
        t()
    if card.mode in (Mode.AGENT, Mode.GHOSTWRITE):
        place = _ENV_WORDS[card.channel]
        t("## Where you'll work")
        t()
        t(f"You'll do this inside a practice {place} in the writing tool. Look through it the way you")
        t("normally would and open whatever you need; you don't have to read everything. The tool")
        t("records which items you open.")
        if card.mode is Mode.GHOSTWRITE:
            t("Some of your earlier messages or posts are in there too.")
        t()

    t("## What to write")
    t()
    multi = card.mode in (Mode.AGENT, Mode.GHOSTWRITE)
    t(f"- {_CHANNEL_WHAT[card.channel]}")
    if multi:
        t("- Write as many drafts as the situation needs (often one, sometimes several), and leave them")
        t("  as drafts in the tool unless the request says to send.")
        t(f"- Length: about {lo}-{hi} words in total across your drafts. Anything from {lo_ok} to")
        t(f"  {hi_ok} is fine; the editor shows a running word count.")
    else:
        t(f"- Length: about {lo}-{hi} words. Anything from {lo_ok} to {hi_ok} is fine;")
        t("  the editor shows a running word count.")
    t(f"- Formatting: {_format_line(card)}")
    if include_requirements and card.must_include:
        t("- Make sure it covers:")
        for req in card.must_include:
            c(f"  - {_requirement(req)}")
    if include_requirements and card.must_not_include:
        t("- Leave out:")
        for phrase in card.must_not_include:
            c(f"  - {phrase}")
    t("- Write it the way you would if you were really sending or posting it.")
    t("- Only the piece itself goes in the editor. Put any questions or notes for us in the comments box.")
    t("- Use the editor for all of it. Pasting is switched off, and that's expected.")
    t()

    lint_brief("\n".join(tpl), "\n".join(content))
    return "\n".join(out).rstrip() + "\n"
