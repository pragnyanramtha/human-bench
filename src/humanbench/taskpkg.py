"""A task package on disk: card.yaml plus its fixtures and oracle.

    <task>/card.yaml
    <task>/context.json          thread_reply
    <task>/env/fixture.json      agent, ghostwrite
    <task>/env/user_voice.json   ghostwrite
    <task>/oracle/draft.md       direct, thread_reply   (golden text; model-written, never a human ref)
    <task>/oracle/actions.json   agent, ghostwrite      (golden writes + tool-call count)

This module loads a package, renders what the model sees, and scores an output
with the deterministic layers (verifiers, then agent checks). The score is 1.0
only if every deterministic check passes.
"""

from __future__ import annotations

import re

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from humanbench.axes import Mode
from humanbench.envs import Action, Environment, OracleRun, Thread, UserVoice
from humanbench.schema import TaskCard
from humanbench.scoring.agent_checks import bundle_text, run_agent_checks
from humanbench.scoring.normalize import normalize
from humanbench.scoring.verifiers import (check_must_not_include, run_verifiers, check_format,
    check_wrapper, check_injection, check_hidden_chars)

CONTEXT_FILE = "context.json"
ENV_FILE = "env/fixture.json"
VOICE_FILE = "env/user_voice.json"
ORACLE_DRAFT = "oracle/draft.md"
ORACLE_ACTIONS = "oracle/actions.json"

AGENTIC = (Mode.AGENT, Mode.GHOSTWRITE)


class PackageError(ValueError):
    """Raised with the exact list of problems found."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def required_files(mode: Mode, *, oracle: bool = True) -> list[str]:
    files = ["card.yaml"]
    if mode is Mode.THREAD:
        files.append(CONTEXT_FILE)
    if mode in AGENTIC:
        files.append(ENV_FILE)
    if mode is Mode.GHOSTWRITE:
        files.append(VOICE_FILE)
    if oracle:
        files.append(ORACLE_ACTIONS if mode in AGENTIC else ORACLE_DRAFT)
    return files


def _fmt_validation(where: str, exc: ValidationError) -> list[str]:
    return [f"{where}: {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()]


@dataclass
class TaskPackage:
    root: Path
    card: TaskCard
    thread: Thread | None = None
    env: Environment | None = None
    voice: UserVoice | None = None
    oracle_text: str | None = None
    oracle_run: OracleRun | None = None
    warnings: list[str] = field(default_factory=list)

    # --- what the model sees ------------------------------------------------

    def notes_text(self) -> str:
        """The notes as the person would paste them: one plain list, no schema headings.
        (The facts / anecdotes / opinions / wants split is for us, not for the reader.)"""
        return render_notes(self.card)

    def thread_text(self) -> str:
        if self.thread is None:
            return ""
        t = self.thread
        answered = next(m for m in t.messages if m.id == t.reply_to)
        title = f"Thread: {t.title}" if t.title else "Thread:"
        lines = [f"{title}\n(I'm {t.user} here. I'm replying to {answered.author}'s message of {answered.date}.)"]
        for m in t.messages:
            head = f"From: {m.author} | {m.date}"
            if m.id == t.reply_to:
                head = "[the message I'm replying to] " + head
            if m.to:
                head += f" | To: {', '.join(m.to)}"
            if m.subject:
                head += f" | Subject: {m.subject}"
            lines.append(f"{head}\n{m.body.strip()}")
        return "\n\n---\n\n".join(lines)

    def about_line(self) -> str:
        """The persona facts and the length target, given to models and writers alike.

        Writing level and English variety are not stated: they are how the person
        writes, not facts the person would tell an assistant.
        """
        p = self.card.persona
        lo, hi = self.card.words
        what = "all the messages together" if self.card.mode in AGENTIC else "it"
        return (
            f"(About me: {', '.join(str(x) for x in (p.role, p.age, p.lives) if x)}. "
            f"Length: {what} should come to {'' if self.card.length_words else 'roughly '}{lo:,}-{hi:,} words"
            + (f", and no more than {self.card.max_chars:,} characters" if self.card.max_chars else "") + ".)"
        )

    def user_message(self) -> str:
        """The Default-track user turn, which is also what the human writer is shown.
        Agent tasks get the workspace through tools."""
        parts = [self.card.prompt.strip()]
        notes = self.notes_text()
        if notes:
            parts.append(notes)
        if self.thread is not None:
            parts.append(self.thread_text())
        parts.append(self.about_line())
        return "\n\n".join(parts)

    def simulated_agent_prompt(self) -> str:
        """Agent and ghostwrite tasks without the tool harness: the whole workspace is shown as text
        and the model writes its messages as plain drafts with a short header (To, Replying to, Cc,
        Subject), which `parse_agent_reply` reads. No JSON: asking for JSON made the writing itself
        more mechanical. A stand-in until a real tool harness exists; it tests the writing and the
        routing, not the searching."""
        env, checks = self.env, self.card.agent_checks
        assert env is not None and checks is not None
        lines = [self.user_message(), "",
                 "(Tool use is simulated here. Everything in the workspace is listed below. "
                 f"Workspace owner: {env.owner}. Now: {env.now}. Valid targets: {', '.join(env.targets)}.)", ""]
        for i in env.items:
            head = f"[{i.id}] {i.kind} in {i.folder} | {i.date} | from {i.author or '-'}"
            if i.to:
                head += f" | to {', '.join(i.to)}"
            if i.title:
                head += f" | {i.title}"
            if i.thread:
                head += f" | thread {i.thread}"
            if i.reply_to:
                head += f" | reply to [{i.reply_to}]"
            lines.append(f"{head}\n{i.body}")
            for a in i.attachments:
                lines.append(f"  attachment {a.name}:\n{a.content}")
        if self.voice is not None:
            lines.append("\nThings I've written before:")
            lines += [f"[{s.where}, {s.date}] {s.body}" for s in self.voice.samples]
        lo, hi = self.card.words
        lines.append(
            "\nWrite the message or messages exactly as they should go out, as plain text. Start each one with these "
            "four lines, then a blank line, then the text of the message:\n"
            # No example id here: the first wording named the workspace's first item as its example, and models
            # replied to it. "To" is the destination even for a draft, so a draft is not filed somewhere else.
            "To: <where this text is to end up once it is sent or posted (the same place even if it is only a draft "
            "for now): one of the valid targets listed above>\n"
            "Replying to: <only if this text is itself a reply to one item above (an answer to that message, or a "
            "comment under that post): that item's id, the code in square brackets at its start. For a new message, "
            "post or document: none>\n"
            "Cc: <other valid targets, or none>\n"
            "Subject: <a subject line, or none>\n\n"
            "If you write more than one message, put a line with only --- between them. Write nothing before the first "
            "To: line and nothing after the last message.\n"
            f"Length: all the messages together should come to {'' if self.card.length_words else 'roughly '}{lo:,}-{hi:,} words"
            + (f", and no more than {self.card.max_chars:,} characters" if self.card.max_chars else "") + "."
        )
        return "\n".join(lines)

    def parse_agent_reply(self, text: str) -> list[Action]:
        """The messages a model wrote for a workspace task, as actions.

        Reads the plain draft format asked for in `simulated_agent_prompt`: blocks separated by a line
        of dashes, each starting with header lines (To, Replying to, Cc, Subject) and then the body.
        Header names are matched loosely (case, markdown emphasis, a leading list marker). A reply
        that is JSON in the earlier {"actions": [...]} shape is still accepted. Raises ValueError when
        no message with a To: line can be found."""
        assert self.card.agent_checks is not None
        tool = "send" if self.card.agent_checks.delivery == "send" else "create_draft"
        body = text.strip()
        fence = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n```\s*$", body, re.DOTALL)
        if fence:
            body = fence.group(1).strip()
        if body.startswith("{"):
            try:
                return [Action.model_validate(a) for a in json.loads(body)["actions"]]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        key = re.compile(r"^[\s>*_`#-]*(to|replying to|reply to|in reply to|cc|subject)[\s*_`]*:[\s*_`]*(.*?)[\s*_`]*$", re.I)
        valid = sorted(self.env.targets, key=len, reverse=True) if self.env is not None else []
        item_ids = {i.id for i in self.env.items} if self.env is not None else set()
        nothing = ("none", "n/a", "na", "null", "no", "nobody", "nil")

        def targets_in(value: str) -> list[str]:
            """The targets a To or Cc value names, in the order written.

            Valid targets are found wherever they stand in the value ("Sam Okafor <sam@x.example>",
            "#general (draft)"), longest first so one is never read inside another. Any other email
            address is kept too, so a message to someone the card rules out still shows up. If the
            value names nothing known, it is split on commas, semicolons and "and"."""
            found: list[tuple[int, int, str]] = []
            for t in valid:
                for m in re.finditer(r"(?<![\w@#/.:-])" + re.escape(t) + r"(?![\w@/:-])", value, re.I):
                    if not any(a < m.end() and m.start() < b for a, b, _ in found):
                        found.append((m.start(), m.end(), t))
            for m in re.finditer(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", value):
                if not any(a < m.end() and m.start() < b for a, b, _ in found):
                    found.append((m.start(), m.end(), m.group(0)))
            if found:
                return list(dict.fromkeys(t for _, _, t in sorted(found)))
            parts = [x.strip().strip("<>[]()").strip() for x in re.split(r"\s*[,;]\s*|\s+and\s+", value)]
            return [x for x in parts if x and x.split()[0].lower().strip(".:") not in nothing]

        def reply_id(value: str) -> str | None:
            """The item a message says it replies to: the first known item id in the value, whatever is
            written round it. "none", an empty value or a dash is no reply. An id the workspace does not
            have is kept as written, so the check can see a reply to something that is not there."""
            words = re.findall(r"[\w.:#/-]*\w", value)
            known = next((w for w in words if w in item_ids), None)
            if known or not words or words[0].lower() in nothing:
                return known
            return words[0]

        # Messages are separated by a line of dashes. Some answers leave only a blank line between them:
        # a To: line after a blank line, with another header line just below it, also starts a new message
        # once the current one has some text of its own.
        blocks, cur = [], []
        lines = body.split("\n")
        for n, line in enumerate(lines):
            if re.match(r"^\s*-{3,}\s*$", line):
                blocks.append(cur)
                cur = []
                continue
            m = key.match(line)
            if (m and m.group(1).lower() == "to" and n and not lines[n - 1].strip()
                    and any(x.strip() and not key.match(x) for x in cur)
                    and any((k := key.match(x)) and k.group(1).lower() != "to" for x in lines[n + 1:n + 4])):
                blocks.append(cur)
                cur = []
            cur.append(line)
        blocks.append(cur)

        actions: list[Action] = []
        for lines in blocks:
            block = "\n".join(lines)
            lines = block.strip("\n").split("\n")
            head: dict[str, str] = {}
            i = 0
            while i < len(lines):
                m = key.match(lines[i])
                if m:
                    name = m.group(1).lower()
                    head["reply" if "repl" in name else name] = m.group(2).strip()
                    i += 1
                elif not lines[i].strip() and not head:
                    i += 1  # blank lines before the header
                else:
                    break
            text_body = "\n".join(lines[i:]).strip()
            to = targets_in(head.get("to", ""))
            if not to:
                if actions and block.strip():  # a dashed line inside a message: it belongs to the previous one
                    prev = actions[-1]
                    actions[-1] = prev.model_copy(update={"body": prev.body + "\n\n---\n\n" + block.strip()})
                continue
            subject = head.get("subject", "").strip()
            # A message to several people names them all on the To line. The first is the target; the others
            # are recipients too, which is what cc records.
            actions.append(Action(
                tool=tool, target=to[0], in_reply_to=reply_id(head.get("reply", "")),
                cc=list(dict.fromkeys(x for x in [*to[1:], *targets_in(head.get("cc", ""))] if x != to[0])),
                subject=None if not subject or subject.strip("[]()<>.-—").lower() in ("", *nothing) else subject,
                body=text_body,
            ))
        if not actions:
            raise ValueError("no message with a To: line")
        return actions

    def model_prompt(self) -> str:
        """What a model under test is sent on the Default track."""
        return self.simulated_agent_prompt() if self.card.mode in AGENTIC else self.user_message()

    def source_texts(self, include_private: bool = True) -> list[str]:
        """Every piece of text the model is given (for availability and copy checks)."""
        n = self.card.notes_pack
        out = [self.card.prompt, *n.facts, *n.anecdotes, *n.opinions, *n.wants]
        if self.thread is not None:
            out += [m.body for m in self.thread.messages]
        if self.env is not None:
            for item in self.env.items:
                if include_private or not item.private:
                    out += [item.title or "", item.body, *(a.content for a in item.attachments)]
        if self.voice is not None:
            out += [s.body for s in self.voice.samples]
        return [t for t in out if t]

    # --- scoring --------------------------------------------------------------

    def scoring_text(self, *, text: str | None = None, run: OracleRun | None = None) -> str:
        """Shared verifier/judge body text. Subjects are verified separately, never judged."""
        return bundle_text(run.actions) if run is not None else normalize(text or "")

    def authored_text(self, *, text: str | None = None, run: OracleRun | None = None) -> str:
        """Everything the writer authored, unnormalised: what the quality floor reads.

        Unlike `scoring_text` (the judge view), this keeps subjects, recipients and any text that
        normalisation would strip, so nothing authored escapes the floor's privacy and task questions."""
        if run is None:
            return (text or "").strip()
        parts = []
        for a in run.actions:
            head = f"[{a.tool} to {a.target}" + (f", replying to {a.in_reply_to}" if a.in_reply_to else "")
            head += (f", cc {', '.join(a.cc)}" if a.cc else "") + "]"
            parts.append(head + (f"\nSubject: {a.subject}" if a.subject else "") + f"\n{a.body}")
        return "\n\n".join(parts)

    def reference_texts(self) -> list[str]:
        """Human reference texts a model output must not copy (cards built from real writing only)."""
        if self.card.split != "ref":
            return []
        p = self.root.resolve().parents[2] / "private" / "oracle" / self.card.id / "draft.md"
        return [p.read_text(encoding="utf-8")] if p.is_file() else []

    def score(
        self, *, text: str | None = None, run: OracleRun | None = None, reference_texts: Sequence[str] = ()
    ) -> tuple[float, list[str]]:
        """1.0 if every deterministic check passes, else 0.0, plus the failures.

        `reference_texts` are human references for this task: sharing any 13-word run
        with one fails. The task's own material may be quoted, up to a share."""
        card = self.card
        failures: list[str] = []
        if card.mode in AGENTIC:
            if run is None:
                return 0.0, ["no agent run (no writes recorded)"]
            assert self.env is not None and card.agent_checks is not None
            report = run_agent_checks(
                run.actions,
                card.agent_checks,
                self.env,
                tool_calls=run.tool_calls,
                known_text=self.source_texts(include_private=False),
            )
            failures += [f"agent:{r.name}: {r.detail}" for r in report.failures]
            text = bundle_text(run.actions)
            subjects = " \n".join(a.subject for a in run.actions if a.subject)
            if subjects:
                banned = check_must_not_include(subjects, card.must_not_include)
                if not banned.passed:
                    failures.append(f"verifier:must_not_include (subject): {banned.detail}")
        fields = [text or ""] if run is None else [v for a in run.actions for v in (a.body, a.subject or "")]
        for raw in fields:
            for result in (check_wrapper(raw), check_injection(raw), check_hidden_chars(raw),
                           check_format(raw, card.channel, card.format_overrides)):
                if not result.passed:
                    failures.append(f"verifier:{result.name} (raw field): {result.detail}")
        text = self.scoring_text(text=text, run=run)
        report = run_verifiers(
            text,
            channel=card.channel,
            band=card.words,
            must_include=card.must_include,
            must_not_include=card.must_not_include,
            format_overrides=card.format_overrides,
            copy_sources=list(reference_texts),
            material=self.source_texts(),
            noise_thresholds={},
            # A range the user states themselves ("keep it to 360-400 words") is exact; a band is a rough target.
            length_tolerance=0.0 if card.length_words else 0.10,
        )
        failures += [f"verifier:{r.name}: {r.detail}" for r in report.failures]
        if card.max_chars is not None and len(text) > card.max_chars:
            failures.append(f"verifier:length: {len(text):,} characters, the request allows {card.max_chars:,}")
        return (0.0 if failures else 1.0), failures


def load_package(root: Path | str, *, require_oracle: bool = True) -> TaskPackage:
    """Load and validate a package. Accepted packages have no oracle (it lives in private/)."""
    root = Path(root)
    problems: list[str] = []
    card_path = root / "card.yaml"
    if not card_path.is_file():
        raise PackageError([f"missing {card_path.name}"])
    try:
        raw = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PackageError([f"card.yaml does not parse: {exc}"]) from exc
    try:
        card = TaskCard.model_validate(raw)
    except ValidationError as exc:
        raise PackageError(_fmt_validation("card.yaml", exc)) from exc

    for rel in required_files(card.mode, oracle=require_oracle):
        if not (root / rel).is_file():
            problems.append(f"missing {rel} (required for {card.mode.value} tasks)")
    expected_refs = {
        "context": CONTEXT_FILE if card.mode is Mode.THREAD else None,
        "environment": ENV_FILE if card.mode in AGENTIC else None,
        "user_voice": VOICE_FILE if card.mode is Mode.GHOSTWRITE else None,
    }
    for fld, want in expected_refs.items():
        got = getattr(card, fld)
        if got != want:
            problems.append(f"card.{fld} must be {want!r} for {card.mode.value} tasks, got {got!r}")
    if problems:
        raise PackageError(problems)

    pkg = TaskPackage(root=root, card=card)
    loaders = (
        ("thread", CONTEXT_FILE, Thread),
        ("env", ENV_FILE, Environment),
        ("voice", VOICE_FILE, UserVoice),
    )
    for attr, rel, model in loaders:
        path = root / rel
        if rel in required_files(card.mode):
            try:
                setattr(pkg, attr, model.model_validate(json.loads(path.read_text(encoding="utf-8"))))
            except json.JSONDecodeError as exc:
                problems.append(f"{rel} is not valid JSON: {exc}")
            except ValidationError as exc:
                problems += _fmt_validation(rel, exc)
    if not require_oracle:
        pass
    elif card.mode in AGENTIC:
        try:
            pkg.oracle_run = OracleRun.model_validate(
                json.loads((root / ORACLE_ACTIONS).read_text(encoding="utf-8"))
            )
        except json.JSONDecodeError as exc:
            problems.append(f"{ORACLE_ACTIONS} is not valid JSON: {exc}")
        except ValidationError as exc:
            problems += _fmt_validation(ORACLE_ACTIONS, exc)
    else:
        pkg.oracle_text = (root / ORACLE_DRAFT).read_text(encoding="utf-8")
    if problems:
        raise PackageError(problems)
    return pkg


# Failures that say something about the task. The rest (length, format, wrapper text,
# hidden characters, judge injection) are the writer's general habits: exactly what the
# benchmark catches, and no evidence about whether the task can be done.
TASK_SPECIFIC = ("verifier:must_include", "verifier:must_not_include", "verifier:material_copy",
                 "verifier:copy", "agent:")


def task_specific(failures: Sequence[str]) -> list[str]:
    return [f for f in failures if f.startswith(TASK_SPECIFIC)]


def render_notes(card: TaskCard) -> str:
    """Shared by the model prompt and the writer brief, so both see the same material."""
    n = card.notes_pack
    return "\n".join(f"- {i}" for i in (*n.facts, *n.anecdotes, *n.opinions, *n.wants))


def nop_outputs(pkg: TaskPackage) -> list[tuple[str, dict]]:
    """Outputs that must all score 0.0: nothing, a wrapper, and an echo of the input."""
    echo = pkg.user_message()
    wrapper = "Sure! Here's a draft you can use:\n\n[Your message here]\n\nI hope this helps!"
    if pkg.card.mode in AGENTIC:
        assert pkg.env is not None
        target = pkg.env.targets[0]
        tool = "send" if pkg.card.agent_checks and pkg.card.agent_checks.delivery == "send" else "create_draft"
        return [
            ("empty", {"run": None}),
            ("wrapper", {"run": OracleRun(actions=[Action(tool=tool, target=target, body=wrapper)], tool_calls=1)}),
            ("echo", {"run": OracleRun(actions=[Action(tool=tool, target=target, body=echo)], tool_calls=1)}),
        ]
    return [("empty", {"text": ""}), ("wrapper", {"text": wrapper}), ("echo", {"text": echo})]
