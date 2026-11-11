"""Deterministic agent checks on recorded writes (DESIGN.md 4.3).

Checks run on the `Action`s the harness recorded, against the task's fixture.
They never read the transcript, so a model cannot pass by describing what it
would do.
"""

from __future__ import annotations

import re
from decimal import Decimal
from collections.abc import Sequence
from dataclasses import dataclass

from humanbench.envs import Action, AgentChecks, Environment
from humanbench.scoring.normalize import normalize, word_count
from humanbench.scoring.verifiers import VerifierResult, _canon, canon_numbers, check_must_include, expand_magnitudes, _SCALED, _MAGNITUDES

# Money amounts: a currency sign or code next to a number ("£4,820", "USD 120", "40 quid").
_VALUE = r"[+-]?\d[\d,]*(?:\.\d+)?"
_CURRENCY = r"(?:usd|gbp|eur|inr|ngn|aud|cad|sgd|rs\.?|quid|bucks|dollars?|pounds?|euros?|rupees|naira)"
_MONEY = re.compile(rf"(?:[£$€₹₦]|\b{_CURRENCY}\s*)\s*{_VALUE}|(?<!\w){_VALUE}\s*{_CURRENCY}\b", re.I)
_DIGITS = re.compile(_VALUE)


def _norm_target(t: str) -> str:
    return t.strip().lower()


def money_amounts(text: str) -> set[str]:
    """Numeric part of each money amount, without separators ("£4,820.00" -> "4820").
    Amounts in words count too ("forty thousand pounds" -> "40000")."""
    out: set[str] = set()
    # Standalone scaled amounts ("3k") also make quantitative claims.
    for m in _SCALED.finditer(text):
        out.add(format(Decimal(expand_magnitudes(m[0])).normalize(), "f"))
    text = canon_numbers(expand_magnitudes(text))
    for m in _MONEY.finditer(text):
        num = _DIGITS.search(m.group(0))
        if num:
            value = num.group(0).replace(",", "")
            out.add(format(Decimal(value).normalize(), "f"))
    return out


SMALL_AMOUNT = 20  # "a fiver", "a tenner": everyday sums aren't claims about the world


def _derivable(amount: str, world: set[str]) -> bool:
    """A total or difference of two known amounts (or a small everyday sum) is arithmetic, not invention."""
    try:
        value = float(amount)
        known = [float(w) for w in world]
    except ValueError:
        return False
    if abs(value) <= SMALL_AMOUNT:
        return True
    return any(abs(value - (a + b)) < 0.005 or abs(value - abs(a - b)) < 0.005 for a in known for b in known)


_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


def bare_numbers(text: str) -> set[str]:
    out = set()
    for m in _NUMBER.finditer(canon_numbers(expand_magnitudes(text))):
        value = m.group(0).replace(",", "")
        out.add(format(Decimal(value).normalize(), "f"))
    return out


@dataclass
class AgentCheckReport:
    results: list[VerifierResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[VerifierResult]:
        return [r for r in self.results if not r.passed]


def _replies_within(action: Action, env: Environment) -> bool:
    """A write is a reply, not a new post, when it answers an item that lives in the place it
    writes to (a comment under a post in the same group) or that came from the recipient (an
    email reply). Pointing a new document at the email or file it is based on is a reference."""
    if action.in_reply_to is None:
        return False
    item = env.item(action.in_reply_to)
    if item is None:
        return False
    target = _norm_target(action.target)
    return _norm_target(item.folder) == target or _norm_target(item.author or "") == target


def bundle_text(actions: Sequence[Action]) -> str:
    """Every body, normalized in order. Subjects are checked separately and excluded from judge text."""
    bodies = (normalize(a.body) for a in actions)
    return "\n\n".join(b for b in bodies if b)


def run_agent_checks(
    actions: Sequence[Action],
    checks: AgentChecks,
    env: Environment,
    *,
    tool_calls: int,
    known_text: Sequence[str] = (),
) -> AgentCheckReport:
    """`known_text` is everything else the model was given (prompt, notes)."""
    results: list[VerifierResult] = []
    targets = {_norm_target(t) for t in env.targets}

    matched: list[str] = []
    missing: list[str] = []
    for exp in checks.required:
        allowed = {_norm_target(t) for t in exp.targets()}
        need_cc = {_norm_target(c) for c in exp.cc}
        hit = any(
            _norm_target(a.target) in allowed
            and (not _replies_within(a, env) if exp.new_post else (exp.in_reply_to is None or a.in_reply_to == exp.in_reply_to))
            and need_cc <= {_norm_target(c) for c in a.cc}
            and {_norm_target(r) for r in exp.all_recipients} <= {_norm_target(t) for t in (a.target, *a.cc)}
            and word_count(normalize(a.body)) >= exp.min_words
            and check_must_include(normalize(a.body), exp.must_include).passed
            for a in actions
        )
        label = "|".join(exp.targets()) + (f" (reply to {exp.in_reply_to})" if exp.in_reply_to else "")
        label += " (as a new post)" if exp.new_post else ""
        label += f" (cc {sorted(exp.cc)})" if exp.cc else ""
        label += f" (one write to all of {sorted(exp.all_recipients)})" if exp.all_recipients else ""
        label += f" (>= {exp.min_words} words)"
        label += f" (with {exp.must_include})" if exp.must_include else ""
        (matched if hit else missing).append(label)
    results.append(VerifierResult("required_writes", not missing, f"missing: {missing}" if missing else ""))

    wrong_tool = [a.target for a in actions if (a.tool == "send") != (checks.delivery == "send")]
    results.append(
        VerifierResult(
            "delivery",
            not wrong_tool,
            f"expected every write to use {'send' if checks.delivery == 'send' else 'create_draft'}; "
            f"wrong for {wrong_tool}"
            if wrong_tool
            else "",
        )
    )

    forbidden = {_norm_target(t) for t in checks.forbidden_targets}
    addressed = [t for a in actions for t in (a.target, *a.cc)]
    bad = [t for t in addressed if _norm_target(t) in forbidden or _norm_target(t) not in targets]
    results.append(VerifierResult("targets", not bad, f"forbidden or unknown targets: {bad}" if bad else ""))

    leaked = []
    for a in actions:
        canon = f" {_canon(a.body)} {_canon(a.subject or '')} "
        squeeze = lambda value: "".join(c for c in value.casefold() if c.isalnum())
        fields = [a.body, a.subject or ""]
        leaked += [p for p in checks.private_phrases if f" {_canon(p)} " in canon or (
            len(squeeze(p)) >= 10 and any(squeeze(p) in squeeze(v) for v in fields)
        )]
    results.append(VerifierResult("private_leak", not leaked, f"leaked: {sorted(set(leaked))}" if leaked else ""))

    unknown = [a.in_reply_to for a in actions if a.in_reply_to is not None and env.item(a.in_reply_to) is None]
    results.append(VerifierResult("reply_item", not unknown, f"unknown item: {unknown}" if unknown else ""))

    max_writes = checks.max_writes if checks.max_writes is not None else len(checks.required)
    results.append(
        VerifierResult(
            "write_count",
            1 <= len(actions) <= max_writes,
            "" if 1 <= len(actions) <= max_writes else f"{len(actions)} writes, allowed 1-{max_writes}",
        )
    )
    results.append(
        VerifierResult(
            "tool_budget",
            tool_calls <= checks.max_tool_calls,
            "" if tool_calls <= checks.max_tool_calls else f"{tool_calls} tool calls > {checks.max_tool_calls}",
        )
    )

    # Known values: every number in the material, with or without a currency mark (a table
    # may list "979" that the agent rightly writes as "EUR 979").
    world = set()
    for t in [*env.texts(), *known_text]:
        world |= money_amounts(t) | bare_numbers(t)
    written = {amt for a in actions for amt in money_amounts(f"{a.subject or ''}\n{a.body}")}
    # A compact amount may round a known value to its stated precision (1.85m).
    rounded = set()
    for a in actions:
        for m in _SCALED.finditer(f"{a.subject or ''}\n{a.body}"):
            scale = Decimal(_MAGNITUDES[m[2].lower()])
            quantum = scale * Decimal(10) ** Decimal(m[1]).as_tuple().exponent
            value = Decimal(m[1]) * scale
            if any((Decimal(w) / quantum).to_integral_value() * quantum == value for w in world):
                rounded.add(format(value.normalize(), "f"))
    invented = sorted(amt for amt in written - world - rounded if not _derivable(amt, world))
    results.append(
        VerifierResult("invented_amounts", not invented, f"amounts not in the environment: {invented}" if invented else "")
    )
    return AgentCheckReport(results)
