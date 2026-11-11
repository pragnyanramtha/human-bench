"""Variance and coverage rules for the task matrix (DESIGN.md section 2.4).

`check_slots` returns a list of human-readable violations; empty means the
matrix satisfies every rule.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from humanbench import axes
from humanbench.axes import Band, Channel, Stakes, Tone, WritingLevel
from humanbench.schema import Persona, Slot


def check_slots(slots: Sequence[Slot], personas: Sequence[Persona]) -> list[str]:
    by_id = {p.id: p for p in personas}
    errors: list[str] = []

    unknown = sorted({s.persona_id for s in slots} - by_id.keys())
    if unknown:
        return [f"unknown persona ids: {unknown}"]

    ids = [s.id for s in slots]
    if len(ids) != len(set(ids)):
        errors.append("duplicate slot ids")
    if len(slots) != axes.total_tasks():
        errors.append(f"expected {axes.total_tasks()} slots, got {len(slots)}")

    by_channel: dict[Channel, list[Slot]] = defaultdict(list)
    for s in slots:
        by_channel[s.channel].append(s)

    for channel, mode_counts in axes.CHANNEL_MODE_COUNTS.items():
        got = Counter(s.mode for s in by_channel[channel])
        if dict(got) != mode_counts:
            errors.append(f"{channel}: mode counts {dict(got)} != {mode_counts}")

    bands = Counter(s.band for s in slots)
    if dict(bands) != axes.BAND_TARGETS:
        errors.append(f"band counts {dict(bands)} != {axes.BAND_TARGETS}")
    over_400 = sum(1 for s in slots if s.band is not Band.SHORT)
    if over_400 * 2 < len(slots):
        errors.append(f"only {over_400}/{len(slots)} tasks are over 400 words")

    tones = Counter(s.tone for s in slots)
    expected_per_tone = len(slots) // len(Tone)
    for tone in Tone:
        if tones[tone] != expected_per_tone:
            errors.append(f"tone {tone}: {tones[tone]} tasks, expected {expected_per_tone}")

    uses = Counter(s.persona_id for s in slots)
    for pid, n in uses.items():
        if n > axes.MAX_PERSONA_USES:
            errors.append(f"persona {pid} used {n} times (max {axes.MAX_PERSONA_USES})")

    seen_channel_persona: set[tuple[Channel, str]] = set()
    seen_persona_tone: set[tuple[str, Tone]] = set()
    for s in slots:
        key_cp = (s.channel, s.persona_id)
        key_pt = (s.persona_id, s.tone)
        if key_cp in seen_channel_persona:
            errors.append(f"persona {s.persona_id} appears twice in {s.channel}")
        if key_pt in seen_persona_tone:
            errors.append(f"persona {s.persona_id} appears twice with tone {s.tone}")
        seen_channel_persona.add(key_cp)
        seen_persona_tone.add(key_pt)
        if s.channel in by_id[s.persona_id].excluded_channels:
            errors.append(f"{s.id}: persona {s.persona_id} is excluded from {s.channel}")

    for channel, cslots in by_channel.items():
        n_tones = len({s.tone for s in cslots})
        if n_tones < axes.MIN_TONES_PER_CHANNEL:
            errors.append(f"{channel}: only {n_tones} tones")
        n_aud = len({s.audience for s in cslots})
        if n_aud < axes.min_audiences(channel):
            errors.append(f"{channel}: only {n_aud} audiences")
        stakes = Counter(s.stakes for s in cslots)
        if dict(stakes) != axes.STAKES_PER_CHANNEL:
            errors.append(f"{channel}: stakes {dict(stakes)} != {axes.STAKES_PER_CHANNEL}")
        levels = {by_id[s.persona_id].writing_level for s in cslots}
        if levels != set(WritingLevel):
            errors.append(f"{channel}: writing levels {sorted(levels)} missing some")
        ages = [by_id[s.persona_id].age for s in cslots]
        if max(ages) - min(ages) < axes.MIN_AGE_SPAN_PER_CHANNEL:
            errors.append(f"{channel}: age span {min(ages)}-{max(ages)} too narrow")
        n_dev = sum(1 for s in cslots if s.split == "dev")
        if n_dev != axes.DEV_PER_CHANNEL:
            errors.append(f"{channel}: {n_dev} dev tasks, expected {axes.DEV_PER_CHANNEL}")

    non_us = sum(1 for s in slots if by_id[s.persona_id].non_us) / len(slots)
    if non_us < axes.MIN_NON_US_SHARE:
        errors.append(f"non-US English share {non_us:.0%} < {axes.MIN_NON_US_SHARE:.0%}")
    lo, hi = axes.NON_NATIVE_SHARE_RANGE
    non_native = sum(1 for s in slots if not by_id[s.persona_id].native) / len(slots)
    if not lo <= non_native <= hi:
        errors.append(f"non-native share {non_native:.0%} outside {lo:.0%}-{hi:.0%}")

    return errors


def coverage_report(slots: Sequence[Slot], personas: Sequence[Persona], seed: int | None = None) -> str:
    """Markdown summary of how the matrix covers every axis."""
    by_id = {p.id: p for p in personas}
    channels = list(Channel)
    lines = ["# Task matrix coverage", ""]
    if seed is not None:
        lines += [f"Generated with seed `{seed}`. {len(slots)} slots.", ""]

    def table(title: str, cols: Sequence[str], value) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| channel | " + " | ".join(cols) + " | total |")
        lines.append("|---" * (len(cols) + 2) + "|")
        for ch in channels:
            cs = [s for s in slots if s.channel is ch]
            counts = Counter(value(s) for s in cs)
            lines.append(f"| {ch} | " + " | ".join(str(counts.get(c, 0) or "") for c in cols) + f" | {len(cs)} |")
        lines.append("")

    table("Modes", [m.value for m in axes.Mode], lambda s: s.mode.value)
    table("Length bands", [b.value for b in Band], lambda s: s.band.value)
    table("Tones", [t.value for t in Tone], lambda s: s.tone.value)
    table("Audiences", [a.value for a in axes.Audience], lambda s: s.audience.value)
    table("Stakes", [k.value for k in Stakes], lambda s: s.stakes.value)
    table("Writing level", [w.value for w in WritingLevel], lambda s: by_id[s.persona_id].writing_level.value)
    table("Split", ["dev", "test"], lambda s: s.split)

    uses = Counter(s.persona_id for s in slots)
    hist = Counter(uses.values())
    unused = [p.id for p in personas if p.id not in uses]
    non_us = sum(1 for s in slots if by_id[s.persona_id].non_us) / len(slots)
    non_native = sum(1 for s in slots if not by_id[s.persona_id].native) / len(slots)
    over_400 = sum(1 for s in slots if s.band is not Band.SHORT) / len(slots)
    lines += [
        "## Personas",
        "",
        f"- {len(uses)} personas used; uses per persona: "
        + ", ".join(f"{k}×: {v}" for k, v in sorted(hist.items())),
        f"- unused: {', '.join(unused) or 'none'}",
        f"- non-US English: {non_us:.0%}; non-native writers: {non_native:.0%}",
        f"- tasks over 400 words: {over_400:.0%}",
        "",
        "## Age span per channel",
        "",
    ]
    for ch in channels:
        ages = [by_id[s.persona_id].age for s in slots if s.channel is ch]
        lines.append(f"- {ch}: {min(ages)}–{max(ages)}")
    lines.append("")
    return "\n".join(lines)
