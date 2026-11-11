"""Constraint sampler for the v1 task matrix.

Produces 160 task slots (channel, mode, tone, audience, stakes, length band,
persona, split) that satisfy every rule in `humanbench.checks`. People then
write the actual prompt, notes and environment for each slot.

    uv run humanbench-matrix --out matrix/slots.json --report matrix/coverage.md
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from humanbench import axes
from humanbench.axes import Band, Channel, Mode, Stakes, Tone, WritingLevel
from humanbench.checks import check_slots, coverage_report
from humanbench.schema import Persona, Slot, load_personas


@dataclass
class _Draft:
    channel: Channel
    mode: Mode
    band: Band | None = None
    tone: Tone | None = None
    audience: axes.Audience | None = None
    stakes: Stakes | None = None
    persona_id: str | None = None
    split: str = "test"


class SamplingError(RuntimeError):
    pass


def _round_robin(drafts: list[_Draft], rng: random.Random) -> list[_Draft]:
    """Order drafts so channels alternate, which spreads any 'take the first k' pick."""
    per_channel: dict[Channel, list[_Draft]] = defaultdict(list)
    for d in drafts:
        per_channel[d.channel].append(d)
    queues = list(per_channel.values())
    for q in queues:
        rng.shuffle(q)
    rng.shuffle(queues)
    out: list[_Draft] = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop())
    return out


def _assign_bands(drafts: list[_Draft], rng: random.Random) -> None:
    # Most constrained bands first; medium is allowed everywhere and takes the rest.
    for band in (Band.EXTENDED, Band.LONG, Band.SHORT):
        candidates = [
            d for d in drafts if d.band is None and band in axes.ALLOWED_BANDS[(d.channel, d.mode)]
        ]
        need = axes.BAND_TARGETS[band]
        if len(candidates) < need:
            raise SamplingError(f"not enough slots can take band {band}")
        for d in _round_robin(candidates, rng)[:need]:
            d.band = band
    for d in drafts:
        if d.band is None:
            d.band = Band.MEDIUM


def _assign_tones(drafts: list[_Draft], rng: random.Random) -> None:
    # Each channel gets 3 of four tones and 2 of the other four (sum 20); a
    # circulant layout makes every tone total 20 across channels too.
    tones = list(Tone)
    channels = list(Channel)
    rng.shuffle(tones)
    rng.shuffle(channels)
    n = len(tones)
    for i, channel in enumerate(channels):
        plan: list[Tone] = []
        for j, tone in enumerate(tones):
            plan += [tone] * (3 if (j - i) % n < n // 2 else 2)
        rng.shuffle(plan)
        cdrafts = [d for d in drafts if d.channel is channel]
        if len(plan) != len(cdrafts):
            raise SamplingError("tone plan size mismatch")
        for d, tone in zip(cdrafts, plan, strict=True):
            d.tone = tone


def _assign_audience_and_stakes(drafts: list[_Draft], rng: random.Random) -> None:
    for channel in Channel:
        cdrafts = [d for d in drafts if d.channel is channel]
        rng.shuffle(cdrafts)
        allowed = list(axes.ALLOWED_AUDIENCES[channel])
        rng.shuffle(allowed)
        for i, d in enumerate(cdrafts):
            d.audience = allowed[i % len(allowed)]
        stakes = [k for k, n in axes.STAKES_PER_CHANNEL.items() for _ in range(n)]
        rng.shuffle(stakes)
        for d, k in zip(cdrafts, stakes, strict=True):
            d.stakes = k


def _assign_personas(drafts: list[_Draft], personas: list[Persona], rng: random.Random) -> None:
    uses: Counter[str] = Counter()
    in_channel: dict[Channel, set[str]] = defaultdict(set)
    tones_of: dict[str, set[Tone]] = defaultdict(set)
    levels_in: dict[Channel, set[WritingLevel]] = defaultdict(set)

    # Most constrained channels first (fewest eligible personas), random within.
    def eligible_count(d: _Draft) -> int:
        return sum(1 for p in personas if d.channel not in p.excluded_channels)

    order = drafts[:]
    rng.shuffle(order)
    order.sort(key=eligible_count)

    for d in order:
        assert d.tone is not None
        options = [
            p
            for p in personas
            if uses[p.id] < axes.MAX_PERSONA_USES
            and p.id not in in_channel[d.channel]
            and d.tone not in tones_of[p.id]
            and d.channel not in p.excluded_channels
        ]
        if not options:
            raise SamplingError(f"no persona left for {d.channel}/{d.tone}")
        missing_levels = set(WritingLevel) - levels_in[d.channel]
        rng.shuffle(options)
        # Prefer least-used personas, then ones that fill a missing writing level.
        options.sort(key=lambda p: (uses[p.id], p.writing_level not in missing_levels))
        pick = options[0]
        d.persona_id = pick.id
        uses[pick.id] += 1
        in_channel[d.channel].add(pick.id)
        tones_of[pick.id].add(d.tone)
        levels_in[d.channel].add(pick.writing_level)


def _assign_split(drafts: list[_Draft], rng: random.Random) -> None:
    for channel in Channel:
        cdrafts = [d for d in drafts if d.channel is channel]
        rng.shuffle(cdrafts)
        cdrafts.sort(key=lambda d: (d.mode.value, d.band.value if d.band else ""))
        step = len(cdrafts) // axes.DEV_PER_CHANNEL
        offset = rng.randrange(step)
        for i in range(axes.DEV_PER_CHANNEL):
            cdrafts[offset + i * step].split = "dev"


def _sample_once(personas: list[Persona], rng: random.Random) -> list[Slot]:
    drafts = [
        _Draft(channel=channel, mode=mode)
        for channel, modes in axes.CHANNEL_MODE_COUNTS.items()
        for mode, n in modes.items()
        for _ in range(n)
    ]
    _assign_bands(drafts, rng)
    _assign_tones(drafts, rng)
    _assign_audience_and_stakes(drafts, rng)
    _assign_personas(drafts, personas, rng)
    _assign_split(drafts, rng)

    slots: list[Slot] = []
    counters: Counter[Channel] = Counter()
    for d in sorted(drafts, key=lambda d: (list(Channel).index(d.channel), d.mode.value, d.band.value)):
        counters[d.channel] += 1
        slots.append(
            Slot(
                id=f"{d.channel.value}-{counters[d.channel]:02d}",
                split=d.split,  # type: ignore[arg-type]
                channel=d.channel,
                mode=d.mode,
                tone=d.tone,  # type: ignore[arg-type]
                audience=d.audience,  # type: ignore[arg-type]
                stakes=d.stakes,  # type: ignore[arg-type]
                band=d.band,  # type: ignore[arg-type]
                persona_id=d.persona_id,  # type: ignore[arg-type]
            )
        )
    return slots


def sample_matrix(
    seed: int = 1, personas: list[Persona] | None = None, max_attempts: int = 500
) -> tuple[list[Slot], int]:
    """Return slots that pass every check, plus the attempt seed that produced them."""
    personas = personas if personas is not None else load_personas()
    last_errors: list[str] = []
    for attempt in range(max_attempts):
        attempt_seed = seed * 100_003 + attempt
        rng = random.Random(attempt_seed)
        try:
            slots = _sample_once(personas, rng)
        except SamplingError as e:
            last_errors = [str(e)]
            continue
        last_errors = check_slots(slots, personas)
        if not last_errors:
            return slots, attempt_seed
    raise SamplingError(f"no valid matrix after {max_attempts} attempts; last errors: {last_errors[:5]}")


def promote_to_dev(slots: list[Slot], per_channel: int, seed: int = 1) -> list[Slot]:
    """Move test slots to dev until each channel has `per_channel` dev slots, keeping every
    existing dev slot (their cards may already be public). Each pick adds the most
    mode / tone / band / audience values the channel's dev set doesn't have yet."""
    rng = random.Random(seed)
    out = [s.model_copy() for s in slots]
    for channel in Channel:
        mine = [i for i, s in enumerate(out) if s.channel is channel]
        dev = [i for i in mine if out[i].split == "dev"]
        pool = [i for i in mine if out[i].split == "test"]
        rng.shuffle(pool)
        while len(dev) < per_channel and pool:
            def gain(i: int) -> int:
                s = out[i]
                return sum(
                    getattr(s, f) not in {getattr(out[d], f) for d in dev}
                    for f in ("mode", "tone", "band", "audience", "stakes")
                )
            pick = max(pool, key=gain)
            pool.remove(pick)
            out[pick] = out[pick].model_copy(update={"split": "dev"})
            dev.append(pick)
    return out


def load_slots(public: Path, private: Path | None = None) -> list[dict]:
    """Read public assignments and, when available, the held-back assignments."""
    slots = json.loads(public.read_text())["slots"]
    if private is not None and private.is_file():
        slots += json.loads(private.read_text())["slots"]
    return slots


def public_coverage(slots: list[Slot], personas: list[Persona]) -> str:
    """Publish public coverage; never publish private assignments or sampler seeds."""
    public = [s for s in slots if s.split == "dev"]
    private = [s for s in slots if s.split == "test"]
    detail = coverage_report(public, personas) if public else "# Task matrix coverage\n"
    return detail + f"\nPublic slots: {len(public)}. Private slots: {len(private) if private else 'not available'}.\nPrivate assignments are held back; only their total count is reported.\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input", type=Path, help="report an existing public matrix instead of generating")
    parser.add_argument("--private-out", type=Path, default=Path("private/matrix/slots-test.json"))
    parser.add_argument("--out", type=Path, default=Path("matrix/slots.json"))
    parser.add_argument("--report", type=Path, default=Path("matrix/coverage.md"))
    args = parser.parse_args(argv)

    personas = load_personas()
    if args.input:
        slots = [Slot.model_validate(s) for s in load_slots(args.input, args.private_out)]
    else:
        # Random by default: a published fixed seed would reconstruct private assignments.
        seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**63)
        slots, used_seed = sample_matrix(seed, personas)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"slots": [s.model_dump(mode="json") for s in slots if s.split == "dev"]}, indent=2) + "\n")
        args.private_out.parent.mkdir(parents=True, exist_ok=True)
        args.private_out.write_text(json.dumps({"seed": seed, "attempt_seed": used_seed,
            "slots": [s.model_dump(mode="json") for s in slots if s.split == "test"]}, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(public_coverage(slots, personas), encoding="utf-8")
    print(f"wrote public coverage to {args.report}")



if __name__ == "__main__":
    main()
