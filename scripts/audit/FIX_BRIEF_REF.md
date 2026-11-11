# Fixing audited reference-matched cards

Read `scripts/audit/FIX_BRIEF.md` first: its rules about scoring, `must_cover`, `must_not_reveal`,
`must_include`, honesty and records all apply. This file says what is different for reference cards.

## What a reference card is
- Cards: `tasks/ref/<slot>/card.yaml` (direct mode only). Each was derived by a model from one real piece of
  writing by a real person (pre-2022). That piece is the card's oracle: `private/oracle/<slot>/draft.md`,
  with `provenance.json` beside it. A model under test gets only the prompt and notes and is compared with
  the real piece.
- **The oracle is real human text and must never be edited, trimmed or "corrected".** Every fix is to the card.
- Audit findings: `private/audit/<ROUND>/ref/<slot>.json`. Records go to `private/audit/<ROUND>/fixes/<slot>.md`.

## How to fix the usual findings
- Literal anchors (`must_include`) that a careful writer could miss: remove them. Reference cards should
  normally have `must_include: []`; keep an anchor only when the prompt explicitly demands that exact string
  and the real piece contains it. Coverage belongs in `must_cover`.
- `must_cover` (2-5 points): each must be something the prompt explicitly asks for AND something the real
  piece clearly does. If the real piece does not clearly do it, drop or reword the point (never the piece).
  If the audit says a lazy answer could skip something the prompt asks for, make the prompt's ask and the
  point match.
- Leakage: no note or prompt sentence may reproduce the piece's sentences, jokes, images, analogies or
  distinctive phrases, and the notes must not follow the piece's order. Rewrite such notes as plain facts
  in a different order. Check: no run of 8+ words shared with the piece.
- Sufficiency: add to the notes any substantive fact, event or opinion the real piece uses that a writer
  could not otherwise know (as short raw notes, max 30 words each, at most 30 notes). Remove notes, persona
  details or reasons-for-writing that the piece does not support. Persona fields: only what the text
  supports. `age` and `lives` may each be `null` when the text gives no basis (then they are simply not shown),
  and `english` may be `unknown`. Do not guess an age, a place, a dialect or a job the text does not support;
  `role` should be the broadest description the text supports ("forum member who studied art history").
  `audience` and `stakes` must be the closest allowed values; if the audit calls one unsupported, pick the
  most neutral allowed value and say so in the record.
- Prompt: first person, in this writer's voice and level, a person asking for help before writing; not a
  brief by someone who has read the finished piece. No style coaching.
- Never put phone numbers, email addresses, street addresses or account numbers in the card.
- Keep `split: ref`, `mode: direct`, `id`, `channel`, `band` and `format_overrides` unless a finding is
  specifically about them.

## When to drop a card
Write `DROP` on the first line of the fix record, with the reason, and make no further edits, if the real
piece itself is unsuitable and no card edit can cure it: factual errors or unsafe advice in the piece that a
fair task would have to reproduce, a damaging claim about a named private person, personal contact details
in the piece, published journalism or marketing, mostly quoted material, or a piece too thin or too
dependent on missing context to brief fairly.

## Checking a card
```
uv run python - <<'PY'
from pathlib import Path
from humanbench.taskpkg import load_package, nop_outputs
from humanbench.scoring.verifiers import copied_share
slot = "ref-email-01"   # change
pkg = load_package(Path("tasks/ref") / slot, require_oracle=False)
text = (Path("private/oracle") / slot / "draft.md").read_text()
print("oracle:", pkg.score(text=text))            # must be (1.0, [])
print("no-ops:", [pkg.score(**kw)[0] for _, kw in nop_outputs(pkg)])   # must all be 0.0
notes = [n for v in pkg.card.notes_pack.model_dump().values() for n in v]
print("copy share:", copied_share("\n".join([pkg.card.prompt, *notes]), [text], n=8))   # must be 0.0
PY
```
Run the full pytest suite after repairs.
