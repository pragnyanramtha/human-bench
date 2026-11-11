# Fixing audited task cards

You fix HumanBench task cards using the findings of a harsh audit. The goal is cards that a hostile
reviewer cannot fault. No cheating: never weaken a check just to make a finding go away, never edit
an audit file, and never make a card easier to hide a defect.

## Where things are
- Dev cards: `tasks/dev/<slot>/` (`card.yaml`, and for some modes `context.json`, `env/fixture.json`, `env/user_voice.json`).
- Reference solutions ("oracles", never shown to models): `private/oracle/<slot>/draft.md` or `actions.json`.
- Audit findings: `private/audit/<ROUND>/dev/<slot>.json`, where ROUND is the round named in your task (earlier rounds and their fix records are beside it, for history only). A card whose verdict there is `clean` must not be touched (severity blocker / major / minor, each with evidence and a proposed fix).
- Format and rules for cards: `pipelines/taskcards/instructions.txt` and `src/humanbench/schema.py`, `envs.py`.
- Check a card: `uv run python -m humanbench.cardcheck tasks/dev/<slot> --accepted tasks/dev` (must report no problems; it also scores the oracle 1.0 and the empty/wrapper/echo outputs 0.0).

## How scoring works (so you can judge a finding)
- `must_include` / `must_not_include` / `private_phrases` match whole words, case-insensitively, after stripping punctuation and normalising numbers, dates and times. A list entry means any one alternative counts.
- Agent checks: required writes (target, `in_reply_to` or `new_post`, `min_words`, per-write `must_include`), `forbidden_targets`, `private_phrases`, `max_writes` (default: the number of required writes), money amounts must come from the material or be a sum/difference of amounts there.
- A separate LLM quality floor checks by meaning: coherent; does the task, including every point in the card's `must_cover` list; tone; persona; and for workspace tasks that nothing from items marked `private: true` is revealed in any wording.
- So coverage goes in `must_cover` (card.yaml, top level): 1-8 plain statements (as few as cover the asks; most cards need 2-5) of what the piece has to get across, each something the prompt explicitly asks for, e.g. "asks for a refund or a replacement", "gives the reopening date and the temporary location". No keywords, no style. Models and writers never see it. Keep `must_include` to at most 4 literal facts the piece cannot do without, with the alternatives a real writer would use.
- Something the user explicitly asks to keep out of the piece goes in `must_not_reveal` (card.yaml, top level, up to 5 plain statements such as "that Raff's father was in hospital"); the floor checks it by meaning in every mode, including direct tasks. A countable constraint the user states ("tag everyone only once") goes in `must_cover` as a plain statement.
- A sensitive workspace item that the task expects to stay out of the piece must be marked `private: true` in env/fixture.json (then the floor catches paraphrases). `private_phrases` stay a short literal backstop of distinctive strings.
- The oracle is a reference solution. It must get across every `must_cover` point and must not state concrete events, numbers, names or history that are not in the material (feelings, greetings and small colour are fine).

## Card repair procedure
1. Read the card files, the oracle and the audit file.
2. For every blocker and major finding (and minors that are cheap), decide:
   - VALID: fix it with the smallest edit. Typical fixes: add alternatives to an anchor or drop an anchor the request does not require; remove a ban on natural wording; make names, dates, weekdays (check the 2026-2027 calendar), amounts and sums agree everywhere; correct facts in the notes, fixture or oracle; rewrite the part of the oracle that contradicts the material or invents facts; make the prompt state something that was only assumed (in the persona's own voice, keeping their English variety and writing level; no style coaching); set `max_writes` when extra messages are legitimately asked for.
   - INVALID (the audit misread the rules above, asks for a semantic check scripts cannot do, or is taste): leave the card alone and record why.
3. Keep everything else unchanged: do not rewrite prompts for style, do not change slot fields, persona, mode, band, or `provenance`. Do not try to fix "model-drafted context" (tracked elsewhere).
4. The oracle must stay a believable answer that does the whole task and passes every check honestly (no keyword stuffing). If you change a check, re-read the oracle against it.
5. Run the card check. It must pass before you move on.
6. Every card must have a `must_cover` list, and its oracle must be re-read against it. A `must_cover` point may only require what the prompt explicitly asks for: if the audit says a point is stricter than the request (it turns an optional offer into a requirement, widens a deadline, demands background detail), narrow or drop it rather than defend it. YAML/JSON must stay valid with no duplicate keys.
7. Write `private/audit/<ROUND>/fixes/<slot>.md` (create the directory): one line per finding from the newest audit of that card: `FIXED: <what changed>` or `REJECTED: <reason>`.

After repairs, run `uv run pytest -q` and `uv run python scripts/freeze_check.py`.
Model-drafted context is a stated v0.1 limitation and does not cause freeze failure.
Record counts of fixed and rejected findings per card and reasons for rebuilding any card.
