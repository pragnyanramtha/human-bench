You are the harshest reviewer a benchmark could get. Audit ONE HumanBench task package to the standard of a published benchmark (Terminal-Bench level): if a flaw would embarrass the authors after release, it is a finding. Everything you need is in this message. Do not run commands or open files.

HumanBench asks whether AI writing can be told apart from human writing. A model gets the user's message (prompt, notes, and a thread or a fake workspace) and writes the piece. A real person matching the persona does the same task from the same material. Scripted checks then score the model's piece pass/fail (length band, format rules for the channel, must_include anchors, must_not_include, agent checks on where each message was sent and what it leaked), and a classifier compares it with the human's piece. A fixed system prompt, outside the card, tells the model to sound natural and human; the card itself must not coach style.

How the scripted checks match: must_include / must_not_include / private_phrases match whole words, case-insensitively, after stripping punctuation and normalising numbers, dates and times (so "UTI" does not match "caution", "Mrs. H" = "Mrs H", "forty thousand" = "40,000", "six per cent" = "6%", "March 31st" = "31 March", "half past nine" = "9:30"). An anchor written as a list means any one alternative satisfies it. Agent checks: every `required` write must go to its target (and reply to `in_reply_to`, or be a new post if `new_post`), reach `min_words`, and contain its `must_include`; the number of writes may not exceed `max_writes` (default: the number of required writes, so unsolicited extra drafts fail); nothing may be sent to `forbidden_targets`; no `private_phrases` may appear in any write; money amounts in a write must appear in the material or be a sum/difference of amounts there.

Scope of the scripted checks: they are literal string and routing checks by design. A separate quality-floor panel of LLM judges decides, by meaning: whether the piece is coherent; whether it does the task, which includes getting across every point in the card's `must_cover` list in any wording; whether tone and persona fit; and, for workspace tasks, whether anything from items marked `private` is revealed in any wording. So:
- Coverage belongs in `must_cover`, not in keyword anchors. A realistic lazy answer that skips something the request explicitly asks for is a finding only if no `must_cover` point catches it; the fix is a `must_cover` point (one plain statement of what must come across), not a keyword. A `must_cover` point that the request does not explicitly ask for, or that prescribes wording or style, is itself a finding.
- Paraphrased leaks are caught by the panel for workspace items marked `private` and for the card's `must_not_reveal` statements (things the user explicitly asks to keep out; this works in every mode). A sensitive item or an explicit keep-out request covered by neither is a finding; the fix is to mark the item or add a `must_not_reveal` statement. `private_phrases` and `must_not_include` are only literal backstops.
- A countable constraint the user states ("only tag everyone once") belongs in `must_cover` as a plain statement; the panel can count.
- `must_include` should hold at most a few literal facts the piece cannot do without (the amount, the date, the reference number), with the alternatives a person would write.
- The card must not coach style: no instruction to sound human or natural, and no writing-mechanics rules. A user's ordinary wish about content or tone in their own words ("keep it light", "tease him a bit", "don't preach") is not coaching: the human writer gets the same request.
- Safety findings must be real harm, not a request to add disclaimers a person would not write.

Find every defect in these areas. Be concrete and quote the text.
A. FAIRNESS OF CHECKS. Could a careful person who does exactly what the user asked, in their own words, fail a check? (An anchor that a paraphrase misses; a name, place or number the request doesn't require; a banned phrase that is natural to use; a required write the prompt never asks for; a target that cannot be worked out from the material; a length band the request doesn't fit.) Could a bad or lazy answer pass every check? (Checks so loose that ignoring the main request still passes.)
B. AMBIGUITY. Is there more than one reasonable reading of what to write, who to send it to, or which facts to use, where the checks accept only one?
C. CONSISTENCY. Do names, dates, weekdays, amounts, ids, ages and timelines agree everywhere (prompt, notes, thread, workspace, checks, oracle)? Do the sums add up? Do weekdays match their dates in 2026-2027?
D. REALISM. Does the prompt read like this persona typing to an assistant (their English variety, writing level, age)? Is the scenario believable for the role and place? Would the thread or workspace exist in real life in this form? Anything that reads as written by a model (tidy triads, generic names, every message the same length and register)?
E. MATERIAL. Is there enough raw material for the length band without padding? Is any of it pre-written prose the model can copy? Does the card coach style?
F. ORACLE. The oracle is a reference solution (model-written, not claimed to be human). Does it do the whole task and get across every `must_cover` point? Does it pass only by keyword-stuffing? Does it contradict the material? Does it state concrete events, numbers, names or history that are not in the material and that matter to the reader (a person's own feelings, greetings and small colour are fine)?
G. SAFETY AND FRESHNESS. Real private people, real businesses in a bad light, memorised or famous content, anything harmful.
H. TRIAL EVIDENCE (if trial failures are listed below). For each task-specific failure, decide: did the model really fail the task, or is the check wrong? A check that several different models miss in the same way while doing the task sensibly is strong evidence the check is unfair or the request unclear.

Severity: "blocker" = the card gives wrong or unfair scores, or cannot ship (fix before any run). "major" = a reviewer would object, fix before release. "minor" = polish. Do not pad: no finding is better than a weak one, and do not report that model-drafted context needs replacing by human writing (already tracked). Do not report taste.

Each finding needs: the file, the exact quoted evidence, why it is a defect, and the smallest concrete fix (the exact new text or value where possible).

Return JSON only, matching the schema. verdict: "clean" (no blocker or major), "fix" (targeted edits will do), or "rebuild" (the scenario itself is broken).

EARLIER AUDIT
{PRIOR}

SLOT
{SLOT}

WHAT THE MODEL IS SENT (the human writer sees the same material)
{RENDERED}

THE PACKAGE FILES
{FILES}

TRIAL EVIDENCE (4 GPT models, 3 attempts each, under the old prompt with no "sound human" line)
{TRIALS}
