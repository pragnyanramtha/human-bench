# Data sourcing: operator guide

Code: `src/humanbench/sourcing/`. CLI: `humanbench-source`. Everything it writes goes under `private/`, which is gitignored (and gets its own catch-all `private/.gitignore`). Never commit or upload anything from `private/`.

There are two lanes, and they never mix:

| Lane | What | Used for | Never used for |
|---|---|---|---|
| 1. Public corpus | Pre-LLM human text from Hugging Face | Checking the judge on real human writing (how often it flags people as AI); public dev-split material | **Test references.** Models may have memorized archival text. |
| 2. Commissioned writing | New pieces from screened, paid writers | Test references (and new dev references) | Publication of test references |

The corpus commands need the optional extra. Use `uv run --extra sourcing ...`; a plain `uv run` re-syncs the environment and removes it. The tests don't need the extra.

## Lane 1: public corpus

### The registry

`src/humanbench/sourcing/sources.yaml` lists each source with a pinned Hugging Face revision, the fields to read, its licence, a licence status, and its allowed uses. `test_ref` is not a valid use; the schema rejects it. `dev_ref` is allowed only with `licence_status: clear`, because the dev split is public.

```
uv run humanbench-source sources                      # the table, plus channels with no source and why
uv run --extra sourcing humanbench-source verify      # re-check every repo, pinned sha and file glob on HF
```

| Channel | Source | Licence status | Uses |
|---|---|---|---|
| email | `corbt/enron-emails` (sent folders only) | questionable | judge_train |
| chat | none: no single-author 250+ word chat text with a usable licence | | |
| forum | `webis/tldr-17` (TL;DR stripped) | questionable | judge_train |
| forum | `HuggingFaceH4/stack-exchange-preferences` (question bodies) | clear (CC-BY-SA) | judge_train, dev_ref |
| social | none: tweets are too short, long posts exist only as unlicensed scrapes | | |
| review | `stanfordnlp/imdb` | questionable | judge_train |
| review | `Yelp/yelp_review_full` | restrictive (non-commercial academic only; used only for private non-commercial research) | judge_train |
| blog | `tasksource/blog_authorship_corpus` | questionable | judge_train |
| work_docs | `common-pile/github_archive` (single-author issue/PR threads) | questionable | judge_train |
| personal | `euclaise/writingprompts` (stories only) | questionable | judge_train |

The project’s own code and cards have the licences listed in `NOTICE.md`. Particular upstream datasets, detector weights and the two derived EditLens files carry non-commercial terms. Those terms apply to their respective materials. Keep the corpus private and never redistribute it. The `judge_train` use covers any judge use: checking the existing judge now, and training data if a judge is ever fine-tuned.

Rejected candidates and the reasons are kept in the YAML (`rejected_candidates`). Chat and social have no public source, and none is commissioned for the judge.

To add a source: find a parquet-backed dataset (or its `refs/convert/parquet` branch), pin the commit sha, set the fields, add cleaners if the packaging needs undoing, then run `verify` and a small `corpus` run.

### Running it

```
uv run --extra sourcing humanbench-source corpus --channel review --limit 200
uv run --extra sourcing humanbench-source corpus --source enron --limit 500 --max-scan 200000
uv run humanbench-source manifest                     # rebuild private/corpus/manifest.json
```

- `--limit` is the number of accepted records per source, and `--max-scan` (default 50,000) is the number of rows read per source. Both are **totals across resumes**: rerunning with a bigger limit continues where the last run stopped, and the result is byte-identical to a fresh run with that limit.
- Reads are streamed: Parquet row groups over HTTP range requests at the pinned sha. Nothing is downloaded in full. The read order is a seeded permutation of files, row groups and rows, interleaved across 8 row groups at a time, so a small sample is spread out (IMDb is stored sorted by label, for example).
- A run is resumable. State is in `private/corpus/.state/<source>.json` and is written only after the records it counts are flushed, so a crash loses at most the rows since the last checkpoint. Records written after the last checkpoint are truncated on resume.
- If the registry entry, gate settings, quotas, revision or pipeline version changed, resume is refused. Rerun with `--restart` to rebuild that source.
- Quotas are per (channel, band), per source. Defaults are in `default_quotas`. Bands the benchmark never uses for a channel (from `axes.ALLOWED_BANDS`, so no long reviews) get 0. Override with `--quota short=100 --quota medium=80`, which needs `--restart`.

### Gates, in order

Each rejection is counted under its reason in the state file and the manifest, with up to 5 example row positions.

| Gate | Rejection reasons | Notes |
|---|---|---|
| extract | `source_filter`, `empty_text` | Registry filters (e.g. Enron sent folders), then source cleaners |
| English | `not_english` | Function-word share and Latin-script share. Also drops code- and log-heavy text. Texts under 50 tokens pass here and fail on length |
| date | `date_unknown`, `date_after_cutoff` | Dated strictly before 2022-11-30. A missing or unparseable date is rejected unless the source is pre-cutoff by construction |
| normalize | `empty_after_normalize` | `humanbench.scoring.normalize`, the same normalizer the scorer uses |
| length | `below_min_words`, `above_max_words` | 250 to 4,000 words; band from `axes.BAND_WORDS` |
| dedup | `exact_duplicate`, `near_duplicate` | SHA-256 of canonical words, then MinHash (128 perms, 5-word shingles, 16x8 LSH) with estimated Jaccard >= 0.7, within the source |
| PII | `pii_dense`, `below_min_after_scrub` | URLs, emails, @handles / u/ names and phone numbers become `[URL]`, `[EMAIL]`, `[HANDLE]`, `[PHONE]`. Over 20 replacements per 1,000 words is rejected |
| quota | `band_unused_for_channel`, `quota_full` | |

Dedup compares PII-masked text. Masking is idempotent, so the index can be rebuilt exactly from the scrubbed records on resume.

### Output

`private/corpus/<source>.jsonl`: one record per line, with `id`, `source`, `original_id`, `position`, `repo`, `revision`, `date`, `date_basis` (`field` or `by_construction`), `licence`, `licence_status`, `allowed_uses`, `channel`, `band`, `word_count`, `hash` (SHA-256 of the text), `pii` counts, `english` scores, `meta` and `text`.

`private/corpus/manifest.json`: per-source counts (rows scanned, accepted, rejections by reason, accepted by band, quota-full by band, PII replacements, stop reason, cursor, config hash) and totals by channel and band. It contains no timestamps, so it is deterministic too.

### Caveats

- Names are not scrubbed. Enron in particular is full of real names. Keep the corpus private.
- The placeholders (`[URL]` and so on) could sway the judge. When corpus texts are compared with AI texts, run `humanbench.sourcing.textgates.scrub_pii` over the AI texts too.
- WritingPrompts is detokenized heuristically, so its quotes are all straight. Balance it, or the judge may learn quote style.
- Enron sent folders include pasted third-party text (forwarded articles without headers), and few sent emails reach 250 words.
- Dates are creation dates. Stack Exchange questions and GitHub threads can have been edited later.
- `refs/convert/parquet` branches are regenerated by Hugging Face. If `verify` fails on a pinned sha, re-pin and rerun with `--restart`.

## Lane 2: commissioned writing

### 1. Recruit

```
uv run humanbench-source screener --all               # private/refs/screeners/<persona>.json
uv run humanbench-source screener --persona leeds-plumber --stdout
```

A screener spec is data for a Prolific-style screener. It contains the age window, the English variety or first language, where the writer grew up, their current country, their occupation (free text, checked by a person), and their writing level (rated from a short paid sample). It makes no API calls; map the generic field names to the platform's filters by hand.

- Only inner-circle varieties (US, UK, Irish, Australian, Canadian, NZ) filter on "first language: English". Indian, Nigerian, Singaporean, Kenyan and similar writers are often multilingual, so they filter on where they grew up.
- Prolific is adults only, so every persona is 18 or over (`teen-gamer` is 18, `sneaker-reseller` 19). A persona under 18 added later is flagged with `involves_minor`, and would need guardian consent and ethics approval.
- Writers are never told the writing level they were matched to.

### 2. Brief

```
uv run humanbench-source brief --card path/to/card.yaml --context-root path/to/task/dir
```

This writes `private/refs/briefs/<task>.md`. The brief has the same raw material the model gets (the request, the notes pack, the thread, the persona facts) in writer-facing words, plus channel, length and formatting instructions.

- The brief never mentions AI, detectors, or sounding human; `lint_brief` raises if it would. Writers who know they are being compared with machines perform humanness, and that ruins the reference. Contract terms (original, no AI tools, rights assigned) belong in the contract, not the brief.
- `writing_level` is never shown.
- For agent and ghostwrite tasks, `must_include` and `must_not_include` are hidden by default (`--requirements auto`). There, what the writer chooses to mention *is* the human baseline (over-inclusion, reading trail). QA sends misses on those tasks to review instead of rejecting.

### 3. Collect and QA

The writing editor posts a `Submission` (`humanbench.sourcing.submission`) with the text, the start and submit times, the attestations, and a `KeystrokeLog`. The log records events, not characters: `t_ms`, `kind` (insert, delete, paste, cut, composition, autocorrect, dictation, undo, redo, focus, blur), `n` characters, and a signed `net` for undo and redo. IME, autocorrect and dictation inserts are legitimate and aren't counted as pastes or bursts.

```
uv run humanbench-source qa --card CARD --submission sub.json --model-outputs dir/ [--detector mymod:MyDetector]
uv run humanbench-source queue
uv run humanbench-source review --card CARD --submission-id s7 --reviewer ops1 --decision accept --note "..."
```

The gates run in order:

| Gate | Fails (reject) | Flags (human review) |
|---|---|---|
| contract | an attestation is missing | |
| verifiers (`run_verifiers`) | length, format, must_include, must_not_include, judge_injection | wrapper, hidden_chars, noise; must_(not_)include on agent/ghostwrite tasks |
| time | under 180 s, or over 150 words/min | over 80 words/min |
| keystrokes | 25%+ of the text pasted | 2%+ pasted, any paste of 80+ chars, paste not blocked, 5+ unassisted multi-char inserts, bursts over 20 chars/s, 5%+ keystrokes under 15 ms apart, log not matching the text, no log |
| similarity | 3+ shared 13-word runs or Jaccard >= 0.5 with a model output; Jaccard >= 0.5 with another writer's submission | 1+ shared run or Jaccard >= 0.25; same writer reusing text across tasks |
| detector | **never** | P(AI) >= 0.8 from any plugged-in detector, or a detector error |

N-grams that come from the task material (prompt, notes, thread) are ignored, so quoting the notes isn't copying. A writer's own earlier draft of the same task is ignored too.

The decision is `reject` if any non-detector gate fails, otherwise `needs_human_review` if anything flagged or errored, otherwise `accept`. **A detector can only ever send a piece to review.** `decide` enforces this even if a detector gate reports a failure, because detectors over-flag human writing, non-native writers most of all. A flag on a non-native persona says so in the reasons. The detector is skipped (not called, not billed) once a piece is already rejected.

A detector plugin is any class with `name`, `version` and `score(text) -> float` returning P(AI). Use only official APIs with no-training terms.

### Layout

```
private/refs/
  briefs/<task_id>.md
  screeners/<persona_id>.json
  submissions/<task_id>/<submission_id>.json   as received; write-once (a resubmission needs a new id)
  qa/<task_id>/<submission_id>.json            gates, reasons, metrics, decision
  reviews/<task_id>/<submission_id>.json       a person's decision on a needs_human_review item
  accepted/<task_id>/<submission_id>.json      the reference: normalized text + provenance
```

Accepted test references carry `never_publish: true`. Send them only to judges with no-training or zero-retention terms, or run judges locally.
