# HumanBench v0.1 paper

The paper is `humanbench.tex`; its compiled copy is `humanbench.pdf`. The author is Pragnyan Ramtha (`pragnyanramtha@agent7.dev`).

## Build and length

Built with tectonic 0.15.0 (XeTeX, TeX Live 2022 bundle; `pdflatex` is not needed):

```bash
cd paper
tectonic --keep-intermediates --keep-logs humanbench.tex
```

Any tectonic 0.15.0 binary works; it downloads the packages it needs on the first run. The build has no
errors, no undefined citations or references and no overfull boxes. Underfull-box warnings remain, and
tectonic reports an internal bibliography change-detection warning although the bibliography is unchanged.

Length: **13 pages**. Body: **10 pages** (1–10); references: **1 page** (11); appendices: **2 pages**
(12–13). Explicit page breaks separate the three parts. The body must remain at most ten pages after edits.

The two addresses the paper prints are macros at the top of `humanbench.tex`: `\hbrepo` (the repository)
and `\hbsite` (the website with the leaderboard). The website is `https://humanbench.agent7.dev`; the repository is
`https://github.com/pragnyanramtha/human-bench`.

## Figures

| Figure | What it shows | Made by | Data |
|---|---|---|---|
| 1 Protocol | Task, model, three outputs, the three gates in order, the four outcomes, the score | TikZ in `humanbench.tex` | none (a diagram) |
| 2 Example task | Public synthetic task `work_docs-15`, shortened: what the model sees beside the hidden lists and the length band | TikZ in `humanbench.tex` | `tasks/dev/work_docs-15/card.yaml`; the length band is the 250–400 words of a short task with the 10% tolerance of `verifiers.check_length` |
| 3 AI-text detector | Headline AI-text detector scores of the 801 human texts and of the outputs that passed the first two gates, log axis, threshold marked | `figures/make_figures.py` → `fig-detector.pdf` | C (human scores, threshold); L (outputs with `scripted` 1 and `floor` 1.0) |
| 4 Leaderboard | Score over all tasks with its 95% interval per model; public and held-out scores as small marks | `fig-leaderboard.pdf` | L, C |
| 5 Where outputs end | One stacked bar per model: failed checks, rejected by judge, flagged as AI, passed all three | `fig-outcomes.pdf` | L, C |
| 6 By kind of task | Score with interval on the 52 public human-reference tasks, the 80 public and the 80 held-out synthetic tasks | `fig-kinds.pdf` | L, C |

Regenerate the four data figures after the ledger or the calibration file changes:

```bash
uv run --with matplotlib python paper/figures/make_figures.py
```

The script reads only `reports/ledger-v0.1.jsonl` and `reports/calibration-v0.1.json`, imports `ci` from
`humanbench.harness.report_v01` so that its intervals are the report's, and writes the four PDFs and
`figures/figure_data.json` (every plotted number, with a fingerprint of the two files it read).
`check_numbers.py` compares that file with the tables of the results report and fails if the figures are
stale. Figures are drawn at their printed size (3.42 in for one column, 7.1 in for the full width) with
9 pt text, the size of the captions, and are included unscaled. The typeface is Latin Modern Roman, the
paper's own: `figures/fonts/` holds the three 9 pt fonts (GUST Font License, taken from the TeX Live
bundle); without them the script falls back to matplotlib's Computer Modern.

One palette is used in every figure and is defined twice, in `make_figures.py` and in the preamble of
`humanbench.tex`: ink blue `#274fa3` for passed, amber `#c4710f` for flagged as AI, dark grey `#50555c`
for rejected by judge, light grey `#b3b7bd` for failed checks. Blue and amber pass the dataviz
palette validator as a pair on a white page (lightness band, chroma, colour-blind separation, 3:1
contrast); the greys pass it as an ordinal ramp. In greyscale the four segments of a stacked bar sit at
L* 74, 36, 56 and 35 from the left, so neighbours differ by 20 or more.

Tables 7 and 8 in Appendix D hold the numbers behind Figures 5 and 6. They replace the two body tables
that those figures made redundant (where outputs stopped; the public score by kind of task). The earlier
gate diagram was replaced by Figure 1.

## Verification

```bash
python3 paper/check_numbers.py
```

This read-only script exits nonzero on a discrepancy. It reads source files and prints only aggregate data,
never human text. It performs:

- Ledger recomputation: task counts, sample multiplicity, answer-quality and score point estimates,
  the outcome of every output, pooled outcomes, concentration by channel, conditional unflagged rates and the most common failure.
- Calibration: threshold values and exceedance counts; pool composition by channel and source from
  `private/judge_cache/in/humans.jsonl` (counts only).
- Exact transcription comparisons of all six model-result tables against the generated Markdown reports;
  a separate check covers the detector/reference-count table.
- Figures: `figures/figure_data.json` against the report's tables (score, public, held-out, the three kinds
  of task with their intervals, the four outcomes, the row order), the detector panel's counts and medians
  against the ledger and the calibration file, and the statements the captions make.
- The example task of Figure 2, piece by piece, against `tasks/dev/work_docs-15/card.yaml`.
- The reasoning-effort control quoted in the setup against `reports/effort-control-v0.1.md`, and the
  efforts of Table 2 against the runs listed in the results report.
- Wording: the outcome and gate names are present, and a list of phrases the paper must not use is absent
  (stored as hashes).
- Derived narrative checks: extrema and rankings, the by-kind comparison, held-out versus public synthetic
  comparison, formatting failure count and largest waiver, manifest totals and pinned revisions.
- Validation aggregate counts from the two `private/floor/validate-*.json` files, field `stats`;
  full Appendix B question wording checked against the source-code string constants.
- An exhaustive inventory of numerical prose in `number_sources.json`. These reviewed source contracts
  bind each paragraph to source-file hashes and fail when either changes. **These guards are review
  checks, not independent statistical recomputation.** They complement the computed checks above;
  claims such as historical sourcing rules and CPU pilot measurements are quoted from their recorded
  sources, not re-executed. Model identifiers, typesetting dimensions and section numbering are not
  experimental numerical claims.

`number_sources.json` is the machine-readable companion to the source table below. Do not simply refresh
its hashes after regeneration: review the relevant claims first. Intervals are transcribed from reports;
the script independently recomputes point estimates, not bootstrap intervals.

## Sources and source exceptions

R = `reports/results-v0.1.md`; L = `reports/ledger-v0.1.jsonl`;
C = `reports/calibration-v0.1.json`; M = `tasks/ref/MANIFEST.json` / `.md`;
S = `reports/surface-features-v0.1.md`; W = `reports/workspace-format-v0.1.md`;
J = `reports/judge-cpu-baselines.md`; E = `experiments/judge_cpu/README.md`.

| Paper material | Numerical evidence |
|---|---|
| Abstract, introduction, score definition, conclusion | R; L (212 tasks, 132 public = 52 + 80, 80 held-out; eleven models, three samples; overall 1.9–23.1%; public 2.3–22.0%, held-out 1.2–25.0%; all-task quality 27.0–83.3%); C (801 human texts) |
| Task channels and modes | L and current task YAML files, recomputed in section G of checker: human-reference tasks 16 review, 10 email, 9 personal, 8 forum, 5 blog, 4 work documents; invented 10 per channel and modes 36/13/23/8 |
| Table 1: AI-text detectors | R detector and reference-count tables; C thresholds; E preprocessing |
| Table 2: models, reasoning effort, judges | R Runs (effort per model, checked by the script); `floor.py` company routing |
| Reasoning-effort control (setup, limitations) | `reports/effort-control-v0.1.md`: 66 tasks, two outputs; Fable 17.4% high against 17.4% low, 0.0 [−6.8, 6.8]; Astra 14.4% medium against 9.8% high, −4.5 [−10.6, 0.8]; Sol 13.6% low against 10.6% high, −3.0 [−7.6, 2.3] |
| How outputs were collected | R Runs: hosted clients or a hosted inference service, default effort, tools off, one fixed system line, client framing |
| Table 3 (leaderboard), Tables 7–8 (outcomes, kind breakdown), Figures 4–6 | First three model tables of R: overall, public, held-out and all-task quality; the three failed outcomes over all tasks; scores by kind of task. All rows and intervals checked; overall ordering and point estimates also checked against L; the figures' data checked against the same tables |
| Figure 3 | C: 801 human scores, threshold 0.1641, human median 0.012; L: 4,181 outputs passed the first two gates, 716 of them at or below the threshold (17%), median 0.33 |
| Table 4: the four AI-text detectors used for AI-text detector | First detailed pass-all-gates table of R, all rows checked |
| Table 5: programmatic-check failures and waiver | First failure-kind table of R, columns 0/1/2/3/8; ten models have formatting as largest kind, including the three GPT models other than Luna; Astra's increase is 1.9 points |
| Table 6: surface features | S, all model cells checked; human row 12%, 18.6 words/sentence, 416 words; 156 outputs except MiniMax 153 |
| Public pooled outcomes (secondary part) | L: 1,455 / 343 / 2,129 / 429 of 4,356 (33.4 / 7.9 / 48.9 / 9.8%) |
| Private pooled outcomes (secondary part) | L: 790 / 227 / 1,336 / 287 of 2,640 (29.9 / 8.6 / 50.6 / 10.9%) |
| Real-writing pooled outcomes | L: 677 / 68 / 910 / 61 of 1,716 (39.5 / 4.0 / 53.0 / 3.6%); 18 of 52 tasks have passed outputs |
| Pass concentration | L: email 26/330, blog 19/165, others 16/1,221 on 37 tasks; no work-document pass |
| Conditional unflagged shares | L: Fable 17/88 (19.3%), Sol 13/131 (9.9%) |
| Ledger size and sampling | L: 6,996 rows = 4,356 public + 2,640 private; 156 per model on human-reference tasks, 240 per model per synthetic split |
| Manifest and probe | M: 319 selected, 57 active, 52 panel-usable; R: five flagged; M: 19 pinned, 38 unpinned |
| Reference selection, dates and median | M sourcing section; S median 416 words |
| Judge references and junk | R plus validation `stats`: Sonnet 217/217, Terra 208/217; each rejects 260 junk pieces |
| Shared judge comparison | R: 377 outputs; 86.5 / 79.3%; 288 both pass, 40 both fail, 38 Sonnet only, 11 Terra only; 87.0% agreement |
| CPU detector pilot | J: 140 pairs; zero-shot AUROC .421–.470, vanguard .960, tropa .959, pairwise 98.6%, small Pangram .919; large NF4 pilot only 20 pairs, .995 AUROC; all ordered correctly, also by other AI-text detectors |
| Human threshold | C and R: 801 scores per AI-text detector, eight exceedances, thresholds .1641/.1604/.5508/3.0628 |
| Calibration composition | Read-only aggregation of private human-input metadata, section L of checker; counts reproduced below |
| Device comparison | E: 2,977 texts and 16 of 8,931 changed verdicts |
| Human comparison baseline | R: 47.9% [40.6,55.6] on 4,644 same-source pairs; 49.0% [41.5,56.8] on 4,888 others; other detectors 45.9–50.7% |
| Workspace format comparison | W: 9.8 to 6.8% pooled routing failures; GPT JSON/first/final 3.9/5.0/3.8% |
| Companion real-piece comparison | R detailed comparison: headline all zero; maximum other-detector rate 3.2% [0.0,8.3]; tie half, parity 50% |
| Formatting perturbation limitation | J: vanguard 33/100 verdicts change, Pangram small 20/100 flagged-to-unflagged; large one of ten with quantized weights |
| Protocol constants | Source code, not measured findings: `verifiers.py` (10% tolerance, 13-word copying, 25% task-material copying); `refs.py` (8-word / 8% of 5-grams); `memprobe.py` (40%, 12 words / three 8-word runs / 25% 5-grams); `report_v01.py` (95%, 2,000 resamples) |
| Builder instructions | `scripts/refcards/build_prompt.md`: 60–200 prompt words, 8–25 notes, at most 30 words each |
| Appendix A/B | Fixed instruction and complete floor-v7 questions from released harness/scoring code; the 25-minute example is quoted prompt text, not benchmark data |
| Figure 2 | `tasks/dev/work_docs-15/card.yaml`: request, three of its ten notes, persona line, `must_include` (Saturday or Sat), three `must_cover` points; band 250–400, accepted 225–440 |
| Appendix C | M entry `ref-review-02`: 333 words, 2011-or-earlier date; current task YAML request (1997, around 330 words), notes and coverage; `taskpkg.py` band 250–400, verifier tolerance yields 225–440 |
| Appendix E / release licences | README and NOTICE, not a new experimental measurement |

**Explicit provenance exceptions to report/ledger-only numerical sourcing:** protocol constants and
quoted example-task numbers necessarily come from their code/card sources above. The pre-release edit
counts (27 invented, 20 human-reference tasks), seven drops, reruns and the later-audit instability
(37 of 72 passed two rounds earlier) were explicitly supplied by the task owner. The drops and instability
are also corroborated in README; the edit counts are not independently recoverable from the released
reports/ledger/manifest. They are retained as owner-reported process facts, not presented as recomputed
experimental results. No remaining experimental result lacks a report or read-only computation source.

### Human calibration composition (counts only)

| Channel | Count |
|---|---:|
| blog | 78 |
| email | 44 |
| forum | 128 |
| personal | 78 |
| review | 428 |
| work_docs | 45 |

| Source | Count |
|---|---:|
| aeslc | 28 |
| blog | 14 |
| blog_authorship | 50 |
| enron | 16 |
| github_threads | 45 |
| imdb | 212 |
| medium | 14 |
| reddit | 28 |
| stackexchange | 64 |
| tldr17 | 50 |
| writingprompts | 64 |
| yelp | 216 |

## Changes in the figures revision

- Six figures added (above). Removed from the body: the old gate diagram, the table of where outputs
  stopped and the table of the public score by kind of task; the two tables' numbers are now Tables 7
  and 8 in Appendix D.
- Terminology: the gates are programmatic checks, LLM judge and AI-text detector; the outcomes are failed checks, rejected by judge, flagged as AI and passed all three. That the first gate is scripted and that a
  language model is the judge behind the second is explained where the gates are defined (Section 3.4).
- Tables print display names (Claude Fable 5.2, GPT-6.1 Sol, …). The source keeps the identifiers
  (`\mdl{claude-fable-5-2}`), which is what `check_numbers.py` reads; Table 2 lists both.
- How models were accessed is stated generically: each provider's hosted client or a hosted inference
  service, tools off, one fixed system line, client framing that could not be removed, default sampling
  settings, an output limit for the open-weight models only. No client tool or service is named. The
  limitation that access differed between model families and that results may differ from plain API
  calls is kept.
- Reasoning effort: each model's default effort is in Table 2, in the setup and in the limitations, with
  the statement that the comparison of the leading model with the leading GPT models is confounded with
  effort, and the result of the control in `reports/effort-control-v0.1.md`.
- Captions are set in 9 pt; headings and float spacing were tightened to keep the body at 10 pages.

## Claims revised or removed

- Sol no longer has a higher public point estimate than Astra; Astra is now second publicly.
- Sol no longer has the highest real-writing point estimate: Fable 10.9%, Astra 9.0%, Sol 8.3%, with overlapping intervals.
- Fable no longer stands out *only* on synthetic tasks; it has the highest real-writing point estimate too.
- Flagged as AI is the most common failure over all tasks for nine models; part-specific counts remain eight public and ten held-out.
- The largest formatting-only waiver increase is 1.9 points (Astra), not 1.7. Ten-of-eleven formatting-most-frequent remains true, as in the corrected README.
- The old pooled outcomes, pass concentration, conditional rates, quality ranges, detector comparisons and surface-feature summaries were all replaced by regenerated values.
- Judge agreement/strictness changed to 377 shared outputs and 86.5/79.3% pass rates. Nine, not twelve, current references fail Terra. The claim that this establishes the causal size of a penalty to Anthropic models was softened.
- The first-sample 13.8% versus three-sample 9.2% anecdote was deleted: it has no current model attribution or regenerated support.
- Unsupported detector-before-model release chronology was deleted; release dates are not recorded.
- The claim that pool channel/source composition is unavailable was replaced by metadata aggregation.
- The claim that the large detector's choice has no recorded evidence was replaced by the weak pilot evidence, including the CPU report's recommendation against treating it as an improvement. No full evaluation of the deployed GPU configuration is claimed.
- The stale red draft note was deleted after reconciling the 57 active and 52 usable tasks with the regenerated results.

## Remaining limits and unavailable facts

The headline detector designation remains a convention, not evidence of superiority. The CPU pilot is
small and quantized, and its report advised only diagnostic use. Its full deployed-configuration
matched-pair evaluation is absent. The paper keeps these facts explicitly rather than claiming a
validated detector selection. Model/detector release dates and ethics review are stated as not
recorded; sampling settings were the defaults. No new citations were added: all 31 bibliography entries were previously
verified against arXiv metadata and are unchanged. No human readers were involved; models under test
participated in drafting, auditing and judging. There is no claim of reader indistinguishability.

## Vocabulary verification

See [vocabulary-verification.md](vocabulary-verification.md) for the per-table numeric comparison, retained technical terms and sentence rewrites. `HUMANBENCH_MANIFEST_OUT` can direct manifest regeneration outside `tasks/`; it was set to `/tmp/hb-manifest` for this change. The complete test suite passed: 277 tests, including `tests/test_ledger_tables.py`.
