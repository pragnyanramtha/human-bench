<p align="center"><img src="assets/icon.svg" width="72" height="72" alt=""></p>

<h1 align="center">HumanBench</h1>

<p align="center"><b>Can AI write like a human?</b><br>
A benchmark of everyday writing. An output has to do the task and not be flagged as AI-written.</p>

<p align="center">
<a href="https://human-bench-iota.vercel.app">Leaderboard</a> ·
<a href="https://human-bench-iota.vercel.app/humanbench.pdf">Paper</a> ·
<a href="https://human-bench-iota.vercel.app/#/tasks">Tasks</a> ·
<a href="#quickstart">Quickstart</a>
</p>

HumanBench gives a model an everyday writing task (an email, a forum reply, a review, a post, a work document; 250 to 4,000 words; any tone) and puts each output through three gates. v0.1 is a research preview: 11 models, 212 tasks, 3 samples per task.

## Leaderboard (v0.1)

| # | Model | HumanBench score | Public | Held-out | Answer quality |
|---|---|---|---|---|---|
| 1 | Claude Fable 5.2 | **23.1%** [18.9, 27.8] | 22.0% | 25.0% | 58.5% |
| 2 | GPT-6 Astra | **12.6%** [8.8, 16.5] | 13.6% | 10.8% | 83.3% |
| 3 | GPT-6.1 Terra | **11.5%** [7.7, 15.4] | 11.6% | 11.2% | 70.9% |
| 4 | Claude Opus 5.5 | **11.3%** [8.0, 14.9] | 10.4% | 12.9% | 57.1% |
| 5 | Claude Sonnet 5.5 | **11.0%** [7.9, 14.5] | 10.4% | 12.1% | 51.9% |
| 6 | GPT-6.1 Sol | **11.0%** [7.4, 14.8] | 11.9% | 9.6% | 81.6% |
| 7 | DeepSeek V4.1 Flash 1017 | **9.7%** [6.9, 12.9] | 7.6% | 13.3% | 60.8% |
| 8 | Kimi K3.1 Pro | **7.9%** [5.0, 11.0] | 7.3% | 8.8% | 70.3% |
| 9 | Claude Haiku 5.5 | **7.1%** [4.4, 10.1] | 5.8% | 9.2% | 41.0% |
| 10 | MiniMax M3.1 Flash | **5.5%** [3.5, 7.7] | 5.6% | 5.4% | 27.0% |
| 11 | GPT-6 Luna | **1.9%** [0.6, 3.3] | 2.3% | 1.2% | 54.9% |

Brackets are 95% bootstrap intervals over tasks. Scores are low because the detector flags most outputs. Every table: [reports/results-v0.1.md](reports/results-v0.1.md).

## How an output is scored

1. **Programmatic checks.** Scripts check length, format, required facts, where each message was sent and what it leaked.
2. **LLM judge.** A judge model from another lab answers yes/no questions: coherent, does the task, covers what was asked, fits the tone and the person, reveals nothing it was asked to keep out.
3. **AI-text detector.** An open detector scores the text. It passes at or below the threshold that 99% of 801 human-written texts stay under.

An output counts only if it passes all three. **The HumanBench score is the share of a model's outputs that count**: the mean over a task's three samples, then the mean over all 212 tasks. *Answer quality* is the share that passes gates 1 and 2, whatever the detector says.

The gates are not added up as weighted points. That would give most of the marks to an output that plainly reads as AI, and the number would mostly report instruction following.

**What this is not.** No human readers were involved. "Not flagged" means a named classifier did not flag the text, not that a person would be fooled. A high score does not make undisclosed AI authorship acceptable. Read [Limitations](#limitations) before quoting a number.

## Tasks

| | Public | Held-out |
|---|---|---|
| Tasks | 132: 52 human-reference (`tasks/ref/`), 80 synthetic (`tasks/dev/`) | 80 synthetic, not released |
| Purpose | Anyone can read them, run a model and check the score | Shows whether the score holds on tasks no model can have been trained or tuned on |

- **Human-reference task.** Built backwards from a real piece of writing by a real person (public text from before 2022-11-30). The request and notes were derived from it; the model never sees the piece. The piece itself is not flagged by the detector, so the model is asked to clear a bar a person cleared on the same content.
- **Synthetic task.** An invented situation and persona. Includes thread replies, ghostwriting and workspace tasks (a simulated inbox, forum or chat). Its reference solution is model-written and only proves the checks can be met.

Each task is a card: the request as a user would type it, the user's notes, a persona, and hidden lists the model never sees (`must_include`, `must_cover`, `must_not_reveal`, routing checks). Every model gets one fixed system line: what it writes will be sent as the user's own words, so it should sound natural and human while doing everything asked.

How tasks were made: drafted by a model, checked mechanically (the reference passes the task's own checks; empty and off-task outputs fail), audited by a second model against a fixed rubric until no blocker or major finding remained, and screened so that no tested model could reproduce a human reference from memory. Before release, seven human-reference tasks were dropped because they carried sensitive details about identifiable people or unsuitable content.

## Quickstart

```bash
uv sync --group dev && uv run pytest -q
```

**Recompute the published scores.** Needs only this repository: the scoring ledger (`reports/ledger-v0.1.jsonl`, one row per output, no text) and the detector calibration.

```bash
uv run python scripts/ledger.py tables
```

**Score a model on the public synthetic tasks.** You need an OpenAI-compatible endpoint for the model, one for a judge model, and a GPU for the detector (about 0.1 s per text on one A40).

```bash
# 1. outputs: three samples per task
uv run python -m humanbench.harness.run_model run --run-id my-run --model MODEL \
    --base-url https://api.example.com/v1 --api-key-env MY_API_KEY --split dev --samples 3
# 2. LLM judge, on the outputs that pass the programmatic checks
export HUMANBENCH_JUDGE_BASE_URL=https://api.example.com/v1 HUMANBENCH_JUDGE_KEY_ENV=MY_API_KEY
uv run python -m humanbench.harness.floor_run run --run-id my-run --judges api:JUDGE_MODEL
# 3. detector: export the texts, score them on the GPU machine, merge the scores back
uv run python scripts/gpu_bundle.py export --run-id my-run            # -> private/gpu/texts.jsonl
#    on the GPU machine, with experiments/judge_cpu/*.py and texts.jsonl copied over:
#      python gpu_score.py --in texts.jsonl --out scores.jsonl
#      python judge_run.py --in texts.jsonl --device cuda --cache fast
uv run python scripts/gpu_bundle.py merge --scores scores.jsonl --fast-dir fast
# 4. report
uv run python -m humanbench.harness.report_v01 --run-id my-run --out my-results.md
```

Working files go under `private/`, which git ignores. The report uses the published detector thresholds, which hold only for scores made the same way (pinned model revisions, GPU, bfloat16; see [experiments/judge_cpu/README.md](experiments/judge_cpu/README.md)). This gives a score on the 80 public synthetic tasks, not the full score.

**Human-reference and held-out tasks** need material that is not here. The human references cannot be redistributed: `tasks/ref/MANIFEST.json` gives each one's dataset, record locator and SHA-256 so a retrieved copy can be checked. A new model must also pass the memorisation probe (`scripts/refcards/memprobe.py`). To have a model scored on the held-out tasks, open an issue.

## In this repository

| Path | What |
|---|---|
| `tasks/dev/`, `tasks/ref/` | The 137 public task cards and the manifest of human references |
| `traces/answers-dev-v0.1.jsonl` | Every model output on the public synthetic tasks |
| `reports/` | Results, the scoring ledger, detector calibration, side analyses |
| `src/humanbench/` | Task schema, programmatic checks, judge prompt, runner, report |
| `scripts/` | Ledger, detector bundling, reference-task tools, audit rubric |
| `experiments/judge_cpu/` | Detector code and the evaluation that chose the detectors |
| `paper/` | The paper and the scripts behind its figures and numbers |
| `site/` | The website, as static files |

## Limitations

- **Detector-relative, not reader-validated.** The score depends on one open classifier at one threshold. Detectors differ and can be fooled; three others are reported beside it and give different numbers.
- **Task mix.** The score is a mean over 52 human-reference and 160 synthetic tasks. The held-out part has no human-reference tasks, so the public and held-out scores are not expected to be equal.
- **Old human text against new model text.** The human calibration texts and human references were written before 2022-11-30, mostly 1999 to 2017. A detector may separate them from model text partly by period, source or formatting.
- **The detector is older than the models.** A newer model is more likely to fall outside what the detector learned, so a higher score may reflect the detector's age.
- **Models under test helped make the benchmark.** Synthetic tasks were drafted by Claude Opus 5.5 and revised by GPT-6.1 Sol; human-reference tasks were built and audited by GPT-6.1 Sol; the judges are Claude Sonnet 5.5 and GPT-6.1 Terra. A model family may be helped by material it wrote or hurt by a rival's judge. Neither effect is measured.
- **Reasoning effort differs between models.** Each model ran at its own default: Claude Fable 5.2 and Claude Sonnet 5.5 high; Claude Opus 5.5, Claude Haiku 5.5, GPT-6 Astra and GPT-6 Luna medium; GPT-6.1 Sol and GPT-6.1 Terra low; open-weight models at the provider default. A control on 66 public tasks found no gain from more effort: Claude Fable 5.2 at low scored the same, and GPT-6 Astra and GPT-6.1 Sol at high scored 4.5 and 3.0 points lower, with intervals that include zero ([reports/effort-control-v0.1.md](reports/effort-control-v0.1.md)).
- **How models were accessed.** Outputs were collected through each provider's hosted client or a hosted inference service, with tools off, default sampling and the fixed system line. Client software can add framing text of its own around a request, which could not be removed; an output that repeated any of it was discarded and regenerated. Access differed between model families, and results may differ from other ways of calling the same models.
- **Two judges, not equally strict.** A judge never scores its own company's model: Claude Sonnet 5.5 judges every model except Anthropic's, which GPT-6.1 Terra judges. On 377 outputs both read, they passed 86.5% and 79.3% and agreed on 87.0%. In validation both rejected all 260 junk outputs; Claude Sonnet 5.5 passed all 217 reference texts and GPT-6.1 Terra 208. Anthropic models therefore face a slightly stricter judge.
- **The audit is done by a model and is not stable.** On a second look at tasks still in the audit loop, the auditor raised new findings on 37 of 72 it had passed earlier. A clean audit means no blocker or major finding in one pass, not that a second pass would find nothing.
- **Training exposure is unknown.** Human references come from public corpora. The memorisation screen shows that no tested model reproduced them on request, not that it never saw them.
- **The human never got the task.** A human-reference task has a request reconstructed from the finished piece. The comparison is with real writing on the same content, not with a person answering the same prompt.
- **An unstated formatting rule.** The checks fail markdown that the venue would show as typed (a bold title on a review, a heading in an email). The prompt names the venue but does not say it is plain text. This fails 26.9 to 38.5% of Claude and open-weight outputs on human-reference tasks against 7.7 to 12.8% for GPT models. Waiving it would raise no model's score on those tasks by more than 1.9 points, because such outputs are almost always flagged as well.
- **Workspace tasks are simulated.** The model sees the workspace as text and writes its messages under a short To / Replying to / Subject header. This tests writing and routing, not tool use. The format was revised once and every workspace output regenerated ([reports/workspace-format-v0.1.md](reports/workspace-format-v0.1.md)).
- **Surface differences.** Markdown appears in 34 to 86% of a model's outputs on human-reference tasks against 12% of the human references, and that feature alone separates them with AUROC 0.61 to 0.87 ([reports/surface-features-v0.1.md](reports/surface-features-v0.1.md)). Part of what the detector picks up may be such habits.
- **Small numbers.** Three samples per task and wide intervals. Samples of one task are not independent tasks.
- **A filtered, English-only sample.** The tasks are the ones that survived sourcing, audit and screening. They are not a representative sample of everyday writing, and there are no chat or social human-reference tasks.
- **Dual use.** The same score that measures detectability can be used to tune for evasion. The held-out tasks, the canary line and the non-commercial detector licence are partial safeguards, not a solution.

## Citation and licence

```bibtex
@misc{ramtha2026humanbench,
  title  = {HumanBench: Can AI Write Like a Human?},
  author = {Ramtha, Pragnyan},
  year   = {2026},
  note   = {v0.1, research preview},
  url    = {https://github.com/pragnyanramtha/human-bench}
}
```

Code is Apache-2.0. Tasks, outputs, reports, the paper and the site are CC BY 4.0. Two detector files adapted from Pangram's EditLens are CC BY-NC-SA 4.0, and the detector weights are non-commercial, so the scores are for non-commercial use. Details and third-party terms: [NOTICE.md](NOTICE.md).

Tasks and outputs carry a canary line (see `NOTICE.md`). Please do not train models on them.
