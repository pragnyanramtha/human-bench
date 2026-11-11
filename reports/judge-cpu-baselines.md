# Can CPU-only, ready-made AI-text detectors be HumanBench's judge?

Status: measurement report, 2026-11-09. Code: `experiments/judge_cpu/`. Data: `private/judge_eval/` (gitignored).

## Summary

- **Zero-shot statistics with CPU-sized models are useless here.** Binoculars and Fast-DetectGPT with
  the Qwen2.5-0.5B or 1.5B base/instruct pairs score at or below chance (AUROC 0.41–0.47, pairwise
  0.35–0.44). The implementation was sanity-checked: it flags Qwen's own output.
- **Most off-the-shelf classifiers are weak.** The RAID- and HC3-era RoBERTa and ModernBERT models sit
  between chance and pairwise 0.83. Desklib reaches 0.89.
- **Two 2026 off-the-shelf classifiers are strong.** `vanguard-modernbert-large` and
  `tropa-mini-deberta-large` both reach AUROC 0.96 and pairwise accuracy 0.986 (95% CI 0.96–1.00).
  They hold up on the 40 "write like a real person" hard negatives: vanguard 0.95 and tropa-mini 1.00
  pairwise. Each costs about 1.1–1.3 s per 500 words with about 3.5 GB of RAM.
- **Light training on top of a frozen detector is better still.** A logistic-regression head on
  tropa-mini's frozen features gets AUROC 0.998, TPR@1%FPR 0.96 and pairwise 1.000. The same probe
  trained only on Claude mirrors scores the GPT-6.1 mirrors at AUROC 0.997. Training it takes seconds on
  CPU once the features are cached.
- **Recommendation.** Use vanguard and tropa-mini (pinned revisions) as the baseline and diagnostic
  columns now. For the v1 scoring judge under CPU-only constraints, replace "fine-tune ModernBERT-large
  on a GPU" with **a private logistic-regression head on frozen tropa-mini and vanguard features,
  ensembled with a TF-IDF model, retrained each release on the training pool.** Keep GPU fine-tuning as
  the fallback if this fails the §4.5 validation on commissioned human texts.
- **Caveats.** n = 140 pairs, 2 generators and archival human text, so everything here is indicative,
  not decisive (§6).

## 0. Results at a glance

Fold view, n = 140 pairs (280 texts). Brackets are 95% pair-bootstrap CIs. When pairwise accuracy is
exactly 1.000 the bootstrap CI collapses, so the rule-of-three lower bound (≈0.979 for 140/140) is used
instead. Speed comes from the idle-machine timing pass. "Probe" cost equals its backbone's cost, because
the logistic regression adds microseconds. TF-IDF and stack costs are milliseconds.

| detector | AUROC | TPR@1%FPR | TPR@5%FPR | pairwise acc | wall s / 500 words | CPU s / 500 w | peak RSS |
|---|---|---|---|---|---|---|---|
| **probe: tropa-mini features + LR** (grouped CV) | **0.998** [0.996, 1.000] | **0.964** | **0.993** | **1.000** (≥0.979) | 1.34 | 2.68 | 3.8 GB |
| probe: vanguard features + LR (grouped CV) | 0.992 [0.982, 0.999] | 0.907 | 0.979 | 0.993 [0.979, 1.000] | 1.10 | 2.20 | 3.4 GB |
| stack of all off-the-shelf scores + LR (grouped CV) | 0.978 [0.962, 0.991] | 0.579 | 0.936 | 0.986 [0.964, 1.000] | (sum of all) | | |
| TF-IDF word+char + LR (grouped CV) | 0.970 [0.950, 0.987] | 0.871 | 0.886 | 1.000 (≥0.979) | <0.01 | <0.01 | <1 GB |
| **vanguard-modernbert-large** (off the shelf) | **0.960** [0.943, 0.977] | 0.643 | 0.821 | **0.986** [0.964, 1.000] | 1.10 | 2.20 | 3.4 GB |
| **tropa-mini-deberta-large** (off the shelf) | **0.959** [0.937, 0.978] | 0.386 | 0.750 | **0.986** [0.964, 1.000] | 1.34 | 2.68 | 3.8 GB |
| desklib-deberta-large | 0.783 [0.742, 0.825] | 0.100 | 0.271 | 0.893 [0.836, 0.943] | 1.35 | 2.69 | 3.8 GB |
| rasbt-modernbert-base | 0.782 [0.738, 0.826] | 0.100 | 0.136 | 0.829 [0.764, 0.889] | 0.41 | 0.83 | 1.3 GB |
| fakespot-roberta | 0.716 [0.669, 0.760] | 0.064 | 0.179 | 0.807 [0.743, 0.871] | 0.20 | 0.40 | 1.4 GB |
| superannotate-roberta-large | 0.683 [0.634, 0.734] | 0.007 | 0.093 | 0.775 [0.707, 0.843] | 0.65 | 1.30 | 3.1 GB |
| tabularis-modernbert-base | 0.674 [0.624, 0.728] | 0.043 | 0.129 | 0.714 [0.643, 0.793] | 0.33 | 0.66 | 1.5 GB |
| hc3-chatgpt-roberta (2023) | 0.609 [0.559, 0.659] | 0.014 | 0.064 | 0.607 [0.529, 0.686] | 0.20 | 0.41 | 1.4 GB |
| tmr-roberta-raid | 0.493 [0.447, 0.540] | 0.007 | 0.057 | 0.529 [0.450, 0.607] | 0.21 | 0.42 | 1.4 GB |
| drayson-modernbert-raid-mage | 0.495 [0.446, 0.546] | 0.014 | 0.079 | 0.439 [0.361, 0.525] | 0.45 | 0.89 | 1.5 GB |
| openai-roberta-base (2019 GPT-2 detector) | 0.460 [0.385, 0.533] | 0.000 | 0.007 | 0.454 [0.375, 0.536] | 0.20 | 0.40 | 1.4 GB |
| Binoculars, Qwen2.5-1.5B pair | 0.470 [0.410, 0.532] | 0.014 | 0.086 | 0.436 [0.357, 0.521] | 6.93 | 13.70 | 8.0 GB |
| Fast-DetectGPT (base→instruct), Qwen2.5-1.5B | 0.469 [0.410, 0.531] | 0.007 | 0.093 | 0.436 [0.357, 0.521] | (same pass) | | |
| Binoculars, Qwen2.5-0.5B pair | 0.424 [0.361, 0.482] | 0.036 | 0.086 | 0.393 [0.314, 0.471] | 3.13 | 6.10 | 3.3 GB |
| Fast-DetectGPT (base→instruct), Qwen2.5-0.5B | 0.421 [0.360, 0.481] | 0.043 | 0.086 | 0.386 [0.307, 0.464] | (same pass) | | |
| mean log-likelihood, Qwen2.5-1.5B-Instruct | 0.518 [0.471, 0.565] | 0.007 | 0.043 | 0.507 [0.421, 0.593] | (same pass) | | |

Peak RSS includes a transient load-time copy (see §5), so steady-state memory is roughly the weight size:
1.6–1.7 GB for the large encoders and 6.2 GB for the 1.5B pair.

## 1. Question

DESIGN.md §4.5 plans one shared ModernBERT-large classifier, trained on GPU for each release, as the
judge. The judge's decision is pairwise: for a task, is the model's text rated more human than the
human reference written from the same notes? The question here is whether ready-made detectors that run on
this machine could do that job instead, or at least serve as a baseline or diagnostic column. The machine
has 4 vCPUs (2 physical cores, AMD EPYC 9V45 with AVX-512 bf16), 15 GB RAM and no GPU.

## 2. Eval set: 140 human texts and 140 synthetic mirrors

### Human side

All texts were written before 2022-11-30 and come from public HF datasets that need no account. Each has
250–1,000 words (median 460). There are 28 per channel, and about half of each source is over 450 words.
Selection is seeded and reproducible (`build_human.py`).

| channel | source (HF id) | n | dates | licence / terms |
|---|---|---|---|---|
| email | Yale-LILY/aeslc (Enron) | 28 | 1999–2002 | Enron corpus, FERC public release; HF card "unknown" |
| forum | HuggingFaceH4/stack-exchange-preferences: answers on gardening 4, poker 4, opensource 4, homebrew 2 | 14 | questions 2011–2018 (answer date not in dataset) | CC BY-SA 4.0 |
| forum | webis/tldr-17: changemyview, ELI5, Fitness, AskHistorians, askscience, etc. | 14 | 2006–2016 | CC BY 4.0 (dataset); posts by Reddit users |
| review | Yelp/yelp_review_full | 16 | ≤2015 | Yelp Dataset terms (non-commercial research) |
| review | stanfordnlp/imdb | 12 | ≤2011 | research use (Maas et al. 2011) |
| blog | tasksource/blog_authorship_corpus | 14 | 2004 | non-commercial research (Schler et al. 2006) |
| blog | fabiochiu/medium-articles | 14 | 2019–2020 | HF card MIT; articles are the authors' copyright |
| personal | webis/tldr-17: TwoXChromosomes, AskMen, relationships, offmychest, etc. | 14 | 2006–2016 | as above |
| personal | euclaise/writingprompts (stories) | 14 | 2016–2018 | HF card MIT; stories by r/WritingPrompts users |

Filters: ≥97% ASCII, at most 2 URLs, no code blocks, and no forwarded or quoted email chains. Only one
text per blog author or Q&A question. WritingPrompts had tokenizer artefacts (`` `` ``, `is n't`), which
were repaired.

### AI side: synthetic mirrors (Pangram-style, matching DESIGN.md's notes-pack design)

1. **Briefs**: `claude-sonnet-5-5` read 10 human texts per call (14 calls). For each text it wrote a
   brief containing genre, writer, audience, situation, 6–12 paraphrased key points in the original
   order, tone, writing level and a format note. The brief contains no text from the original, but it
   keeps names and numbers.
2. **Mirrors**: one call per human text. The prompt is a plain user request built from the brief, with
   a target length equal to the human word count (the median AI/human length ratio came out at 1.02).
   - 70 by `claude-sonnet-5-5` through its provider's hosted client, with system prompt "You are a helpful assistant.", no
     tools and no settings sources.
   - 70 by `gpt-6.1-sol` through its provider's hosted client, with a read-only sandbox, `model_instructions_file` replaced by
     the same one-line system prompt, and reasoning effort low (the host's default).
   - Generators alternate within each channel. A third of each generator's mirrors (40 in total, 8 per
     channel) are **instructed hard negatives**. Their prompt adds: "write exactly as the real person
     would… avoid AI tells: no em dashes, no markdown, no neat summary, no 'it's not X, it's Y', no lists
     of three, no stock words… vary sentence length…".
3. **Copy check**: a mirror is dropped if more than 15% of its 8-grams appear in its source. The maximum
   was 11.9% (names and document titles carried through the brief), so none were dropped.

Total model calls: 158 (14 brief calls, 140 mirror calls, 4 CLI smoke tests).

Raw tells, as the share of texts containing each: em/en dash 25/140 AI vs 10/140 human; `**` markdown
23 vs 5; bullet lines 15 vs 8; curly quotes 51 vs 18. Instructed mirrors had 0 dashes, 0 `**` and 0
headings, so the instruction removed the surface tells.

### Text view used for scoring

The primary view, `fold`, is applied identically to both sides. It does NFKC normalization, turns curly
quotes straight and dashes into hyphens, and collapses every whitespace run (paragraph breaks included)
to one space. Markdown symbols are kept. The human sources were scraped by different pipelines: the 2004
blogs have no paragraph breaks, and Yelp and IMDB are ASCII-only. Without folding, a detector can win on
dataset artefacts. The detectors' own preprocessing (for example tabularis's NFKC and whitespace
collapse) is a subset of this. A secondary `light` view, which keeps typography and paragraphs, was run
for TF-IDF, vanguard and tropa-mini to show how big the artefact effect is.

## 3. Detectors

Each model id was checked against the HF API before use, and all of them exist. Scores are oriented so
that higher means more AI. Classifiers with a 512- or 768-token window run over equal-sized consecutive
windows, and their AI logits are averaged with length weights, so nothing is truncated. Everything runs
in bf16 autocast on 2 threads. The box has 2 physical cores: 4 threads was slower, and bf16 was 2.8×
faster than fp32 on RoBERTa-base.

| detector | HF id | released | backbone, window | licence |
|---|---|---|---|---|
| openai-roberta-base | openai-community/roberta-base-openai-detector | 2019 | RoBERTa-base, 512 | MIT |
| hc3-chatgpt-roberta | Hello-SimpleAI/chatgpt-detector-roberta | 2023-01 | RoBERTa-base, 512 | none stated |
| fakespot-roberta | fakespot-ai/roberta-base-ai-text-detection-v1 | 2025-02 | RoBERTa-base, 512 | Apache-2.0 |
| tmr-roberta-raid | Oxidane/tmr-ai-text-detector | 2026-01 | RoBERTa-base, 512; RAID + hard negatives | MIT |
| superannotate-roberta-large | SuperAnnotate/ai-detector | 2024-09 | RoBERTa-large, 512 (head reimplemented from their v1.1.0 source) | SAIPL (AGPL-style copyleft) |
| desklib-deberta-large | desklib/ai-text-detector-v1.01 | 2025-02 | DeBERTa-v3-large, 768 | MIT |
| tropa-mini-deberta-large | wasitaigeneratedcom/tropa-mini-ai-text-detector | 2026-08 | DeBERTa-v3-large, 768; 4-class head; score = log((1−P(human))/P(human)) | Apache-2.0 |
| tabularis-modernbert-base | tabularisai/ai-text-detection | 2026-05 | ModernBERT-base, 512; RAID + 6 datasets | MIT |
| drayson-modernbert-raid-mage | GeorgeDrayson/modernbert-ai-detection-raid-mage | 2025-05 | ModernBERT-base, full text | Apache-2.0 |
| rasbt-modernbert-base | rasbt/ai-text-detector-modernbert | 2026-08 | ModernBERT-base, full text | Apache-2.0 |
| vanguard-modernbert-large | ShantanuT01/vanguard-ai-text-detector | 2026-05 | ModernBERT-large, full text; DACTYL 2.0 + LLMTrace + MAGA-Bench; part of a PAN-CLEF 2026 2nd-place system | MIT |
| Binoculars / Fast-DetectGPT | Qwen/Qwen2.5-{1.5B,0.5B} (observer/reference) + -Instruct (performer/scorer) | 2024-09 | 1,536-token cap | Apache-2.0 |
| TF-IDF + logistic regression | trained here (word 1–2-grams + char 2–5-grams, C=10) | — | grouped 5-fold CV × 5 seeds | — |
| stack (off-the-shelf) | logistic regression on all off-the-shelf scores | — | grouped CV | — |
| probe-\<detector\> | frozen detector backbone, mean-pooled, + logistic regression | — | grouped CV | — |

"Grouped" CV means a human text and its own mirror always fall in the same fold. Without that, a
trained model can partly memorise topics. Zero-shot scores keep the polarity their papers specify
(lower Binoculars score means AI; higher Fast-DetectGPT discrepancy means AI) and are never flipped
after the fact.

### Metrics

- AUROC.
- TPR at 1% and 5% FPR. The threshold is set on these 140 human texts, so 1% FPR means one human text
  may score above it. That makes TPR@1% very noisy at this n.
- **Pairwise accuracy**: P(score(mirror) > score(its own human source)), with ties counted as 0.5. This
  is the HumanBench judge decision.
- Breakdowns by channel, generator, mode and source. Each uses within-group pairwise accuracy plus the
  detection rate at the global 5%-FPR threshold.
- 95% CIs from a 2,000-sample pair-level bootstrap.
- CPU cost from a separate idle-machine pass (`timing.py`): 16 texts of 400–700 words, one warm-up,
  wall and process-CPU seconds normalised to 500 words, and peak RSS of the process.

## 4. Breakdowns

Each cell is within-group pairwise accuracy / detection rate at the global 5%-FPR threshold. n is 28
per channel, 70 per generator, and 100 default vs 40 instructed. All breakdowns are noisy: one pair is
3.6 points of a channel cell.

| detector | blog | email | forum | personal | review | Claude Sonnet 5.5 | GPT-6.1 Sol | default | instructed |
|---|---|---|---|---|---|---|---|---|---|
| probe: tropa-mini + LR | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.96 | 1.00 / 1.00 | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.97 |
| probe: vanguard + LR | 1.00 / 1.00 | 0.96 / 0.93 | 1.00 / 1.00 | 1.00 / 0.96 | 1.00 / 1.00 | 0.99 / 0.96 | 1.00 / 1.00 | 1.00 / 0.99 | 0.97 / 0.95 |
| TF-IDF + LR | 1.00 / 0.96 | 1.00 / 0.64 | 1.00 / 0.96 | 1.00 / 1.00 | 1.00 / 0.86 | 1.00 / 0.91 | 1.00 / 0.86 | 1.00 / 0.88 | 1.00 / 0.90 |
| vanguard (off the shelf) | 1.00 / 0.89 | 0.96 / 0.64 | 1.00 / 0.93 | 0.96 / 0.82 | 1.00 / 0.82 | 0.97 / 0.83 | 1.00 / 0.81 | 1.00 / 0.93 | 0.95 / 0.55 |
| tropa-mini (off the shelf) | 0.93 / 0.89 | 1.00 / 0.75 | 1.00 / 0.71 | 1.00 / 0.71 | 1.00 / 0.68 | 1.00 / 0.77 | 0.97 / 0.73 | 0.98 / 0.81 | 1.00 / 0.60 |
| desklib | 0.89 / 0.39 | 0.89 / 0.43 | 0.89 / 0.18 | 0.86 / 0.14 | 0.93 / 0.21 | 0.91 / 0.24 | 0.87 / 0.30 | 0.94 / 0.34 | 0.78 / 0.10 |
| Binoculars 1.5B | 0.39 / 0.11 | 0.71 / 0.11 | 0.39 / 0.07 | 0.18 / 0.00 | 0.50 / 0.14 | 0.63 / 0.14 | **0.24** / 0.03 | 0.46 / 0.09 | 0.38 / 0.07 |

What stands out:

- **The instruction to "write like a real person" works against threshold-based detection but not
  against pairwise comparison.** For vanguard, detection at 5% FPR falls from 0.93 (default) to 0.55
  (instructed), but pairwise accuracy only falls from 1.00 to 0.95. The instructed mirrors removed every
  em dash, `**` and heading (§2), so the strong detectors are not relying on those surface tells.
- **Binoculars is inverted for GPT-6.1.** Its pairwise accuracy is 0.24 on GPT-6.1 mirrors: to a
  1.5B scorer, that text looks *more* human than the real human text. This is the "polarity inversion"
  reported in the 2026 detection literature (research notes, KQ1). A pinned zero-shot detector would
  rank generators by their perplexity profile, not by human-likeness.
- **False positives cluster in particular sources.** At the global 5%-FPR threshold, the human texts
  that get flagged are mostly Medium posts (polished 2019–20 web prose; vanguard 14%), WritingPrompts
  stories (vanguard 21%, probe 21%, TF-IDF 29%) and, for tropa-mini, the 2004 blogs (29%). Since the
  threshold is global, these sources can only be lower-bound concerns, but they are the right place to
  look when choosing commissioned human pieces.

### Does a trained head transfer? (train on everything except one group, test on that group)

| held-out group | TF-IDF AUROC / pairwise | probe vanguard AUROC / pairwise | probe tropa-mini AUROC / pairwise |
|---|---|---|---|
| generator = Claude Sonnet 5.5 (trained on GPT-6.1 only) | 0.943 / 1.000 | 0.983 / 0.971 | 0.998 / 1.000 |
| generator = GPT-6.1 Sol (trained on Claude only) | 0.820 / 0.957 | 0.995 / 1.000 | 0.997 / 1.000 |
| mode = instructed (trained on default prompts only) | 0.956 / 1.000 | 0.981 / 0.975 | 0.995 / 1.000 |
| channel = email | 0.931 / 1.000 | 0.966 / 0.964 | 1.000 / 1.000 |
| channel = review | 0.963 / 1.000 | 0.999 / 1.000 | 1.000 / 1.000 |
| channel = blog / forum / personal | 0.986–0.995 / 1.000 | 0.991–1.000 / 1.000 | 0.989–1.000 / 1.000 |

Frozen-feature probes transfer across generator, prompt style and channel far better than TF-IDF does
(TF-IDF drops to AUROC 0.82 on an unseen generator). With only two generator families, this is a weak
version of DESIGN §4.5 check 4 (leave-one-family-out), not a substitute for it.

What TF-IDF keys on (from the coefficients when fitted on everything) is interpretable, and it matches
known tells:

- AI side: "honestly", "nobody", "rather than", "still", "something", "I'd", "matters", `**`.
- Human side: "...", parentheses, "etc", "however", "of the", "will", "regarding".

Some of the human-side vocabulary (Enron's "will be", "regarding", "issues") is era and register
vocabulary, which commissioned 2026 human text won't share.

### Artefact sensitivity (fold vs light view)

Keeping typography and paragraph breaks barely changes the picture:

- TF-IDF: AUROC 0.970 fold vs 0.971 light.
- vanguard: pairwise 0.986 fold vs 0.971 light, with TPR@1% 0.64 vs 0.76.
- tropa-mini: pairwise 0.986 fold vs 0.979 light.

The strong detectors are not winning on dataset formatting. HumanBench should still normalize both
sides the same way (DESIGN §4.1), because the real references come from a different editor than any
model's output.

## 5. CPU cost and the fine-tuning alternative

**Scoring is cheap.** The test split is 600 outputs per model at roughly 800 words on average under
DESIGN §2.5's length bands. With tropa-mini's backbone that is about 2.1 s per text on this box, so
about 21 minutes per model. A 25-model leaderboard takes about 9 hours per backbone, which is fine for
an overnight batch.

With a frozen backbone, **features are computed once and cached.** Adding a model to the leaderboard
then means two steps:

1. Embed its training-pool and test outputs.
2. Refit the logistic regression, which takes seconds.

Nothing else is recomputed. This is the CPU-only version of DESIGN's "add a model → retrain the shared
judge → rescore everything". Rescoring is just applying the new head to cached features.

**Full fine-tuning on CPU is impractical.** Measured with `ft_timing.py` (bf16 autocast, AdamW, 2
threads):

- **ModernBERT-base**: 3.3 s per 4×512-token step, or 0.82 s per sequence, at 4.4 GB RSS. That puts
  3,000 training texts × 3 epochs at **about 3 hours** per run, before any of the three seeds §4.5
  requires.
- **ModernBERT-large**: two attempts (batch 4 and batch 2) went past 8 GB RSS and thrashed the page
  cache on this shared machine, without completing a timed step in 13 minutes. Scaling the base
  measurement by parameter count (≈2.65×) gives **about 8 hours** per run if memory allowed. That is
  roughly 3 seeds × 8 h per release, versus "about an hour on one GPU".

**Engineering notes** (the scripts already handle these):

- transformers 5 keeps safetensors weights memory-mapped. On this box, under page-cache pressure from
  other agents' processes, the 1.5B Binoculars run re-read 68 GB from disk at about 80% iowait.
  `detectors.materialize()` copies the weights into RAM once at load time.
- Binoculars statistics are computed in 256-position chunks, because each full `[tokens × 152k]` float32
  tensor is about 0.8 GB.
- Use 2 threads (one per physical core).

## 6. Caveats

- **Small n.** There are 140 pairs. TPR@1%FPR is set by a single human text and swings widely (vanguard
  0.64 fold vs 0.76 light). Pairwise CIs are about ±2–4 points for the strong detectors and ±7 for the
  weak ones. Channel and generator cells (n = 28–70) are indicative only.
- **The human side is archival and public, not commissioned.** All human texts are pre-2022 public
  datasets.
  - Some sources overlap detector training data at the dataset level. vanguard's DACTYL includes
    WritingPrompts and reviews, and MAGA uses Reddit and Amazon reviews. The strong detectors do just
    as well on sources no listed training set uses (Enron, the 2004 blogs, StackExchange), and
    WritingPrompts is in fact vanguard's weakest source, so contamination does not explain the result.
    It can't be ruled out item by item, though.
  - HumanBench's real references are fresh 2026 text by persona-matched writers working from a notes
    pack. Those writers may have absorbed LLM style, and the Medium false positives hint at what
    polished human prose does to these detectors.
  - Era vocabulary (Enron "regarding", 2004 blogs) makes archival human text easier to tell apart than
    commissioned text will be.
  - **Expect real-reference FPR to be higher than measured here.**
- **Two generators, one recipe.** The mirrors come from Sonnet 5.5 and GPT-6.1 Sol (reasoning effort
  low) through their providers' hosted clients, with a minimal system prompt. The clients still add environment
  context and tool framing of their own. Not tested: Gemini, open-weight models, base models,
  fine-tuned "humanlike" models, humanizer tools, or the multi-sample-per-task setting. There is a
  single fixed "sound human" instruction, and stronger elicitation (style imitation, ghostwriting from
  past posts) will be harder.
- **The mirrors track their sources closely.** The brief keeps the points in the original order and the
  target length is the human length. This is closer than Pangram's topic-only mirrors and close to
  HumanBench's notes-pack design, which suits a pairwise test. The briefs were written by Sonnet, which
  could put Claude-ish wording into prompts for both generators.
- **Public weights can be gamed.** vanguard and tropa-mini are downloadable, so anyone can optimize
  against them. As a diagnostic column that is acceptable. As the scoring judge it is not, unless the
  head is private and retrained on fresh hard examples. A private head on public features is weaker
  secrecy than fully private fine-tuned weights, because features can be attacked. That is one more
  reason to keep the GPU fine-tune as the upgrade path.
- **Licences.**
  - Yelp, IMDB and the Blog Authorship Corpus are research-only. Medium articles remain their authors'
    copyright. The eval set must stay private (it lives in gitignored `private/`).
  - SuperAnnotate's SAIPL licence is copyleft (source must be offered to network users). Avoid it in a
    hosted service; it's not recommended here anyway.
  - tropa-mini is a vendor's open model (Apache-2.0), and its card makes vendor benchmark claims.
    Validate it on our own data, as was done here, and pin the revision (`10fecd08…`; vanguard
    `823061be…`).
- **Precision.** Everything ran in bf16 autocast. On RoBERTa-base, logits moved by up to about 0.08
  against fp32. Score ranks were not re-verified in fp32 for every detector.
- **Timing environment.** Other agents' pipelines were running on the same box. The timing pass ran
  when it was otherwise quiet, but numbers may be ±20%.

## 7. Recommendation

**Baseline and diagnostic column (§4.7): use the off-the-shelf models now.**

- Report vanguard-modernbert-large and tropa-mini-deberta-large, with pinned revisions, logged versions
  and dates. Each costs about 1–1.3 s per 500 words on CPU, AUROC 0.96, pairwise 0.986.
- Drop Binoculars and Fast-DetectGPT with CPU-sized scorers from the plan. They are at or below chance,
  and they invert by generator, so they would mis-rank models. If a zero-shot column is wanted for
  continuity with the literature, it needs a 7B-class pair (Falcon-7B or a Qwen 7B pair). That needs
  about 28 GB RAM in bf16 for two models, which is a GPU-or-bigger-box item, and the 2025–26
  literature suggests it will still be weak on frontier text.
- Drop the 2019–2023 RoBERTa detectors, which sit at chance.

**Scoring judge (§4.5) under CPU-only constraints: a ready-made detector is not enough on its own, but
it gets close, and light private training closes the gap.**

v1 judge: private logistic-regression heads on frozen tropa-mini and vanguard features, plus a TF-IDF
(word + char) logistic regression. Average their standardized scores and retrain each release on
DESIGN's training pool:

- the ~300 commissioned pieces plus archival human text;
- mirrors from every leaderboard model: default, persona and "write like a human" prompts, plus
  humanizer outputs;
- hard negatives mined from flagged human texts.

On this set the components score AUROC 0.97–1.00, pairwise ≥0.99, and they transfer across generator,
prompt style and channel. Training takes seconds once features are cached, and the judge stays
deterministic between retrains.

It must still pass all six §4.5 validation checks before scores are published. Watch two especially:

- **Check 2: at most 1% of held-out *commissioned* human pieces flagged.** This is the most likely
  failure, given the Medium and WritingPrompts false positives above.
- **Check 4: leave-one-family-out with at least 4 families.**

**What a GPU would change:**

- It makes full fine-tuning of ModernBERT-large practical (about 1 h per run against about 8 h FLOP-scaled
  here, which in practice didn't fit in memory). That allows the three-seed check and quick
  hard-negative mining rounds.
- It would let the judge's weights, not just its head, be private.

Use a GPU if the frozen-feature judge fails check 2 or check 4, or once the leaderboard has enough
models that a public backbone becomes an obvious optimization target.

The GPU would not change scoring cost, which is already cheap on CPU. It would not rescue zero-shot
detection, which is an approach problem, not a compute one.

**Benchmark-design implication:** the strongest CPU judges here catch both frontier generators in 98.6–100%
of matched pairs, even when the model is told to sound human. The headline HumanScore (DESIGN §4.6)
would be about 0–1.4% for both, which leaves room at the top but cannot separate models near zero. Keep
a continuous companion statistic next to the headline, such as the mean judge margin between model text
and human text, or per-model AUROC (already listed in §4.7), so that differences between near-zero
models still show.

## 8. Files

- `experiments/judge_cpu/`: its own uv project (CPU torch 2.15, transformers 5.22).
  - `build_human.py`, `generate.py`: the eval set.
  - `detectors.py`, `score.py`, `run_all.sh`: scoring.
  - `probe.py`, `probe_transfer.py`: frozen-feature heads.
  - `timing.py`, `run_timing.sh`, `ft_timing.py`: CPU cost.
  - `analyze.py`: metrics and tables.
  - `fetch_models.py`, `post.sh`, `README.md`.
- `private/judge_eval/` (gitignored, about 620 MB):
  - `human.jsonl`, `briefs.jsonl`, `generations/`, `pairs.jsonl`: the 140 pairs with provenance.
  - `scores/`: per-detector scores.
  - `emb/`: cached probe features.
  - `results/summary.json`, `tables.md`, `timing/`, `probe_transfer.json`, `ft_timing.json`.
  - `raw/`: downloaded sources.
  - `logs/`.
- HF cache: about 17 GB of weights in `~/.cache/huggingface`. Total new disk use is about 19 GB,
  including the 1.4 GB venv.

## 9. EditLens CPU follow-up — 2026-11-10

**Result: keep the current CPU judge.** EditLens RoBERTa is weaker on this evaluation and sensitive to
small presentation changes. Llama NF4 is promising on a small fixed subset, but shows no improvement
over the existing probe there and costs substantially more CPU time. It can serve as an optional
non-commercial research diagnostic; neither EditLens model should become a v1 scoring component on
this evidence.

This follow-up evaluates Pangram's open EditLens checkpoints on the same private texts. It is
non-commercial research. The [paper](https://arxiv.org/abs/2510.03154) frames the task as estimating
AI editing extent, rather than calibrated binary authorship probability. The
[release post](https://www.pangram.com/blog/introducing-open-pangram) describes the two open models;
these are separate from Pangram's production detector.

Inference follows [Pangram's source at 05a588f](https://github.com/pangramlabs/EditLens/tree/05a588f15d792330ccaf91be8ee4fdb54ce26835):
plain text, no instruction or chat template; four bucket logits converted to
`sum(softmax(logits)[i] * i) / 3`, with larger values meaning more AI intervention.
HumanBench's `common.fold` is applied first, followed by the official preprocessing (emoji names,
think-tag/header handling, lowercasing, whitespace normalization). As fold has already flattened
paragraphs, the official multi-paragraph header-removal branch normally does not apply. RoBERTa
uses the first 512 tokens including special tokens; Llama uses the first 1,024. This deliberately
retains the intended truncation instead of substituting the existing detectors' full-document
window averaging. Results therefore compare the intended EditLens configuration against the
existing operational configurations. The Llama adapter requires the trained LayerNorm + linear
`NormedLinear` score head, not a newly initialized ordinary classification layer.

Pinned downloads:

| artifact | repository | revision |
|---|---|---|
| RoBERTa full model | `ml-resources/editlens_roberta-large` | `637532be5fd2965eb00ad16273f52588e6d1f216` |
| Llama adapter | `ml-resources/editlens_Llama-3.2-3B` | `cf23fd0738fc97f2420a8132e0866fa0ca840c3d` |
| Llama base, non-Instruct | `unsloth/Llama-3.2-3B` | `d4446454d87d51aa42e1fb174f25acc5f8762331` |

The adapter's `base_model_name_or_path` explicitly names `meta-llama/Llama-3.2-3B`. Both Unsloth
weight-shard SHA-256 hashes match Meta's official repository metadata at
`13afe5124825b4f3751f836b40dafda64c1ed062`: `584d8d3e…856e002` and `4719a045…5a9cefb`.
RoBERTa's weight SHA-256 `869f33df…dfc8cfb` also matches the official Pangram repository metadata
at `f93e1ace74528cfb48f337ab2fe946fb71a728cb`.

Both [mirror model cards](https://huggingface.co/ml-resources/editlens_roberta-large) retain
Pangram's [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license: attribution,
non-commercial use, notice of changes, and the same license when distributing adaptations.
The [Llama adapter card](https://huggingface.co/ml-resources/editlens_Llama-3.2-3B) has the same
terms; its base additionally carries Meta's Llama 3.2 Community License and acceptable-use terms.
The ungated mirrors do not remove those terms.

New implementation: `experiments/judge_cpu/editlens.py`. Raw scores, hashes, timing, probabilities,
truncation counts and manifests are append-only under `private/judge_eval/editlens/`. The run uses
at most two Torch threads (one under contention) and one model at a time. All tracked results below are aggregate numbers.

## 2026-11-10: CPU format audit (measurements in progress)

The 2026-11-09 claim that the strong detectors are not winning on formatting is too broad for the
new dev-task distribution. On a seeded sample of 100 scorable dev outputs, **vanguard verdicts flip
33 times after stripping Markdown, 10 after lowercasing sentence/line initials, and 4 after adding two
word-initial transpositions**. The prose view eliminates the Markdown flips but still flips 5/100
under line-break collapse. These are verdict flips at a separately calibrated cutoff, not just score
changes. Vanguard is gameable under these edits and is not a robust standalone v1 judge.

Using five grouped outer folds (three training, one calibration, one test; seed 20261110), vanguard's
original/prose AUROC is 0.9604/0.9571, descriptive TPR at eval-human 1% FPR is 64.3%/57.1%, and archival
pilot pairwise accuracy is 99.0%/98.8% (1,680 outputs). With the preselected fold-0 calibration cutoff,
311/953 dev outputs read as human in the original view versus 547/953 in the prose view. Seven of the
960 dev records have empty text or infrastructure failures and are excluded from judge denominators.
Per-channel thresholds raise held-out human FPR from 1.0% to 3.5%; small channel samples do not support
using them as a v1 repair.

The original probe calibration was reconstructed exactly from cached embeddings (cutoff
−0.5682089327560975). Its eight human flags come from GitHub 2, StackExchange 2, WritingPrompts 2,
Enron/AESLC 1 and IMDb 1; the ninth-highest score setting the cutoff is a Yelp review. Five of its
12 human verdicts on dev outputs are Markdown-heavy (density ≥1 marker/100 words), versus 293/953
outputs overall; six of the 12 contain no detected Markdown at all. The Markdown artefact is real,
but cannot explain every miss, and GitHub is not the sole source driving calibration.

The full tropa/probe held-out and edit-robustness comparison is still running with two Torch threads.
No repaired-probe recommendation is validated yet. Code: `common.prose_view` (leaves `fold` unchanged),
`score_trials.py --view prose-v1`, `audit.py`, and `audit_report.py`. Old scores and thresholds are
preserved. New content-addressed caches and preliminary aggregates live only under
`private/judge_eval/audit-prose-v1/`; distinct trial output names use `__audit-<variant>`. Model downloads
are now pinned to the exact vanguard/tropa revisions used by the original cached baseline; previously
`fetch()` followed `main` despite this report describing them as pinned.

The adapter weight SHA-256 `17162383…f2d474` also matches the official Pangram repository
metadata at `b5f8044f631f5b455eafbcb569dcf175f2b0726d`. Tokenization checks confirmed identical
adapter/base Llama token IDs, BOS insertion, and 1,024-token truncation. A regression check exercises
the actual RoBERTa scoring method and verifies that native truncation preserves the final EOS token
and that a uniform four-bucket distribution maps to 0.5. Superseded incorrectly truncated scores
are isolated outside the report loader; all results below use the corrected encoding.

**3B CPU fallback.** The unquantized bf16 six-text pass incurred about 64 GiB of disk reads
through mmap/page-cache reclamation (94 process CPU seconds including loading; 352 seconds total
wall time). NF4 is the official 4-bit quantization type and reduced the paging penalty, but warm
500-word forwards still took roughly 12–19 wall seconds on one thread. The larger pilot uses at
most two threads. These costs justify the fixed subset below rather than a many-hour full 3B run.
The bf16 six-text pass is a load/latency experiment, not a complete accuracy evaluation.

The fixed random seed is `20261110`. The Llama NF4 pilot contains 215 scored records: 20 matched
eval pairs (40 texts), 50 corpus humans, 10 archival sample-1 outputs per generator (40), 49 dev
originals, 30 perturbations of 10 sampled dev originals, and six sanity examples. The 20 eval pairs
contain 17 default and three instructed mirrors and no email pairs. All selected IDs are private.
Baselines are also evaluated on those exact 20 pairs. Llama's calibration pool has only 70 humans;
it does **not** measure the requested 801-human operating point. Any pilot human-reading rates
below are explicitly marked as exploratory, and the full operating point is N/A.

Numerical configuration: CPU Torch `2.15.0+cpu`, transformers `5.22.2`, PEFT `0.21.2`, bitsandbytes
`0.50.2`, bf16 autocast; the reconstructed Llama NormedLinear parameters use bf16. Full fp32 rank
agreement was not tested. Baseline full-eval CIs come from the existing summary; same-subset metrics
and CIs are recomputed from its cached fold scores. Existing dev scores and thresholds are reused
without retraining the probe. All CIs below use the existing 2,000 paired-bootstrap procedure.

### Judge eval (fold + official EditLens preprocessing; 95% paired bootstrap CIs)

| judge | pairs | AUROC | TPR@1% FPR | TPR@5% FPR | pairwise |
|---|---|---|---|---|---|
| vanguard-modernbert-large__fold | 140 | 0.960 [0.943, 0.977] | 0.643 [0.564, 0.800] | 0.821 [0.721, 0.893] | 0.986 [0.964, 1.000] |
| tropa-mini-deberta-large__fold | 140 | 0.959 [0.937, 0.978] | 0.386 [0.236, 0.700] | 0.750 [0.600, 0.929] | 0.986 [0.964, 1.000] |
| probe-tropa-mini-deberta-large__fold | 140 | 0.998 [0.996, 1.000] | 0.964 [0.929, 1.000] | 0.993 [0.964, 1.000] | 1.000 [1.000, 1.000] |
| llama-nf4 | 20 | 0.995 [0.977, 1.000] | 0.950 [0.850, 1.000] | 0.950 [0.850, 1.000] | 1.000 [1.000, 1.000] |
| vanguard-modernbert-large / same Llama subset | 20 | 0.977 [0.940, 1.000] | 0.900 [0.750, 1.000] | 0.900 [0.750, 1.000] | 1.000 [1.000, 1.000] |
| tropa-mini-deberta-large / same Llama subset | 20 | 0.982 [0.948, 1.000] | 0.750 [0.600, 1.000] | 0.950 [0.650, 1.000] | 1.000 [1.000, 1.000] |
| probe-tropa-mini-deberta-large / same Llama subset | 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| editlens-roberta / same Llama subset | 20 | 0.957 [0.905, 0.995] | 0.750 [0.550, 0.950] | 0.800 [0.600, 1.000] | 1.000 [1.000, 1.000] |
| llama-nf4/default | 17 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| llama-nf4/instructed | 3 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| roberta | 140 | 0.919 [0.888, 0.945] | 0.529 [0.429, 0.714] | 0.736 [0.571, 0.807] | 0.957 [0.921, 0.986] |
| roberta/default | 100 | 0.946 [0.918, 0.970] | 0.660 [0.530, 0.860] | 0.840 [0.680, 0.910] | 0.990 [0.970, 1.000] |
| roberta/instructed | 40 | 0.856 [0.769, 0.927] | 0.225 [0.100, 0.600] | 0.475 [0.175, 0.700] | 0.875 [0.750, 0.975] |

Small-subset bootstrap intervals can be degenerate: even 20/20 pairwise successes have a Wilson 95% interval of [0.839, 1.000]. A perfect bootstrap interval is not evidence of zero population error. With 20 eval humans, the empirical 1%-FPR threshold is their maximum score (zero observed false positives), not a validated 1% population operating point.

The 40 instructed mirrors are already inside the 140 pairs; they are not 40 additional human texts.

### Corpus human false positives by channel

| judge | channel | n | flags at eval-human 1% threshold | flags at pooled-801 1% threshold |
|---|---|---|---|---|
| llama-nf4 | blog | 1 | 0 (0.0%) | N/A (subset calibration) |
| llama-nf4 | email | 1 | 0 (0.0%) | N/A (subset calibration) |
| llama-nf4 | forum | 7 | 1 (14.3%) | N/A (subset calibration) |
| llama-nf4 | personal | 3 | 0 (0.0%) | N/A (subset calibration) |
| llama-nf4 | review | 33 | 0 (0.0%) | N/A (subset calibration) |
| llama-nf4 | work_docs | 5 | 0 (0.0%) | N/A (subset calibration) |
| roberta | blog | 50 | 2 (4.0%) | 0 (0.0%) |
| roberta | email | 16 | 0 (0.0%) | 0 (0.0%) |
| roberta | forum | 100 | 5 (5.0%) | 2 (2.0%) |
| roberta | personal | 50 | 0 (0.0%) | 0 (0.0%) |
| roberta | review | 400 | 20 (5.0%) | 6 (1.5%) |
| roberta | work_docs | 45 | 1 (2.2%) | 0 (0.0%) |
| vanguard-modernbert-large | blog | 50 | 1 (2.0%) | 1 (2.0%) |
| vanguard-modernbert-large | email | 16 | 0 (0.0%) | 0 (0.0%) |
| vanguard-modernbert-large | forum | 100 | 2 (2.0%) | 2 (2.0%) |
| vanguard-modernbert-large | personal | 50 | 1 (2.0%) | 2 (4.0%) |
| vanguard-modernbert-large | review | 400 | 1 (0.2%) | 1 (0.2%) |
| vanguard-modernbert-large | work_docs | 45 | 0 (0.0%) | 0 (0.0%) |
| tropa-mini-deberta-large | blog | 50 | 2 (4.0%) | 4 (8.0%) |
| tropa-mini-deberta-large | email | 16 | 0 (0.0%) | 0 (0.0%) |
| tropa-mini-deberta-large | forum | 100 | 0 (0.0%) | 0 (0.0%) |
| tropa-mini-deberta-large | personal | 50 | 0 (0.0%) | 0 (0.0%) |
| tropa-mini-deberta-large | review | 400 | 0 (0.0%) | 0 (0.0%) |
| tropa-mini-deberta-large | work_docs | 45 | 0 (0.0%) | 0 (0.0%) |

Pooled thresholds: llama-nf4 = 0.106955439 (n=70), roberta = 0.158062413 (n=801). Strict score > threshold flags AI; ties read as human. The pooled FPR is calibration performance, not held-out validation. Any Llama subset threshold is exploratory and does not measure the requested pooled-801 operating point; N/A marks this distinction.

### Archival pilot, sample 1 only

| judge | generator | n | pairwise [95% CI] |
|---|---|---|---|
| llama-nf4 | gpt-6-astra | 10 | 1.000 [1.000, 1.000] |
| llama-nf4 | gpt-6-luna | 10 | 1.000 [1.000, 1.000] |
| llama-nf4 | gpt-6.1-sol | 10 | 1.000 [1.000, 1.000] |
| llama-nf4 | gpt-6.1-terra | 10 | 1.000 [1.000, 1.000] |
| roberta | gpt-6-astra | 140 | 0.993 [0.979, 1.000] |
| roberta | gpt-6-luna | 140 | 0.986 [0.964, 1.000] |
| roberta | gpt-6.1-sol | 140 | 0.986 [0.964, 1.000] |
| roberta | gpt-6.1-terra | 140 | 0.957 [0.921, 0.986] |
| vanguard-modernbert-large | gpt-6-astra | 140 | 1.000 [1.000, 1.000] |
| vanguard-modernbert-large | gpt-6-luna | 140 | 0.993 [0.979, 1.000] |
| vanguard-modernbert-large | gpt-6.1-sol | 140 | 0.979 [0.950, 1.000] |
| vanguard-modernbert-large | gpt-6.1-terra | 140 | 0.993 [0.979, 1.000] |
| tropa-mini-deberta-large | gpt-6-astra | 140 | 0.971 [0.943, 0.993] |
| tropa-mini-deberta-large | gpt-6-luna | 140 | 0.986 [0.964, 1.000] |
| tropa-mini-deberta-large | gpt-6.1-sol | 140 | 0.979 [0.957, 1.000] |
| tropa-mini-deberta-large | gpt-6.1-terra | 140 | 0.964 [0.929, 0.993] |

### Dev outputs reading as human at the listed human-calibration threshold

| judge | generator | n | reads human |
|---|---|---|---|
| llama-nf4 | gpt-6-astra | 11 | N/A; pilot 9.1% at n=70 calibration |
| llama-nf4 | gpt-6-luna | 12 | N/A; pilot 0.0% at n=70 calibration |
| llama-nf4 | gpt-6.1-sol | 12 | N/A; pilot 33.3% at n=70 calibration |
| llama-nf4 | gpt-6.1-terra | 14 | N/A; pilot 0.0% at n=70 calibration |
| roberta | gpt-6-astra | 240 | 48.3% |
| roberta | gpt-6-luna | 233 | 24.9% |
| roberta | gpt-6.1-sol | 240 | 51.2% |
| roberta | gpt-6.1-terra | 240 | 41.2% |
| vanguard-modernbert-large | gpt-6-astra | 240 | 24.6% |
| vanguard-modernbert-large | gpt-6-luna | 233 | 21.5% |
| vanguard-modernbert-large | gpt-6.1-sol | 240 | 27.9% |
| vanguard-modernbert-large | gpt-6.1-terra | 240 | 26.2% |
| tropa-mini-deberta-large | gpt-6-astra | 240 | 22.1% |
| tropa-mini-deberta-large | gpt-6-luna | 233 | 22.3% |
| tropa-mini-deberta-large | gpt-6.1-sol | 240 | 24.6% |
| tropa-mini-deberta-large | gpt-6.1-terra | 240 | 31.7% |
| probe-tropa-mini-deberta-large | gpt-6-astra | 240 | 0.4% |
| probe-tropa-mini-deberta-large | gpt-6-luna | 233 | 1.3% |
| probe-tropa-mini-deberta-large | gpt-6.1-sol | 240 | 0.8% |
| probe-tropa-mini-deberta-large | gpt-6.1-terra | 240 | 2.5% |

### Robustness on fixed random dev outputs

| judge | change | n | verdict flips | AI→human | human→AI | mean absolute score change |
|---|---|---|---|---|---|---|
| llama-nf4 | markdown | 10 | 1 | 1 | 0 | 0.088620 |
| llama-nf4 | linebreaks | 10 | 0 | 0 | 0 | 0.000000 |
| llama-nf4 | typos | 10 | 0 | 0 | 0 | 0.026683 |
| roberta | markdown | 100 | 20 | 20 | 0 | 0.155155 |
| roberta | linebreaks | 100 | 0 | 0 | 0 | 0.000000 |
| roberta | typos | 100 | 6 | 6 | 0 | 0.047068 |

Line-break variants are identical after fold by design. Markdown stripping uses common.prose_view; typos transpose distinct second/third letters in the first two eligible words of at least six letters (exactly two changed words).

### CPU cost (at most two Torch threads, shared machine)

| judge | scored texts | mean wall s/500 original words | median wall s/500 words | CPU s/500 words | peak RSS MiB | texts truncated |
|---|---|---|---|---|---|---|
| llama-nf4 | 209 | 7.71 | 7.30 | 13.97 | 5767 | 6.2% |
| llama (sanity timing only) | 6 | 48.76 | 46.55 | 13.92 | 5803 | 0.0% |
| roberta | 2754 | 3.30 | 1.28 | 1.89 | 3249 | 53.8% |

Timings mix one- and two-thread segments (one thread after host contention). Times exclude loading and include tokenization; original-word normalization understates full-document cost when truncating. These are shared-machine throughput measurements, not idle benchmarks.

### Six-example direction sanity check

| judge | known origin | n | scores (more AI → larger) |
|---|---|---|---|
| llama-nf4 | human | 3 | 0.0104, 0.0096, 0.0142 |
| llama-nf4 | ai | 3 | 0.4358, 0.5197, 0.4952 |
| llama | human | 3 | 0.0129, 0.0096, 0.0093 |
| llama | ai | 3 | 0.4218, 0.4555, 0.4438 |
| roberta | human | 3 | 0.0093, 0.0124, 0.0086 |
| roberta | ai | 3 | 0.2608, 0.0358, 0.0942 |

### Recommendation and measurement limits

Keep the existing private frozen-feature probe as the CPU judge, with vanguard and tropa-mini as
reference diagnostics. Do not add EditLens RoBERTa: AUROC is 0.919 versus about 0.96 for those
ready-made detectors and 0.998 for the probe; pairwise accuracy on instructed mirrors is 0.875.
Markdown stripping moves 20/100 dev outputs from AI to human and two real word typos move 6/100.
These are absolute EditLens perturbation measurements; baseline perturbations were not rerun here.

Llama NF4 is worth retaining as an **optional research diagnostic**, using the pinned non-Instruct
base + adapter, NormedLinear head, four-bucket expected score, fold followed by published cleaning,
and native 1,024-token truncation. It should not replace or join the scoring ensemble yet: the probe
also gets all 20 selected pairs right, no ensemble improvement was measured, and median Llama wall
cost is 7.30 s/500 original words versus RoBERTa's 1.28 s. The published existing-detector timings
were collected under different load, so they are not an idle-machine speed control for this run.
A larger full-calibration evaluation is needed before treating the Llama result as an improvement.

Not measured in full:

- Llama: the full 140-pair evaluation, all 661 corpus texts, all 560 archival sample-1 outputs,
  all 953 dev outputs at the **801-human** threshold, and the 100-output perturbation audit.
  The explicit 215-record fixed pilot above substitutes because the 3B CPU pass is slow even
  after NF4; it took 24m32s including loading, with only 209 new forwards. The 70-human pilot
  calibration and its exploratory dev rates cannot be compared as the requested operating point.
- The optional archival samples 2 and 3 (1,120 additional outputs) were not scored with EditLens;
  RoBERTa covers all 560 sample-1 outputs and Llama covers a fixed 40-output subset.

The public-archival and training-overlap caveats in the original report still apply. EditLens's
published training sources include WritingPrompts, also present here; item-level overlap was not
checked. Neither the pooled calibration FPR nor this pilot establishes the false-positive rate on
fresh commissioned human writing.

Validation: native-tokenization/EOS regression check passed; all selected cache keys, current text
hashes and model revisions were checked before generating these tables; all final scores are finite.
No private texts were copied into tracked files. No commits or pushes were made.
