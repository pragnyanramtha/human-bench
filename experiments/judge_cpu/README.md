# judge_cpu: can CPU-only ready-made detectors be the HumanBench judge?

Standalone uv project (CPU torch wheels), kept out of the main `humanbench` package.
The write-up is `reports/judge-cpu-baselines.md`. Data lives only under `private/judge_eval/`,
which is gitignored. Never commit it.

## Pipeline (each step caches its work and skips what's already done)

```bash
cd experiments/judge_cpu
uv sync

# 0. raw human sources -> private/judge_eval/raw/  (see "Raw downloads" below)
uv run python build_human.py          # 140 human texts, 5 channels -> human.jsonl
uv run python generate.py briefs      # 14 batched claude calls -> briefs.jsonl
uv run python generate.py mirrors     # 140 calls, claude-sonnet-5-5 / the runner gpt-6.1-sol -> generations/
uv run python generate.py pairs       # -> pairs.jsonl (drops a mirror if >15% of its 8-grams copy the source)
uv run python fetch_models.py         # ~17 GB of HF weights into ~/.cache/huggingface
./run_all.sh all                      # every detector -> scores/<detector>__fold.jsonl
uv run python probe.py <detector>   # optional: frozen-feature + logistic-regression probe
uv run python probe_transfer.py       # optional: probe held-out generator / mode / channel tests
./run_timing.sh                       # clean CPU timing; run only on an otherwise idle box
uv run python ft_timing.py            # optional: CPU fine-tuning cost estimate
uv run python analyze.py              # -> results/summary.json, results/tables.md
```

`score.py <name> [--view light] [--fp32]` scores one detector. `score.py list` lists them.
Default is bf16 autocast on 2 threads. This box has 2 physical cores, and 4 threads were slower.
Weights are copied into RAM at load time (`detectors.materialize`). transformers 5 memory-maps
safetensors, and under memory pressure that re-reads weights from disk on every forward pass. Don't run
the 1.5B LM pair next to other heavy jobs: it needs about 8 GB at peak.

## Raw downloads (into private/judge_eval/raw/)

| dir | source |
|---|---|
| aeslc/train.parquet | Yale-LILY/aeslc, refs/convert/parquet, default/train/0000 |
| stackexchange/s*.parquet | HuggingFaceH4/stack-exchange-preferences parquet shards 486, 510, 630, 606 (gardening, homebrew, poker, opensource) |
| tldr17/part0.parquet | webis/tldr-17, partial-train/0000 |
| yelp/test.parquet | Yelp/yelp_review_full, test |
| imdb/test.parquet | stanfordnlp/imdb, plain_text/test |
| writingprompts/test.parquet | euclaise/writingprompts, test |
| blog/head.csv | first 40 MB of tasksource/blog_authorship_corpus blogtext.csv (HTTP range request) |
| medium/head.csv | first 60 MB of fabiochiu/medium-articles medium_articles.csv (HTTP range request) |

## Views

* `fold` (primary): NFKC, curly quotes to straight, dashes to hyphens, and every whitespace run to a
  single space, applied the same way to both sides. The human sources come from scrapes with different
  typography and paragraph handling, so without this a detector can win on dataset artefacts.
* `light`: typography and paragraphs kept. Used to show how much the artefacts inflate scores.

## Use as the HumanBench judges

The evaluation above chose the detectors. The benchmark then scores texts with two scripts in this folder:

* `judge_run.py --in texts.jsonl [--device cuda] [--cache DIR]`: the three smaller detectors
  (`vanguard-modernbert-large`, `tropa-mini-deberta-large`, `editlens-roberta-large`), one cache file
  per detector, keyed by the hash of the text.
* `gpu_score.py --in texts.jsonl --out scores.jsonl`: the headline detector (`editlens-llama-3.2-3b`),
  standalone so it can be copied to a rented GPU.

`scripts/gpu_bundle.py` (repository root) writes the input file and merges the headline scores back.

Score every text in one cache on one kind of device, the human calibration pool included. The models run
under bfloat16 autocast, whose results differ slightly between a CPU and a GPU and between CPU types: on
2,977 texts scored on both a 4-core CPU and an A40, the largest differences were 0.5 logit (vanguard),
0.13 logit (tropa-mini) and 0.03 (EditLens small, on a 0-1 scale), and 16 of 8,931 human-or-AI verdicts
changed. The v0.1 results use GPU scores for all four detectors.
