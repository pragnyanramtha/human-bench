# Running and scoring a model

Install with `uv sync --group dev`. The public tasks live in `tasks/dev/` and `tasks/ref/`.
The runner sends each task's model prompt with the fixed instructed system line, with no tools.
It accepts an OpenAI-compatible chat completions endpoint; the base URL should include any API version prefix.

```sh
export MODEL_KEY=your-key
uv run python -m humanbench.harness.run_model run \
  --run-id example --model your-model --base-url https://example.org/v1 \
  --api-key-env MODEL_KEY --split dev --samples 3 --concurrency 4
```

Use `--effort` to set reasoning effort explicitly and `--max-tokens` to set the output limit.
The key is read from the environment and never written to records. Outputs, invocation manifests,
and infrastructure failures are stored under `private/trials/example/`. Repeating a command skips
current successful outputs, retries infrastructure failures, and repeats outputs whose task prompt changed.
The runner removes reasoning enclosed in `<think>` blocks and records programmatic-check results for each output.

To use an LLM judge, configure a judge endpoint and name the judge explicitly:

```sh
export HUMANBENCH_JUDGE_BASE_URL=https://example.org/v1
export HUMANBENCH_JUDGE_KEY_ENV=MODEL_KEY
uv run python -m humanbench.harness.floor_run run \
  --run-id example --judges api:judge-model
```

Use recognised model names so that the LLM judge can exclude judges from the tested model's
own provider. Provider mappings live in `src/humanbench/scoring/floor.py`.
The same endpoint configuration supports `api:<model>` names in `scripts/refcards/memprobe.py`.

The public ledger and detector calibration are sufficient to recompute published leaderboard tables
without private data:

```sh
uv run python scripts/ledger.py tables
```

Re-running detector inference or the full results generator requires the corresponding detector weights,
calibration texts and quality votes. Those private research inputs are not distributed with this release.
