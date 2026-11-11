"""Data sourcing for HumanBench (docs/sourcing.md).

Two lanes:

* Lane 1, public corpus (`corpus`): pre-LLM human text from Hugging Face,
  streamed and gated. Only for judge training data and dev material, never
  for test references.
* Lane 2, commissioned writing (`briefs`, `screener`, `submission`, `qa`,
  `refs`): fresh human references for the test split.

Everything this package writes goes under ``private/``, which is gitignored.
"""
