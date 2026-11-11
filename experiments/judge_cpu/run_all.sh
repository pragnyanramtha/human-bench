#!/usr/bin/env bash
# Score every detector (resumable). Usage: ./run_all.sh [classifiers|lm|all]
set -u
cd "$(dirname "$0")"
what=${1:-all}
LOG=../../private/judge_eval/logs
mkdir -p "$LOG"
if [[ $what == classifiers || $what == all ]]; then
  for d in $(uv run python score.py list | grep -v '^lm:'); do
    uv run python score.py "$d" >> "$LOG/score_$d.log" 2>&1 || echo "FAILED $d" >> "$LOG/failures.log"
  done
fi
if [[ $what == lm || $what == all ]]; then
  for d in lm:qwen2.5-1.5b lm:qwen2.5-0.5b; do
    uv run python score.py "$d" >> "$LOG/score_${d/:/_}.log" 2>&1 || echo "FAILED $d" >> "$LOG/failures.log"
  done
fi
echo "run_all $what done"
