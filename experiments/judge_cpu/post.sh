#!/usr/bin/env bash
# After scoring: light-view runs for the two strongest classifiers, probes, clean timing, analysis.
set -u
cd "$(dirname "$0")"
LOG=../../private/judge_eval/logs
for d in vanguard-modernbert-large tropa-mini-deberta-large; do
  uv run python score.py "$d" --view light >> "$LOG/score_${d}_light.log" 2>&1 || echo "FAILED light $d" >> "$LOG/failures.log"
done
uv run python probe.py vanguard-modernbert-large tropa-mini-deberta-large >> "$LOG/probe.log" 2>&1 || echo "FAILED probe" >> "$LOG/failures.log"
./run_timing.sh
uv run python analyze.py > "$LOG/analyze.log" 2>&1 || echo "FAILED analyze" >> "$LOG/failures.log"
echo "post done"
