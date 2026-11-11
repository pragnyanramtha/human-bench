#!/usr/bin/env bash
# Clean timing pass: run only when nothing else heavy is running. One process per detector.
set -u
cd "$(dirname "$0")"
LOG=../../private/judge_eval/logs/timing.log
for d in $(uv run python score.py list); do
  uv run python timing.py "$d" 2 >> "$LOG" 2>/dev/null || echo "FAILED timing $d" >> "$LOG"
done
uv run python ft_timing.py >> "$LOG" 2>&1 || echo "FAILED ft_timing" >> "$LOG"
echo "timing done" >> "$LOG"
