#!/usr/bin/env bash
# Small-numbers test on Yoshi's Island 1 (bias cap 1; bias and weight cap 1; 2 seeds, 6 generations), then videos of every generation, detached:
# scripts/calm_yi1.sh   (log /tmp/calm_yi1.log). Waits for the sparse_yi1 test to finish first.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  until grep -q "all done" /tmp/sparse_yi1.log 2>/dev/null; do sleep 10; done
  python -W ignore -m smwai.cli experiment --spec configs/experiments/calm_yi1.toml
  scripts/video_all_gens.sh $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-calm-yi1-")
  echo "all done"
' > /tmp/calm_yi1.log 2>&1 < /dev/null &
disown
echo "started; log /tmp/calm_yi1.log"
