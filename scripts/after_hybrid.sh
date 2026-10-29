#!/usr/bin/env bash
# Wait for the hybrid-yi1 experiment, then run the ride2-yi3 experiment and render videos of all its runs.
cd "$(dirname "$0")/.."
until [ -f runs/_experiments/hybrid-yi1/summary.md ]; do sleep 10; done
echo "hybrid-yi1 finished"; cat runs/_experiments/hybrid-yi1/summary.md
bash scripts/experiment.sh configs/experiments/ride2_yi3.toml
. ~/venv/bin/activate
for d in runs/exp-ride2-yi3-*/; do
  python -W ignore -m smwai.cli video --run "$(basename "$d")" 2>&1 | grep -v pix_fmt | tail -1
done
echo "videos: http://localhost:8765/videos"
