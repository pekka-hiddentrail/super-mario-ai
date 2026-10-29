#!/usr/bin/env bash
# Wait for the ride-yi3 experiment to finish, then render the best-run video of every run in it.
cd "$(dirname "$0")/.."
. ~/venv/bin/activate
until [ -f runs/_experiments/ride-yi3/summary.md ]; do sleep 10; done
echo "ride-yi3 finished"; cat runs/_experiments/ride-yi3/summary.md
for d in runs/exp-ride-yi3-*/; do
  name=$(basename "$d")
  python -W ignore -m smwai.cli video --run "$name" 2>&1 | grep -v pix_fmt | tail -2
done
echo "videos are listed at http://localhost:8765/videos"
