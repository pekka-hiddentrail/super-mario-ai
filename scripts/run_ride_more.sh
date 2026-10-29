#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash scripts/experiment.sh configs/experiments/ride_climb_more.toml
. ~/venv/bin/activate
for d in runs/exp-ride-climb-more-*/; do
  python -W ignore -m smwai.cli video --run "$(basename "$d")" 2>&1 | grep -v pix_fmt | tail -1
done
echo "videos: http://localhost:8765/videos"
