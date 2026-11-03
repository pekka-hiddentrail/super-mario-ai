#!/usr/bin/env bash
# Yoshi's Island 1 and 2, seeds 14 and 24, 40 genomes, up to 40 generations, each run stops at its first clear; then videos.
# Run it as a background job: scripts/yi12_quick.sh   (log /tmp/yi12_quick.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi12_quick_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi12-quick-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
