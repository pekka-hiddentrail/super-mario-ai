#!/usr/bin/env bash
# Screening on Yoshi's Island 1 (small start, 10 seeds, 80 generations, 30 genomes): videos of the improving generations and one long video per run.
# Run it as a background job: scripts/screen_yi1.sh   (log /tmp/screen_yi1.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/screen_yi1.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-screen-yi1-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
