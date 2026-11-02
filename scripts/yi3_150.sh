#!/usr/bin/env bash
# Yoshi's Island 3 from the start, doubled opportunity bonus, seeds 4, 14, 24, 34, 150 genomes, 150 generations, then videos.
# Run it as a background job: scripts/yi3_150.sh   (log /tmp/yi3_150.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi3_150_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi3-150-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
