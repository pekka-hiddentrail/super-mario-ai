#!/usr/bin/env bash
# Yoshi's Island 3 from the start, seeds 1-3 and 5-14, 40 genomes, 40 generations, then videos.
# Run it as a background job: scripts/yi3_seeds.sh   (log /tmp/yi3_seeds.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi3_seeds_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi3-seeds-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
