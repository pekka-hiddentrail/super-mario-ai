#!/usr/bin/env bash
# Yoshi's Island 3 from the start with the corrected last chance: double, sparse, both on seeds 14 and 24 (40 genomes, 40 generations), then videos.
# Run it as a background job: scripts/yi3_start2.sh   (log /tmp/yi3_start2.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi3_start2_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi3-start2-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
