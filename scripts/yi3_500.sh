#!/usr/bin/env bash
# Waits for scripts/yi3_seeds.sh (its log says "all done"), then Yoshi's Island 3 from the start, seeds 11, 12, 24, 14, 40 genomes, 500 generations, then videos.
# Run it as a background job: scripts/yi3_500.sh   (log /tmp/yi3_500.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
until grep -q "^all done" /tmp/yi3_seeds.log; do sleep 30; done
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi3_500_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi3-500-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
