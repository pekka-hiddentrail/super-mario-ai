#!/usr/bin/env bash
# Waits for scripts/yi3_start.sh to finish (its log says "all done"), then Yoshi's Island 1 and 2, seeds 1-4, 50 genomes, 60 generations, with videos.
# Run it as a background job: scripts/yi12_seeds.sh   (log /tmp/yi12_seeds.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
until grep -q "^all done" /tmp/yi3_start.log; do sleep 30; done
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi12_seeds_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi12-seeds-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
