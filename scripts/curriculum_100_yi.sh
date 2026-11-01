#!/usr/bin/env bash
# Seed 1 through Yoshi's Island 1 and 2 (30 genomes, 8 hidden nodes, 2.5% start, 100 generations per level), then videos of the improving generations
# of every level and one long video per level. Run it as a background job: scripts/curriculum_100_yi.sh   (log /tmp/curriculum_100_yi.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli curriculum --spec configs/experiments/curriculum_100_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-curriculum-100-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
