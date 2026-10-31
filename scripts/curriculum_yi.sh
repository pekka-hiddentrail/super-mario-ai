#!/usr/bin/env bash
# One population through Yoshi's Island 1 and 2 (15 genomes, 15 generations per level, 3 seeds), then videos of the improving generations of every level
# and one long video per level. Run it as a background job: scripts/curriculum_yi.sh   (log /tmp/curriculum_yi.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli curriculum --spec configs/experiments/curriculum_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-curriculum-yi-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
