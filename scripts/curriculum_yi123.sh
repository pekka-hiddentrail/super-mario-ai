#!/usr/bin/env bash
# Seeds 4, 14, 24, 34 through Yoshi's Island 1, 2 and 3 (30 genomes, 30 generations per level, nuke on), then videos of every level. Run as a background job: scripts/curriculum_yi123.sh
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli curriculum --spec configs/experiments/curriculum_yi123_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-curriculum-yi123-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
