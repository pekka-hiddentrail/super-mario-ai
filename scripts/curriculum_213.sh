#!/usr/bin/env bash
# Seeds 4, 14, 24, 34 through Yoshi's Island 2, 1 and 3 (30 genomes, 40 generations on every level, nuke on), then videos. Run as a background job: scripts/curriculum_213.sh
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli curriculum --spec configs/experiments/curriculum_213_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-curriculum-213-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
