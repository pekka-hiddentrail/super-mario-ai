#!/usr/bin/env bash
# The checkpoint rule on Yoshi's Island 1 and 2, seed 14: train, replay each chain from the real start, then videos.
# Run it as a background job: scripts/yi12_checkpoints.sh   (log /tmp/yi12_checkpoints.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi12_checkpoints_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi12-chk-"); do
  echo "chain $run"
  python -W ignore -m smwai.cli chain runs/$run --video
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
