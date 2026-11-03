#!/usr/bin/env bash
# The generic checkpoint rule (every 300 px) on Yoshi's Island 3, seed 12, then the chain replayed from the real start, then videos.
# Run it as a background job: scripts/checkpoints_every.sh   (log /tmp/checkpoints_every.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/checkpoints_every_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-checkpoints-every-"); do
  echo "chain $run"
  python -W ignore -m smwai.cli chain runs/$run --video
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
