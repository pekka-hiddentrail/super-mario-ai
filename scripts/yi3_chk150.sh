#!/usr/bin/env bash
# The checkpoint rule on Yoshi's Island 3, seeds 12, 14, 24, 40 genomes, 150 generations; each chain replayed from the real start and rendered as one video, then the usual videos.
# Run it as a background job: scripts/yi3_chk150.sh   (log /tmp/yi3_chk150.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi3_chk150_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi3-chk150-"); do
  echo "chain $run"
  python -W ignore -m smwai.cli chain runs/$run --video
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
