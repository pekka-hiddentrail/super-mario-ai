#!/usr/bin/env bash
# Waits for scripts/long_ledges_yi3.sh to finish (its log says "all done"), then the short Yoshi's Island 1 and 2 check (seeds 4 and 14, 20 generations) with videos.
# Run it as a background job: scripts/check_yi12.sh   (log /tmp/check_yi12.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
until grep -q "^all done" /tmp/long_ledges_yi3.log; do sleep 30; done
python -W ignore -m smwai.cli experiment --spec configs/experiments/check_yi12.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-check-yi12-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
