#!/usr/bin/env bash
# Waits for scripts/yi3_start.sh to finish (its log says "all done"), then Yoshi's Island 3 from the start with the doubled opportunity bonus for 150 generations, with videos.
# Run it as a background job: scripts/yi3_long.sh   (log /tmp/yi3_long.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
until grep -q "^all done" /tmp/yi3_start.log; do sleep 30; done
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi3_long_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi3-long-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
