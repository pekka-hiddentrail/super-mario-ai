#!/usr/bin/env bash
# Waits for scripts/wall_yi3.sh (its log says "all done"), then Yoshi's Island 3 from the start with the new base config, then videos.
# Run it as a background job: scripts/yi3_new.sh   (log /tmp/yi3_new.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
until grep -q "^all done" /tmp/wall_yi3.log; do sleep 30; done
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi3_new_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi3-new-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
