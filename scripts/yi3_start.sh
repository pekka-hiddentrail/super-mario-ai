#!/usr/bin/env bash
# Waits for scripts/check_yi12.sh to finish (its log says "all done"), then Yoshi's Island 3 from the start, 40 generations: doubled opportunity bonus, sparser quieter start, both; with videos.
# Run it as a background job: scripts/yi3_start.sh   (log /tmp/yi3_start.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
until grep -q "^all done" /tmp/check_yi12.log; do sleep 30; done
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi3_start_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi3-start-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
