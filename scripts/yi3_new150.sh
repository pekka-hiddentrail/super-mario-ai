#!/usr/bin/env bash
# Waits for scripts/yi3_new.sh (its log says "all done"), then Yoshi's Island 3 from the start with the new base config for 150 generations (3 seeds, 40 genomes), then videos.
# Run it as a background job: scripts/yi3_new150.sh   (log /tmp/yi3_new150.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
until grep -q "^all done" /tmp/yi3_new.log; do sleep 30; done
python -W ignore -m smwai.cli experiment --spec configs/experiments/yi3_new150_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-yi3-new150-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
