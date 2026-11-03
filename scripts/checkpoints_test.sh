#!/usr/bin/env bash
# The checkpoint logic: seeds 11 and 12, 40 genomes, 25 generations on Yoshi's Island 3, then videos.
# Run it as a background job: scripts/checkpoints_test.sh   (log /tmp/checkpoints_test.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/checkpoints_test_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-checkpoints-test-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
