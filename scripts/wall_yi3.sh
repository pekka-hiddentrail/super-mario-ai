#!/usr/bin/env bash
# The tall wall: plain, block_kinds, no spin (3 seeds, 30 genomes, 30 generations), then videos.
# Run it as a background job: scripts/wall_yi3.sh   (log /tmp/wall_yi3.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/wall_yi3.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-wall-yi3-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
