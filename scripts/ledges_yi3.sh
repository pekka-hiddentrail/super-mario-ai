#!/usr/bin/env bash
# Ledge grid and bonus on Yoshi's Island 3 from three saved starts (3 setups, 3 seeds, 40 generations, 30 genomes): videos of the improving generations and one long video per run.
# Run it as a background job: scripts/ledges_yi3.sh   (log /tmp/ledges_yi3.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/ledges_yi3.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-ledges-yi3-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
