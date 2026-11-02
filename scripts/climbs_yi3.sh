#!/usr/bin/env bash
# Yoshi's Island 3 from the four saved climb starts (3 seeds, 40 generations, 30 genomes): videos of the improving generations and one long video per run.
# Run it as a background job: scripts/climbs_yi3.sh   (log /tmp/climbs_yi3.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/climbs_yi3.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-climbs-yi3-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
