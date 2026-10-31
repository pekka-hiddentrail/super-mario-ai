#!/usr/bin/env bash
# One run on Yoshi's Island 1 with a 2.5% start (seeds 2 and 3, 80 generations, 30 genomes): videos of the improving generations and one long video.
# Run it as a background job: scripts/two_half_pct_more_yi1.sh   (log /tmp/two_half_pct_more_yi1.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/two_half_pct_more_yi1.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-two-half-pct-more-yi1-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
