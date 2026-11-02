#!/usr/bin/env bash
# Four long re-runs of ledges-yi3 (50 genomes, 80 generations), then Yoshi's Island 3 from its start, 3 seeds x 40 genomes x 40 generations. Videos of the improving generations and one long video per run.
# Run it as a background job: scripts/long_ledges_yi3.sh   (log /tmp/long_ledges_yi3.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
for spec in ledge_bonus_edge ledge_bonus_up opp_edge opp_up; do
  python -W ignore -m smwai.cli experiment --spec configs/experiments/long_ledges_yi3_$spec.toml
done
python -W ignore -m smwai.cli curriculum --spec configs/experiments/curriculum_yi3_40_yi.toml
for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-(long-ledges-yi3|yi3-40x40)-"); do
  scripts/video_improved.sh $run
  python -W ignore -m smwai.cli stitch --run $run
done
echo "all done"
