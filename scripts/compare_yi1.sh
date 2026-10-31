#!/usr/bin/env bash
# Hidden-node comparison on Yoshi's Island 1 (15 generations, 25 genomes, seeds 1-3), videos of every generation and one long video per run, detached:
# scripts/compare_yi1.sh   (log /tmp/compare_yi1.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  python -W ignore -m smwai.cli experiment --spec configs/experiments/compare_yi1.toml
  for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-compare-yi1-"); do
    scripts/video_all_gens.sh $run
    python -W ignore -m smwai.cli stitch --run $run
  done
  echo "all done"
' > /tmp/compare_yi1.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/compare_yi1.log"
