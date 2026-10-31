#!/usr/bin/env bash
# Small start on Yoshi's Island 1 (8 hidden nodes, 0.24% links; seed 10, 20 generations, 25 genomes), videos of every generation and one long video, detached:
# scripts/small_yi1.sh   (log /tmp/small_yi1.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  python -W ignore -m smwai.cli experiment --spec configs/experiments/small_yi1.toml
  for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-small-yi1-"); do
    scripts/video_all_gens.sh $run
    python -W ignore -m smwai.cli stitch --run $run
  done
  echo "all done"
' > /tmp/small_yi1.log 2>&1 < /dev/null &
disown
echo "started; log /tmp/small_yi1.log"
