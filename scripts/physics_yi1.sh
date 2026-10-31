#!/usr/bin/env bash
# Jump-prediction inputs on Yoshi's Island 1 (3 seeds, 15 generations, 25 genomes), videos of every generation and one long video per run, detached:
# scripts/physics_yi1.sh   (log /tmp/physics_yi1.log). Waits for compare_yi1 to finish first.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  until grep -q "all done" /tmp/compare_yi1.log 2>/dev/null; do sleep 10; done
  python -W ignore -m smwai.cli experiment --spec configs/experiments/physics_yi1.toml
  for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-physics-yi1-"); do
    scripts/video_all_gens.sh $run
    python -W ignore -m smwai.cli stitch --run $run
  done
  echo "all done"
' > /tmp/physics_yi1.log 2>&1 < /dev/null &
disown
echo "started; log /tmp/physics_yi1.log"
