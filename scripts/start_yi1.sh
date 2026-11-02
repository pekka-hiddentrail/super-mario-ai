#!/usr/bin/env bash
# Exact small starts on Yoshi's Island 1 (4 nodes/20 links and 8 nodes/40 links; 3 seeds, 15 generations, 25 genomes), videos of every generation and one long video per run, detached:
# scripts/start_yi1.sh   (log /tmp/start_yi1.log). Waits for physics_yi1 to finish first.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  until grep -q "all done" /tmp/physics_yi1.log 2>/dev/null; do sleep 10; done
  python -W ignore -m smwai.cli experiment --spec configs/experiments/start_yi1.toml
  for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-start-yi1-"); do
    scripts/video_all_gens.sh $run
    python -W ignore -m smwai.cli stitch --run $run
  done
  echo "all done"
' > /tmp/start_yi1.log 2>&1 < /dev/null &
disown
echo "started; log /tmp/start_yi1.log"
