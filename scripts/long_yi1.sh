#!/usr/bin/env bash
# Long run on Yoshi's Island 1 (3 seeds, 100 generations, 30 genomes), videos of the improving generations and one long video per run, detached:
# scripts/long_yi1.sh   (log /tmp/long_yi1.log). Waits for the physics and re-render video jobs to finish first.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  until grep -q "all done" /tmp/physics_yi1.log 2>/dev/null && grep -q "all done" /tmp/rerender_compare.log 2>/dev/null; do sleep 10; done
  python -W ignore -m smwai.cli experiment --spec configs/experiments/long_yi1.toml
  for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-long-yi1-"); do
    scripts/video_improved.sh $run
    python -W ignore -m smwai.cli stitch --run $run
  done
  echo "all done"
' > /tmp/long_yi1.log 2>&1 < /dev/null &
disown
echo "started; log /tmp/long_yi1.log"
