#!/usr/bin/env bash
# Short test on Yoshi's Island 1 (1 seed, 20 genomes, 5 generations), videos of every generation, then one long video, detached:
# scripts/stitch_yi1.sh   (log /tmp/stitch_yi1.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  python -W ignore -m smwai.cli experiment --spec configs/experiments/stitch_yi1.toml
  for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-stitch-yi1-"); do
    scripts/video_all_gens.sh $run
    python -W ignore -m smwai.cli stitch --run $run
  done
  echo "all done"
' > /tmp/stitch_yi1.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/stitch_yi1.log"
