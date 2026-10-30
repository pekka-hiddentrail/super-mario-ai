#!/usr/bin/env bash
# The one-hour Yoshi Island 3 run (seeds 4 and 7, 80 generations), detached: scripts/hour_yi3.sh   (log: /tmp/hour_yi3.log, summary in runs/_experiments/hour-yi3/)
# Videos of every run are rendered at the end. Progress: tail /tmp/hour_yi3.log ; per-run curves: runs/exp-hour-yi3-*/metrics.csv
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  python -W ignore -m smwai.cli experiment --spec configs/experiments/hour_yi3.toml
  for d in runs/exp-hour-yi3-*/; do python -W ignore -m smwai.cli video --run "$(basename "$d")" 2>&1 | grep -v pix_fmt | tail -1; done
  echo "all done"
' > /tmp/hour_yi3.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/hour_yi3.log"
