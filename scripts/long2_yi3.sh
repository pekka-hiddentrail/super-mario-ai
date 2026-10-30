#!/usr/bin/env bash
# The second Yoshi Island 3 test (seeds 1 and 2, 120 generations), detached: scripts/long2_yi3.sh   (log: /tmp/long2_yi3.log, summary in runs/_experiments/long2-yi3/)
# Videos of every run are rendered at the end. Progress: tail /tmp/long2_yi3.log ; per-run curves: runs/exp-long2-yi3-*/metrics.csv
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  python -W ignore -m smwai.cli experiment --spec configs/experiments/long2_yi3.toml
  for d in runs/exp-long2-yi3-*/; do python -W ignore -m smwai.cli video --run "$(basename "$d")" 2>&1 | grep -v pix_fmt | tail -1; done
  echo "all done"
' > /tmp/long2_yi3.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/long2_yi3.log"
