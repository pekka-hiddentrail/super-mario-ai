#!/usr/bin/env bash
# The few-hour Yoshi Island 3 test, detached: scripts/long_yi3.sh   (log: /tmp/long_yi3.log, summary in runs/_experiments/long-yi3/)
# Videos of every run are rendered at the end. Progress: tail /tmp/long_yi3.log ; per-run curves: runs/exp-long-yi3-*/metrics.csv
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
nohup bash -c '
  python -W ignore -m smwai.cli experiment --spec configs/experiments/long_yi3.toml
  for d in runs/exp-long-yi3-*/; do python -W ignore -m smwai.cli video --run "$(basename "$d")" 2>&1 | grep -v pix_fmt | tail -1; done
  echo "all done"
' > /tmp/long_yi3.log 2>&1 &
echo "started; log /tmp/long_yi3.log"
