#!/usr/bin/env bash
# Short runs on Yoshi's Island 2 then 1, then videos of the best run of each, detached: scripts/short_yi12.sh   (log /tmp/short_yi12.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  python -W ignore -m smwai.cli experiment --spec configs/experiments/short_yi2.toml
  python -W ignore -m smwai.cli experiment --spec configs/experiments/short_yi1.toml
  for d in runs/exp-short-yi2-yi2-s1 runs/exp-short-yi1-yi1-s4; do python -W ignore -m smwai.cli video --run "$(basename "$d")" 2>&1 | grep -v pix_fmt | tail -1; done
  echo "all done"
' > /tmp/short_yi12.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/short_yi12.log"
