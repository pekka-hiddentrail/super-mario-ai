#!/usr/bin/env bash
# Starting-links test on Yoshi's Island 1 (0.5% and 2.5%, 3 seeds, 10 generations), then videos of every generation, detached:
# scripts/sparse_yi1.sh   (log /tmp/sparse_yi1.log). Waits for a running network_yi1 run to finish first.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  until grep -q "all done" /tmp/network_yi1.log 2>/dev/null; do sleep 10; done
  python -W ignore -m smwai.cli experiment --spec configs/experiments/sparse_yi1.toml
  scripts/video_all_gens.sh $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-sparse-yi1-")
  echo "all done"
' > /tmp/sparse_yi1.log 2>&1 < /dev/null &
disown
echo "started; log /tmp/sparse_yi1.log"
