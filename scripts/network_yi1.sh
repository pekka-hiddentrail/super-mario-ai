#!/usr/bin/env bash
# 10-generation run on Yoshi's Island 1, then videos (game + network) of the best run of every generation, detached: scripts/network_yi1.sh   (log /tmp/network_yi1.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  python -W ignore -m smwai.cli experiment --spec configs/experiments/network_yi1.toml
  scripts/video_all_gens.sh $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-network-yi1-")
  echo "all done"
' > /tmp/network_yi1.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/network_yi1.log"
