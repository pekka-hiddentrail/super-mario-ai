#!/usr/bin/env bash
# Hidden-neuron test on Yoshi's Island 1 (3 seeds x 3 variants x 10 generations), then videos of every generation, detached: scripts/hidden_yi1.sh   (log /tmp/hidden_yi1.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  python -W ignore -m smwai.cli experiment --spec configs/experiments/hidden_yi1.toml
  scripts/video_all_gens.sh $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-hidden-yi1-")
  echo "all done"
' > /tmp/hidden_yi1.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/hidden_yi1.log"
