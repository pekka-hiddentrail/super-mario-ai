#!/usr/bin/env bash
# The reach map with the rotating platform model and a wider window, ten seeds x 20 generations (about 70 minutes), detached: scripts/swing_yi3.sh   (log /tmp/swing_yi3.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c 'python -W ignore -m smwai.cli experiment --spec configs/experiments/swing_yi3.toml; echo "all done"' > /tmp/swing_yi3.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/swing_yi3.log"
