#!/usr/bin/env bash
# The reach map, ten seeds x 20 generations (about 50 minutes), detached: scripts/reach_yi3.sh   (log /tmp/reach_yi3.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c 'python -W ignore -m smwai.cli experiment --spec configs/experiments/reach_yi3.toml; echo "all done"' > /tmp/reach_yi3.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/reach_yi3.log"
