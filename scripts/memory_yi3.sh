#!/usr/bin/env bash
# Takeoff memory and riding-motion test (about 25 minutes), detached: scripts/memory_yi3.sh   (log /tmp/memory_yi3.log, summary in runs/_experiments/memory-yi3/)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c 'python -W ignore -m smwai.cli experiment --spec configs/experiments/memory_yi3.toml; echo "all done"' > /tmp/memory_yi3.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/memory_yi3.log"
