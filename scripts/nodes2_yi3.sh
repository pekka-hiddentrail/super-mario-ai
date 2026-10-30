#!/usr/bin/env bash
# Death nodes against base, ten seeds x 20 generations (about 50 minutes), detached: scripts/nodes2_yi3.sh   (log /tmp/nodes2_yi3.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c 'python -W ignore -m smwai.cli experiment --spec configs/experiments/nodes2_yi3.toml; echo "all done"' > /tmp/nodes2_yi3.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/nodes2_yi3.log"
