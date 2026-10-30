#!/usr/bin/env bash
# Ten seeds x ten generations screen (about 40 minutes), detached: scripts/screen_yi3.sh   (log /tmp/screen_yi3.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c 'python -W ignore -m smwai.cli experiment --spec configs/experiments/screen_yi3.toml; echo "all done"' > /tmp/screen_yi3.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/screen_yi3.log"
