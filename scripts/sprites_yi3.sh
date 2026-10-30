#!/usr/bin/env bash
# sprite_platforms (checkerboard platforms as ground), ten seeds x 20 generations (about 70 minutes), detached: scripts/sprites_yi3.sh   (log /tmp/sprites_yi3.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c 'python -W ignore -m smwai.cli experiment --spec configs/experiments/sprites_yi3.toml; echo "all done"' > /tmp/sprites_yi3.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/sprites_yi3.log"
