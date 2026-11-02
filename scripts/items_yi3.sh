#!/usr/bin/env bash
# sprite_platforms + big_coins + block_kinds, ten seeds x 20 generations (about 70 minutes), detached: scripts/items_yi3.sh   (log /tmp/items_yi3.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"        # this checkout, not the installed one (a worktree)
setsid nohup bash -c 'python -W ignore -m smwai.cli experiment --spec configs/experiments/items_yi3.toml; echo "all done"' > /tmp/items_yi3.log 2>&1 < /dev/null &
disown
sleep 8
echo "started; log /tmp/items_yi3.log"
