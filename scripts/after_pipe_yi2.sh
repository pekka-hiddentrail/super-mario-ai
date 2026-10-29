#!/usr/bin/env bash
# Wait for the pipe-yi2 experiment, then run the Yoshi Island 4 pipe experiment.
cd "$(dirname "$0")/.."
until [ -f runs/_experiments/pipe-yi2/summary.md ]; do sleep 10; done
echo "pipe-yi2 finished"; cat runs/_experiments/pipe-yi2/summary.md
bash scripts/experiment.sh configs/experiments/pipe_yi4.toml
. ~/venv/bin/activate
python scripts/count_events.py "exp-pipe-yi4-*"
