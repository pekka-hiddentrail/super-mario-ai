#!/usr/bin/env bash
# Wait for the gap experiment (Yoshi Island 3/4) to finish, then run the pipe experiment.
cd "$(dirname "$0")/.."
until [ -f runs/_experiments/gaps-yi34/summary.md ]; do sleep 10; done
echo "gaps-yi34 finished"; cat runs/_experiments/gaps-yi34/summary.md
bash scripts/experiment.sh configs/experiments/pipe_yi2.toml
. ~/venv/bin/activate
python scripts/count_events.py "exp-pipe-yi2-*"
