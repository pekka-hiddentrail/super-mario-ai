#!/usr/bin/env bash
# Wait for the pipe-yi4 experiment, then run the explore-pipes experiment and count pipe events.
cd "$(dirname "$0")/.."
until [ -f runs/_experiments/pipe-yi4/summary.md ]; do sleep 10; done
echo "pipe-yi4 finished"; cat runs/_experiments/pipe-yi4/summary.md
. ~/venv/bin/activate
python scripts/count_events.py "exp-pipe-yi4-*" 2>&1 | tail -8
deactivate 2>/dev/null
bash scripts/experiment.sh configs/experiments/explore_pipes.toml
. ~/venv/bin/activate
python scripts/count_events.py "exp-explore-pipes-*"
