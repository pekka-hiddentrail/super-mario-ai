#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash scripts/experiment.sh configs/experiments/rooms_pipes.toml
. ~/venv/bin/activate
python scripts/count_events.py "exp-rooms-pipes-*"
