#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash scripts/experiment.sh configs/experiments/rooms_pipes_3k.toml
. ~/venv/bin/activate
python scripts/pipe_summary.py
