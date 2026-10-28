#!/usr/bin/env bash
# Run the test suite inside WSL (where gymnasium, neat and stable-retro are installed).
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -m pytest -q "$@"
