#!/usr/bin/env bash
# Run the smwai CLI inside WSL from the repo root: scripts/smwai.sh train --generations 3
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
exec smwai "$@"
