#!/usr/bin/env bash
# Render the best-run video of each named run: scripts/video_runs.sh <run-name> [<run-name> ...]
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
for r in "$@"; do python -W ignore -m smwai.cli video --run "$r" 2>&1 | grep -v pix_fmt | tail -1; done
