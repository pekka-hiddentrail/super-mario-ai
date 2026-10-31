#!/usr/bin/env bash
# Render the best genome of chosen generations of one run: scripts/video_gens.sh <run-name> <gen> [<gen> ...]
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
run="$1"; shift
for g in "$@"; do python -W ignore -m smwai.cli video --run "$run" --gen "$g" 2>&1 | grep -v pix_fmt | tail -1; done
