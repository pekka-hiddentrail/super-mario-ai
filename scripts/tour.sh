#!/usr/bin/env bash
# Train 5 generations that keep every population (runs/tour-yi1), then render the species tour video.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
rm -rf runs/tour-yi1
python -W ignore -m smwai.cli train --config configs/tour_yi1.toml --run-dir runs/tour-yi1 2>&1 | grep -v pix_fmt | tail -8
python -W ignore -m smwai.cli tour --run tour-yi1 "$@" 2>&1 | grep -v pix_fmt | tail -30
