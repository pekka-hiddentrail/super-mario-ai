#!/usr/bin/env bash
# Takeoff memory test (about 8 minutes): scripts/takeoff_yi3.sh ; then scripts/takeoff_summary.py runs/exp-takeoff-yi3-takeoff-s2
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli experiment --spec configs/experiments/takeoff_yi3.toml 2>&1 | grep -v pix_fmt | tail -12
