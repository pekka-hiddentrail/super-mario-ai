#!/usr/bin/env bash
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
for a in 2 5 10 15; do python -W ignore scripts/validate_platforms.py runs/exp-longer-yi3-baseline-s4 $a 2>&1 | grep -v pix_fmt; done
