#!/usr/bin/env bash
# Look at the goal bar scoring in two cleared runs (one low hit, one higher).
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
for r in runs/level1-20261028-185904 runs/yi1-adaptive-20261028-215704; do
  echo "== $r"
  python -W ignore scripts/probe_goal.py "$r" 2>&1 | tail -16
done
