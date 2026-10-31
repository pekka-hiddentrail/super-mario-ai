#!/usr/bin/env bash
# Render the best genome of EVERY generation of the given runs: scripts/video_all_gens.sh <run-name> [<run-name> ...]
# FORCE=1 scripts/video_all_gens.sh ... renders again even where a video exists.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
for run in "$@"; do
  for f in runs/"$run"/best/gen_*.json; do
    g=$((10#$(basename "$f" .json | sed 's/gen_//')))
    python -W ignore -m smwai.cli video --run "$run" --gen "$g" ${FORCE:+--force} 2>&1 | grep -v pix_fmt | tail -1
  done
done
