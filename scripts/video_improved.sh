#!/usr/bin/env bash
# Render only the generations where the best run improved (plus the first and the last): scripts/video_improved.sh <run-name> [<run-name> ...]
# For long runs, where a video of every generation takes too long. `smwai stitch` joins whatever videos exist.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
for run in "$@"; do
  for g in $(awk -F, 'NR>1 { if (NR==2 || $2 > best) { print $1; best = $2 } last = $1 } END { print last }' runs/"$run"/metrics.csv | sort -un); do
    python -W ignore -m smwai.cli video --run "$run" --gen "$g" 2>&1 | grep -v pix_fmt | tail -1
  done
done
