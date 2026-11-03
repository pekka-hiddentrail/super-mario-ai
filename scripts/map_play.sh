#!/usr/bin/env bash
# Mario plays the whole game (the start level, then the map and the levels it leads to): seeds 1-5, 8 visits each, one video per run.
# Run it as a background job: scripts/map_play.sh   (log /tmp/map_play.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
for seed in 1 2 3 4 5; do
  echo "== seed $seed"
  python -W ignore -m smwai.cli map-play --seed $seed --visits 8
done
echo "all done"
