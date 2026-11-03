#!/usr/bin/env bash
# Mario plays the whole game with one population that keeps learning in every level (from the save made when the level began): seeds 1-3, 6 visits, 40 genomes,
# 40 generations per round, up to 3 rounds per level; one video per run with the network.
# Run it as a background job: scripts/map_learn.sh   (log /tmp/map_learn.log)
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
for seed in 1 2 3; do
  echo "== seed $seed"
  python -W ignore -m smwai.cli map-play --seed $seed --visits 6 --learn --generations 40 --pop 40 --rounds 3
done
echo "all done"
