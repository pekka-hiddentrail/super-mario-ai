#!/usr/bin/env bash
# Wait for the physics-yi1 experiment to finish, then run the Yoshi Island 3 / 4 gap experiment.
cd "$(dirname "$0")/.."
until [ -f runs/_experiments/physics-yi1/summary.md ]; do sleep 10; done
echo "physics-yi1 finished"; cat runs/_experiments/physics-yi1/summary.md
exec bash scripts/experiment.sh configs/experiments/gaps_yi34.toml
