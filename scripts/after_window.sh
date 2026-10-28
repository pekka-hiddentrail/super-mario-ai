#!/usr/bin/env bash
# Wait for the window experiment to finish (its summary.md appears), then run the physics experiment.
cd "$(dirname "$0")/.."
until [ -f runs/_experiments/window-yi1/summary.md ]; do sleep 10; done
echo "window experiment finished"; cat runs/_experiments/window-yi1/summary.md
exec bash scripts/experiment.sh configs/experiments/physics_yi1.toml
