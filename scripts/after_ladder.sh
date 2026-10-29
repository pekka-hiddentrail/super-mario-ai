#!/usr/bin/env bash
# Wait for the ladder experiment to finish (its summary.md appears), then run the adaptive-mutation experiment.
cd "$(dirname "$0")/.."
until [ -f runs/_experiments/ladder-yi3/summary.md ]; do sleep 10; done
echo "ladder finished, starting adapt-yi3"
cat runs/_experiments/ladder-yi3/summary.md
exec bash scripts/experiment.sh configs/experiments/adapt_yi3.toml
