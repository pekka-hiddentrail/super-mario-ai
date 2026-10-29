#!/usr/bin/env bash
# Wait for the ride-climb-more experiment, then run the NEAT + PPO experiment.
cd "$(dirname "$0")/.."
until [ -f runs/_experiments/ride-climb-more/summary.md ]; do sleep 10; done
echo "ride-climb-more finished"; cat runs/_experiments/ride-climb-more/summary.md
bash scripts/experiment.sh configs/experiments/hybrid_yi1.toml
echo; echo "hybrid refinement logs (generation, before->after, * = kept):"
grep -h "hybrid:" runs/_experiments/hybrid-yi1/neat-ppo-s*.log | head -40
