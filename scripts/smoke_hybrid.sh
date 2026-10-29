#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash scripts/experiment.sh configs/experiments/smoke_hybrid.toml
cat runs/_experiments/smoke-hybrid/hybrid-s1.log | tail -25
