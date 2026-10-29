#!/usr/bin/env bash
# Tiny PPO run to check the pipeline end to end and measure speed.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
rm -rf ~/ppo_smoke
sed -e 's/^eval_every = .*/eval_every = 2048/' configs/ppo_level1.toml > ~/ppo_smoke.toml
start=$(date +%s)
python -W ignore -m smwai.cli train-ppo --config ~/ppo_smoke.toml --steps 8192 --envs 4 --run-dir ~/ppo_smoke 2>&1 | tail -12
echo "seconds: $(( $(date +%s) - start ))"
cat ~/ppo_smoke/metrics.csv
