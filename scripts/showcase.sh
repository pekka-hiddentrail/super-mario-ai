#!/usr/bin/env bash
# One training run just to watch (Yoshi Island 1, 3 workers) plus the live view that follows it: http://localhost:8767
# Other experiments can run at the same time. Stop the training with scripts/stop_run.sh.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
name=showcase-yi1-$(date +%Y%m%d-%H%M%S)
nohup python -W ignore -m smwai.cli train --config configs/showcase_yi1.toml --run-dir runs/$name > /tmp/showcase.log 2>&1 &
echo "training: runs/$name (log /tmp/showcase.log)"
bash scripts/restart_live.sh --run $name
