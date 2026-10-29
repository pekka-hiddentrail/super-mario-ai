#!/usr/bin/env bash
# Resume a NEAT run from its newest checkpoint.
# Usage: scripts/resume_run.sh <run-dir-glob> <config.toml> <extra-generations> <workers>
# Example: scripts/resume_run.sh 'runs/yi1-channels-*' configs/yi1_channels.toml 9 3
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
run=$(ls -d $1 | tail -1)
ckpt=$(ls -v "$run"/ckpt-* | tail -1)
echo "resuming $ckpt for $3 more generations"
exec python -W ignore -m smwai.cli train --config "$2" --resume "$ckpt" --generations "$3" --workers "$4"
