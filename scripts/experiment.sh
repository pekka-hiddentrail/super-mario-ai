#!/usr/bin/env bash
# Run an experiment spec: scripts/experiment.sh configs/experiments/ladder_yi3.toml [--summary-only]
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
spec="$1"; shift
exec python -W ignore -m smwai.cli experiment --spec "$spec" "$@"
