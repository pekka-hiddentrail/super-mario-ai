#!/usr/bin/env bash
# Render best-run videos: scripts/video.sh [--run NAME] [--gen N] [--force]
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
exec python -W ignore -m smwai.cli video "$@"
