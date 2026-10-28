#!/usr/bin/env bash
# Launch manual play inside WSL (window appears on Windows via WSLg). Args pass through.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
PYTHONPATH=src python -W ignore -m smwai.play_manual "$@"
