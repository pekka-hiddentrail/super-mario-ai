#!/usr/bin/env bash
# Start the replay viewer in WSL, then open http://localhost:8765 in your Windows browser.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
exec python -W ignore -m smwai.cli view "$@"
