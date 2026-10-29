#!/usr/bin/env bash
# Start the dashboard in WSL, then open http://localhost:8501 in your Windows browser.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
exec python -W ignore -m smwai.cli ui "$@"
