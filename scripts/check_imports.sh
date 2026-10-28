#!/usr/bin/env bash
# Quick import check of the web/UI modules (catches syntax errors before restarting servers).
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -c "import smwai.viewer.server, smwai.viewer.mapdata, smwai.train, smwai.env, smwai.ui.app" 2>&1 | tail -5
echo "import check done"
