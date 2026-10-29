#!/usr/bin/env bash
# Restart the Streamlit dashboard in the background (picks up code changes). Also restarts the viewer server on :8765.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
pkill -f "streamlit run" 2>/dev/null
pkill -f "smwai.cli view" 2>/dev/null
sleep 2
nohup python -W ignore -m smwai.cli ui > /tmp/smwai_ui.log 2>&1 &
sleep 8
curl -s -o /dev/null -w "dashboard http status: %{http_code}\n" http://localhost:8501
curl -s -o /dev/null -w "viewer http status: %{http_code}\n" http://localhost:8765/api/runs
