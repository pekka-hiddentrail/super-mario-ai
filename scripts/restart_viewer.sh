#!/usr/bin/env bash
# Restart the replay/map/video server in the background (picks up code changes).
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
pkill -f "smwai.cli view" 2>/dev/null
sleep 1
nohup python -W ignore -m smwai.cli view > /dev/null 2>&1 &
sleep 3
curl -s -o /dev/null -w "viewer http status: %{http_code}\n" http://localhost:8765/api/videos
