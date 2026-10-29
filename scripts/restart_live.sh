#!/usr/bin/env bash
# (Re)start the live view in the background: http://localhost:8767 shows the newest best genome of the newest run, with its network.
# Optional: scripts/restart_live.sh --run exp-foo-s4
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
pkill -f "smwai.cli live" 2>/dev/null
sleep 1
nohup python -W ignore -m smwai.cli live "$@" > /tmp/smwai_live.log 2>&1 &
sleep 4
curl -s -o /dev/null -w "live http status: %{http_code}\n" http://localhost:8767/status
