#!/usr/bin/env bash
# Stop a running experiment (the runner and its training processes) by the experiment name.
# Usage: scripts/kill_experiment.sh ladder-yi3     (finished generations stay on disk)
name="$1"
pkill -f "after_ladder" 2>/dev/null
pkill -f "experiment --spec configs/experiments/${name//-/_}.toml" 2>/dev/null
pkill -f "_experiments/$name/" 2>/dev/null
sleep 3
echo "processes left: $(ps -eo args= | grep -c -e "_experiments/$name/" -e "experiment --spec")"
