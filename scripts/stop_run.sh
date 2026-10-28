#!/usr/bin/env bash
# Stop a training run by config name (main process and its workers). Finished generations are
# already saved on disk; only the generation in progress is lost.
# Usage: scripts/stop_run.sh level1_time
pattern="--config configs/$1.toml"      # matches both `train` and `train-ppo`
pids=$(ps -eo pid=,args= | grep -F -e "$pattern" | grep -v grep | awk '{print $1}')
if [ -z "$pids" ]; then echo "no running run for $1"; exit 1; fi
main=$(ps -eo pid=,etimes=,args= | grep -F -e "$pattern" | grep -v grep | sort -k2 -n -r | head -1 | awk '{print $1}')
kill -INT "$main" 2>/dev/null
sleep 5
left=$(ps -eo pid=,args= | grep -F -e "$pattern" | grep -v grep | awk '{print $1}')
if [ -n "$left" ]; then kill -TERM $left 2>/dev/null; sleep 3; fi
echo "processes left: $(ps -eo args= | grep -F -e "$pattern" | grep -vc grep)"
