#!/usr/bin/env bash
# Pause (or resume) the running training so that the emulator window of play_manual gets the CPU:
#   scripts/pause_training.sh pause [minutes]   (resumes by itself after that many minutes, default 25)
#   scripts/pause_training.sh resume
# Only the processes of `smwai.cli train` / `smwai.cli experiment` are touched.
pids() { pgrep -f 'smwai[.]cli (train|experiment)'; }
case "$1" in
  pause)
    p=$(pids)
    [ -z "$p" ] && { echo "no training running"; exit 0; }
    kill -STOP $p
    echo "paused $(echo $p | wc -w) processes"
    setsid nohup bash -c "sleep $(( ${2:-25} * 60 )); kill -CONT $(echo $p) 2>/dev/null" > /dev/null 2>&1 < /dev/null &
    ;;
  resume)
    p=$(pids)
    [ -z "$p" ] && { echo "no training running"; exit 0; }
    kill -CONT $p
    echo "resumed $(echo $p | wc -w) processes"
    ;;
esac
