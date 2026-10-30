#!/usr/bin/env bash
# Wait until a video file exists, then render more generations: scripts/video_after.sh <run-name> <wait-for-gen> <gen> [<gen> ...]
cd "$(dirname "$0")/.."
run="$1"; wait_gen="$2"; shift 2
until [ -f "runs/$run/videos/best_gen_$(printf %04d "$wait_gen").mp4" ]; do sleep 5; done
sleep 3
exec bash scripts/video_gens.sh "$run" "$@"
