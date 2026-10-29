#!/usr/bin/env bash
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
run=$1; shift
python -W ignore -m smwai.cli video --run "$run" --force 2>&1 | grep -v pix_fmt | tail -1
python -W ignore scripts/video_frame.py runs/$run/videos/$(ls runs/$run/videos | grep best_gen | tail -1) "$@"
