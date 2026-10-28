#!/usr/bin/env bash
# Render one video and extract a still frame to check the layout: scripts/video_one.sh <run> [gen]
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
run="$1"; gen="${2:-}"
if [ -n "$gen" ]; then python -W ignore -m smwai.cli video --run "$run" --gen "$gen" --force; else python -W ignore -m smwai.cli video --run "$run" --force; fi
f=$(ls -t runs/$run/videos/*.mp4 | head -1)
ff=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p runs/_frames
"$ff" -y -loglevel error -ss 20 -i "$f" -frames:v 1 runs/_frames/still.png
echo "still: runs/_frames/still.png  video: $f"
