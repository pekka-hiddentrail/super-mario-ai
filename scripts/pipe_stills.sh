#!/usr/bin/env bash
# Extract stills from the pipe videos: just before the pipe, entering it, and in the next room.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
ff=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p runs/_frames
for pair in "exp-rooms-pipes-yi2-rooms-s2 8" "exp-rooms-pipes-yi4-rooms-s1 6"; do
  set -- $pair
  run=$1; gen=$2
  frame=$(python - <<PY
import json
r = json.load(open("runs/$run/best/gen_$(printf %04d $gen).json"))
print([e[0] for e in r["events"] if e[1] == "pipe"][0])
PY
)
  secs=$(python -c "print($frame/60)")
  echo "$run gen $gen: pipe entered at frame $frame (${secs}s)"
  video=runs/$run/videos/best_gen_$(printf %04d $gen).mp4
  for off in -1.2 0.3 2.5; do
    t=$(python -c "print(max(0, $secs + ($off)))")
    "$ff" -y -loglevel error -ss "$t" -i "$video" -frames:v 1 "runs/_frames/${run}_$off.png"
  done
done
ls runs/_frames
