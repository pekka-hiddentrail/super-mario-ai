#!/usr/bin/env bash
# Install video encoding deps (imageio + a bundled ffmpeg) into the WSL venv.
set -e
export PATH="$HOME/.local/bin:$PATH"
. ~/venv/bin/activate
uv pip install imageio imageio-ffmpeg
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
