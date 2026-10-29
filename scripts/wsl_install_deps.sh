#!/usr/bin/env bash
# Install project deps into the WSL venv (editable install of this repo).
set -e
export PATH="$HOME/.local/bin:$PATH"
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
uv pip install -e ".[dev]" stable-retro
nproc
python -c "import neat; print('neat-python ok')"
