#!/usr/bin/env bash
# One-time setup inside WSL Ubuntu: Python 3.12 venv at ~/venv with stable-retro.
set -e
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/venv ] || uv venv --python 3.12 ~/venv
. ~/venv/bin/activate
uv pip install stable-retro pygame numpy
python -c "import retro; print('retro', retro.__file__)"
