#!/usr/bin/env bash
# Install PPO dependencies into the WSL venv (CPU-only PyTorch; this machine has no NVIDIA GPU).
set -e
export PATH="$HOME/.local/bin:$PATH"
. ~/venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install stable-baselines3
python -c "import torch, stable_baselines3 as sb3; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'sb3', sb3.__version__)"
