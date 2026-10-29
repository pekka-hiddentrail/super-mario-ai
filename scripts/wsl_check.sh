#!/usr/bin/env bash
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m stable_retro.import roms 2>&1 | tail -4
python -W ignore - <<'PY'
import stable_retro as r
print([g for g in r.data.list_games() if "World" in g])
print(r.data.list_states("SuperMarioWorld-Snes"))
PY
echo "DISPLAY=$DISPLAY WAYLAND=$WAYLAND_DISPLAY"
