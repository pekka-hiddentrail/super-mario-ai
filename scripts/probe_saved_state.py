"""What is in a saved state? Mario's position, the sprites around him, and how each sprite moves over the next frames.
Usage: probe_saved_state.py <state-file> [frames]"""
import sys

import numpy as np

from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv

state_file = sys.argv[1]
frames = int(sys.argv[2]) if len(sys.argv) > 2 else 120
from pathlib import Path

env = SmwEnv(EnvConfig(levels=["YoshiIsland3"]))
env.reset()
env._env.unwrapped.em.set_state(Path(state_file).read_bytes())      # states saved by play_manual are raw emulator states
w = R.wram(env._env.unwrapped.get_ram())
st = R.read_state(w)
print(f"Mario x {st.x} y {st.y} in_air {st.in_air} x_speed {st.x_speed} y_speed {st.y_speed} powerup {st.powerup}")


def sprite_table(w):
    rows = []
    for k in range(12):
        if w[R.SPRITE_STATUS + k]:
            rows.append((k, int(w[R.SPRITE_NUMBER + k]), int(w[R.SPRITE_STATUS + k]),
                         int(w[R.SPRITE_X_LO + k]) | int(w[R.SPRITE_X_HI + k]) << 8,
                         int(w[R.SPRITE_Y_LO + k]) | int(w[R.SPRITE_Y_HI + k]) << 8))
    return rows


print("sprites now (slot, number hex, status, x, y):")
for r in sprite_table(w):
    print(f"   slot {r[0]:2d}  number {r[1]:#04x}  status {r[2]}  x {r[3]}  y {r[4]}")
print(f"\nhow they move while Mario stands still ({frames} frames, a sample every 10):")
history = {}
pad = np.zeros(12, dtype=np.int8)
for t in range(frames):
    env._env.step(pad)
    if t % 10 == 0:
        w = R.wram(env._env.unwrapped.get_ram())
        for r in sprite_table(w):
            history.setdefault((r[0], r[1]), []).append((r[3], r[4]))
for (slot, number), pts in history.items():
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    print(f"   slot {slot:2d} number {number:#04x}: x {min(xs)}..{max(xs)}  y {min(ys)}..{max(ys)}   path {pts[:8]}")
