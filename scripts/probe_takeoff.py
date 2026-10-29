"""Takeoff y speed for jump button A and B at several run speeds (so the jump model can cover both buttons).
Usage: probe_takeoff.py [state]"""
import sys

import numpy as np

from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv

env = SmwEnv(EnvConfig(levels=[sys.argv[1] if len(sys.argv) > 1 else "YoshiIsland2"]))


def s8(v):
    return v - 256 if v > 127 else v


def step(held):
    pad = np.zeros(12, dtype=np.int8)
    for name in held:
        pad[env._pad[BUTTONS.index(name)]] = 1
    env.frame = env._env.step(pad)[0]
    w = R.wram(env._env.unwrapped.get_ram())
    return s8(int(w[R.PLAYER_X_SPEED])), s8(int(w[0x7D])), int(w[0x72]), R.s16(w, R.PLAYER_Y)


print("button  run_frames  x_speed  takeoff_y_speed  apex_px(hold 10)")
for run_frames in (0, 6, 14, 24, 40, 70):
    for button in ("A", "B"):
        env.reset()
        for _ in range(run_frames):
            step(["RIGHT", "Y"])
        vx, vy, air, y0 = step(["RIGHT", "Y", button])
        first = vy
        top = y0
        for t in range(1, 60):
            vx2, vy2, air2, y = step(["RIGHT", "Y"] + ([button] if t < 10 else []))
            top = min(top, y)
            if t > 3 and not air2:
                break
        print(f"{button:>6}  {run_frames:10d}  {vx:7d}  {first:15d}  {y0 - top:5d}")
