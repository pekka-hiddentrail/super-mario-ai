"""Which button gives the jump the formula describes: A or B? Stand still, hold one button for 1 / 10 / 25 frames,
and print how high Mario rises (pixels) and how long he is in the air. Usage: probe_buttons.py [state]"""
import sys

import numpy as np

from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv

env = SmwEnv(EnvConfig(levels=[sys.argv[1] if len(sys.argv) > 1 else "YoshiIsland2"]))


def step(held):
    pad = np.zeros(12, dtype=np.int8)
    for name in held:
        pad[env._pad[BUTTONS.index(name)]] = 1
    env.frame = env._env.step(pad)[0]
    w = R.wram(env._env.unwrapped.get_ram())
    return R.s16(w, R.PLAYER_Y), int(w[0x72]), int(w[0x7D])


for button in ("A", "B"):
    for hold in (1, 10, 25):
        env.reset()
        for _ in range(30):
            y0, air, _ = step([])
            if not air:
                break
        top, air_frames, vy_first = y0, 0, None
        for t in range(120):
            y, air, vy = step([button] if t < hold else [])
            if vy_first is None:
                vy_first = vy
            top = min(top, y)
            air_frames += 1 if air else 0
            if t > 3 and not air:
                break
        print(f"button {button} hold {hold:2d}: apex {y0 - top:3d} px, first y speed {vy_first - 256 if vy_first > 127 else vy_first}, "
              f"air frames {air_frames}")
