"""Find out how a pipe entry shows up in game memory. Runs right with jumps and taps DOWN now and then; whenever the
pipe timer ($88), the pipe action ($89) or the room counter ($141A) changes it prints Mario's position.
Usage: probe_pipes.py <state> [<state> ...]"""
import sys

import numpy as np

from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv

for level in sys.argv[1:]:
    env = SmwEnv(EnvConfig(levels=[level], stuck_frames=600))
    found = 0
    for jump_every, down_every in ((9, 7), (14, 5), (6, 11), (20, 3)):
        env.reset()
        last = (0, 0, 0)
        for t in range(1500):
            a = np.zeros(len(BUTTONS), dtype=bool)
            a[BUTTONS.index("RIGHT")] = a[BUTTONS.index("Y")] = (t % 40) < 34
            a[BUTTONS.index("B")] = (t % jump_every) < 3
            a[BUTTONS.index("DOWN")] = (t % down_every) < 2
            _, _, term, trunc, info = env.step(a)
            w = R.wram(env._env.unwrapped.get_ram())
            cur = (int(w[0x88]), int(w[0x89]), int(w[0x141A]))
            if cur != last:
                print(f"{level} t={t} x={info['x']} y={info['y']} timer($88)={cur[0]} action($89)={cur[1]} rooms($141A)={cur[2]}"
                      f" level_mode={int(w[0x100])} anim($71)={int(w[0x71])}")
                found += 1
                last = cur
            if term or trunc or found > 30:
                break
    print(level, "events:", found)
