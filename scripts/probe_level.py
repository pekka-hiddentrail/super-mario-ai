"""Look at a level: size, start position, and how a random agent does. Usage: probe_level.py YoshiIsland1"""
import sys

import numpy as np

from smwai import ram as R
from smwai.env import EnvConfig, SmwEnv

level = sys.argv[1] if len(sys.argv) > 1 else "YoshiIsland1"
cfg = EnvConfig(levels=[level], inputs="channels")
env = SmwEnv(cfg)
obs, info = env.reset()
wram = R.wram(env._env.unwrapped.get_ram())
print("level", level, "screens($5D)", wram[0x5D], "start x,y", info["x"], info["y"], "obs size", obs.shape)
rng = np.random.default_rng(0)
for ep in range(6):
    obs, info = env.reset()
    done, xs = False, []
    while not done:
        action = rng.random(8) < np.array([0.1, 0.3, 0.1, 0.3, 0.05, 0.05, 0.1, 0.6])
        obs, r, term, trunc, info = env.step(action)
        xs.append(info["x"])
        done = term or trunc
    drops = int((np.diff(xs) < -100).sum())
    print(f"ep {ep}: max_x {info['max_x']} frames {info['frames']} died={info['died']} cleared={info['cleared']} "
          f"big x drops={drops} water={env.state.in_water} yoshi={env.state.on_yoshi}")
