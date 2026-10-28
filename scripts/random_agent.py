"""Smoke test: random agent plays level 1 for a few episodes and reports speed."""
import time
import numpy as np
from smwai.env import SmwEnv

env = SmwEnv()
rng = np.random.default_rng(0)
t0, steps = time.time(), 0
for ep in range(5):
    obs, info = env.reset(seed=ep)
    total, done = 0.0, False
    while not done:
        action = rng.random(8) < np.array([0.1, 0.3, 0.1, 0.3, 0.05, 0.05, 0.1, 0.6])
        obs, r, term, trunc, info = env.step(action)
        total += r; steps += 1; done = term or trunc
    print(f"episode {ep}: reward {total:.0f} max_x {info['max_x']} frames {info['frames']} died={info['died']}")
print(f"{steps * env.cfg.frame_skip / (time.time() - t0):.0f} emulator frames/s")
