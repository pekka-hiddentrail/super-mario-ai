import time, numpy as np, stable_retro as retro
env = retro.make("SuperMarioWorld-Snes-v0", state="YoshiIsland2", use_restricted_actions=retro.Actions.ALL, render_mode=None)
env.reset(); a = np.zeros(12, dtype=np.int8)
t = time.time()
for _ in range(1500): env.step(a)
print("emulator fps:", 1500 / (time.time() - t))
import pygame, os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
