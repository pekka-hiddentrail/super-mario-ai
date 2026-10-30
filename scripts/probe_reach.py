"""Print the reach map on a real level: Mario (M), solid tiles (#), cells some jump lands on (digits = how many arcs), and the time it costs.
Usage: probe_reach.py [level] [steps of running right before the picture]"""
import sys
import time

import numpy as np

from smwai import jumpview
from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv, view

level = sys.argv[1] if len(sys.argv) > 1 else "YoshiIsland3"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
cfg = EnvConfig(levels=[level], inputs="ride", window=[4, 9, 12, 3], jump_fixes=True, cancel_opposites=True, reach_map=True)
env = SmwEnv(cfg)
env.reset()
a = np.zeros(len(BUTTONS), dtype=bool)
a[BUTTONS.index("RIGHT")] = a[BUTTONS.index("Y")] = True
for _ in range(steps):
    env.step(a)
ram = R.wram(env._env.unwrapped.get_ram())
st = env.state
left, right, up, down = R.window(view(cfg))
reach = jumpview.reach_grid(ram, st, view(cfg), ride=True, fixed=True)
solid = R.solid_grid(ram, st, view(cfg))
print(f"{level}: Mario x {st.x} y {st.y} speed {st.x_speed} air {int(st.in_air)}")
for r in range(reach.shape[0]):
    print("".join("M" if (r, c) == (up, left) else str(reach[r, c]) if reach[r, c] else "#" if solid[r, c] else "." for c in range(reach.shape[1])))
t0 = time.time()
for _ in range(100):
    jumpview.reach_grid(ram, st, view(cfg), ride=True, fixed=True)
t1 = time.time()
for _ in range(100):
    jumpview.features(ram, st, ride=True, fixed=True)
    jumpview.run_features(ram, st, ride=True, fixed=True)
t2 = time.time()
print(f"reach map {1000 * (t1 - t0) / 100:.1f} ms per observation; the jump options + run options {1000 * (t2 - t1) / 100:.1f} ms")
