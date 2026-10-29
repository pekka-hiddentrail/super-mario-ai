"""Is each observation channel populated? Plays right-and-jump and prints the cell count per channel."""
import sys

import numpy as np

from smwai.env import BUTTONS, EnvConfig, SmwEnv, input_layout

level, mode = (sys.argv[1] if len(sys.argv) > 1 else "YoshiIsland1"), (sys.argv[2] if len(sys.argv) > 2 else "collect")
env = SmwEnv(EnvConfig(levels=[level], inputs=mode))
obs, info = env.reset()
n = 13 * 13
names = input_layout(env.cfg)
for t in range(240):
    a = np.zeros(len(BUTTONS), dtype=bool)
    a[BUTTONS.index("RIGHT")] = a[BUTTONS.index("Y")] = True
    a[BUTTONS.index("B")] = (t % 12) < 3
    obs, _, term, trunc, info = env.step(a)
    if t % 40 == 0 or term or trunc:
        parts, i = [], 0
        for name, kind in names:
            size = n if kind == "grid" else 1
            parts.append(f"{name}={int(np.count_nonzero(obs[i:i + size]))}")
            i += size
        print(f"t={t} x={info['x']} y={info['y']}  " + "  ".join(parts))
    if term or trunc:
        break
