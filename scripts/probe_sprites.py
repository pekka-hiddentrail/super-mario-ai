"""Which sprite numbers are classed as enemies and sit close to Mario? Usage: probe_sprites.py [state]"""
import sys
from collections import Counter

import numpy as np

from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv

env = SmwEnv(EnvConfig(levels=[sys.argv[1] if len(sys.argv) > 1 else "YoshiIsland1"], inputs="physics"))
seen, near = Counter(), Counter()
for jump_every in (9, 14):
    env.reset()
    for t in range(500):
        a = np.zeros(len(BUTTONS), dtype=bool)
        a[BUTTONS.index("RIGHT")] = a[BUTTONS.index("Y")] = True
        a[BUTTONS.index("B")] = (t % jump_every) < 3
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
        for s in env.state.sprites:
            seen[(s.number, s.kind)] += 1
            if abs(s.x + 8 - (env.state.x + 8)) < 24 and abs(s.y - env.state.y) < 40:
                near[(hex(s.number), s.kind)] += 1
print("all sprites by (number, kind):", {f"{hex(n)}/{k}": c for (n, k), c in seen.most_common(12)})
print("within ~1.5 tiles of Mario:", dict(near.most_common(12)))
