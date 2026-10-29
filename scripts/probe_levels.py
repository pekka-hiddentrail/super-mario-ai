"""Compare candidate levels: size, tile variety, water, and how far a naive right-runner gets.
Usage: probe_levels.py YoshiIsland1 YoshiIsland3 YoshiIsland4"""
import sys
from collections import Counter

import numpy as np

from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv


def naive_run(env, jump_every):
    """Hold RIGHT+run, jump for 12 frames every `jump_every` steps."""
    env.reset()
    done, t, info = False, 0, {}
    while not done:
        a = np.zeros(len(BUTTONS), dtype=bool)
        a[BUTTONS.index("RIGHT")] = a[BUTTONS.index("Y")] = True
        a[BUTTONS.index("B")] = (t % jump_every) < 3
        _, _, term, trunc, info = env.step(a)
        done = term or trunc
        t += 1
    return info["max_x"], bool(info["cleared"])


for level in sys.argv[1:]:
    env = SmwEnv(EnvConfig(levels=[level], inputs="collect"))
    env.reset()
    w = R.wram(env._env.unwrapped.get_ram())
    pages = int(w[0x5D])
    tiles = Counter()
    for p in range(pages):
        for i in range(R.TILES_PER_PAGE):
            off = p * R.TILES_PER_PAGE + i
            tiles[(int(w[R.MAP16_HIGH + off]), int(w[R.MAP16_LOW + off]))] += 1
    solid = sum(n for (h, l), n in tiles.items() if h == 1)
    nonempty_types = sum(1 for (h, l) in tiles if not (h == 0 and l == 0x25))
    sprites = sum(1 for s in R.read_state(w).sprites if s.kind == "enemy")
    best = max(naive_run(env, j) for j in (6, 9, 14, 20))
    print(f"{level}: screens {pages}, solid tiles {solid}, tile types {nonempty_types}, water-level {int(w[0x85])}, "
          f"start x,y {env.state.x},{env.state.y}, enemies at start {sprites}, naive right-runner best {best}")
