"""Which Map16 tiles change when Mario collects coins, big (Yoshi) coins or hits blocks?
Plays some episodes with a jumpy right-runner and reports tile ids whose count in the level changed,
next to the game's counters. Usage: probe_collectibles.py YoshiIsland2"""
import sys
from collections import Counter

import numpy as np

from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv

level = sys.argv[1] if len(sys.argv) > 1 else "YoshiIsland2"
env = SmwEnv(EnvConfig(levels=[level], inputs="collect", stuck_frames=400))


def tile_counts():
    w = R.wram(env._env.unwrapped.get_ram())
    c = Counter()
    for p in range(int(w[0x5D])):
        for i in range(R.TILES_PER_PAGE):
            off = p * R.TILES_PER_PAGE + i
            c[(int(w[R.MAP16_HIGH + off]), int(w[R.MAP16_LOW + off]))] += 1
    return c, w


rng = np.random.default_rng(3)
for ep, jump_every in enumerate((5, 7, 9, 12, 16, 20)):
    env.reset()
    before, w0 = tile_counts()
    done, t, info = False, 0, {}
    while not done:
        a = np.zeros(len(BUTTONS), dtype=bool)
        a[BUTTONS.index("RIGHT")] = a[BUTTONS.index("Y")] = True
        a[BUTTONS.index("B")] = (t % jump_every) < 3 or rng.random() < 0.05
        _, _, term, trunc, info = env.step(a)
        done, t = term or trunc, t + 1
    after, w1 = tile_counts()
    changes = {f"{h:x}:{l:02x}": after[(h, l)] - before[(h, l)] for (h, l) in set(before) | set(after)
               if after[(h, l)] != before[(h, l)]}
    print(f"ep {ep}: max_x {info['max_x']} coins {int(w1[R.COINS])} yoshi_coins {int(w1[0x1420])} "
          f"powerup {int(w1[R.POWERUP])}  tile count changes: {dict(sorted(changes.items()))}")
