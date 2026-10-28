"""Histogram of non-solid Map16 tiles in a level, to find the coin tile number."""
import sys
from collections import Counter

from smwai import ram as R
from smwai.env import EnvConfig, SmwEnv

for level in sys.argv[1:] or ["YoshiIsland1", "YoshiIsland2"]:
    env = SmwEnv(EnvConfig(levels=[level]))
    env.reset()
    w = R.wram(env._env.unwrapped.get_ram())
    pages = int(w[0x5D])
    c = Counter()
    for p in range(pages):
        for i in range(R.TILES_PER_PAGE):
            off = p * R.TILES_PER_PAGE + i
            c[(int(w[R.MAP16_HIGH + off]), int(w[R.MAP16_LOW + off]))] += 1
    print(level, "pages", pages, "tiles (high, low) top:", [(f"{h:x}:{l:02x}", n) for (h, l), n in c.most_common(14)])
    small = [(f"{h:x}:{l:02x}", n) for (h, l), n in sorted(c.items()) if h == 0 and 0 < n <= 60 and l != 0x25]
    print("  rarer non-solid tiles:", small)
