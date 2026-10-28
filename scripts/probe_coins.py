"""Check that Map16 tile 0x2B really is the coin: tile count should drop when the coin counter rises."""
import sys

import numpy as np

from smwai import ram as R
from smwai.env import EnvConfig, SmwEnv

level = sys.argv[1] if len(sys.argv) > 1 else "YoshiIsland1"
env = SmwEnv(EnvConfig(levels=[level], inputs="collect"))
rng = np.random.default_rng(0)


def coin_tiles():
    w = R.wram(env._env.unwrapped.get_ram())
    pages = int(w[0x5D])
    n = 0
    for p in range(pages):
        for i in range(R.TILES_PER_PAGE):
            off = p * R.TILES_PER_PAGE + i
            n += int(w[R.MAP16_LOW + off] == R.COIN_TILE and w[R.MAP16_HIGH + off] == 0)
    return n


for ep in range(12):
    env.reset()
    start_tiles, start_coins = coin_tiles(), env.state.coins
    done = False
    while not done:
        action = rng.random(8) < np.array([0.1, 0.3, 0.1, 0.3, 0.05, 0.05, 0.1, 0.7])
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
    print(f"ep {ep}: coin tiles {start_tiles} -> {coin_tiles()}, coin counter {start_coins} -> {env.state.coins}, max_x {info['max_x']}")
