"""Pits in a level: runs of tile columns with no solid tile in the lower part of the screen. Usage: probe_gaps.py <state> ..."""
import sys

import numpy as np

from smwai import ram as R
from smwai.env import EnvConfig, SmwEnv

for level in sys.argv[1:]:
    env = SmwEnv(EnvConfig(levels=[level]))
    env.reset()
    w = R.wram(env._env.unwrapped.get_ram())
    pages = int(w[0x5D])
    ground = []
    for tx in range(pages * 16):
        col = [R.solid_at(w, tx * 16 + 8, ty * 16 + 8) for ty in range(14, 27)]
        ground.append(any(col))
    pits, run = [], None
    for tx, g in enumerate(ground):
        if not g and run is None:
            run = tx
        if g and run is not None:
            pits.append((run, tx - run))
            run = None
    wide = [(x * 16, n) for x, n in pits if n >= 2]
    print(f"{level}: {pages} screens, {len(pits)} pit runs, wider than 1 tile: "
          + ", ".join(f"x {x}px {n} tiles" for x, n in wide[:15]))
