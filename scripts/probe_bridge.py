"""Which per-sprite memory values change while a turn-block bridge (sprite 0x5A) extends and retracts?
Loads a saved manual-play state and watches every 12-byte sprite table for the bridge slots. Usage: probe_bridge.py <state-file>"""
import sys
from pathlib import Path

import numpy as np

from smwai import ram as R
from smwai.env import EnvConfig, SmwEnv

env = SmwEnv(EnvConfig(levels=["YoshiIsland3"]))
env.reset()
env._env.unwrapped.em.set_state(Path(sys.argv[1]).read_bytes())
TABLES = [0x00AA, 0x00B6, 0x00C2, 0x00D8, 0x00E4, 0x14D4, 0x14E0, 0x1504, 0x1510, 0x151C, 0x1528, 0x1534, 0x1540, 0x154C, 0x1558,
          0x1564, 0x1570, 0x157C, 0x1588, 0x1594, 0x15AC, 0x15B8, 0x15D0, 0x15F6, 0x160E, 0x1626, 0x163E, 0x187B, 0x14C8]
w = R.wram(env._env.unwrapped.get_ram())
slots = [k for k in range(12) if w[R.SPRITE_STATUS + k] and w[R.SPRITE_NUMBER + k] == 0x5A]
print("bridge slots:", slots)
pad = np.zeros(12, dtype=np.int8)
series = {(k, a): [] for k in slots for a in TABLES}
for t in range(480):
    env._env.step(pad)
    if t % 4 == 0:
        w = R.wram(env._env.unwrapped.get_ram())
        for k in slots:
            for a in TABLES:
                series[(k, a)].append(int(w[a + k]))
for k in slots:
    print(f"\nslot {k}: tables that change (min..max, distinct values)")
    for a in TABLES:
        v = series[(k, a)]
        if len(set(v)) > 1:
            print(f"   ${a:04X}: {min(v)}..{max(v)}, {len(set(v))} values; first samples {v[:24]}")
