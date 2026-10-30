"""Which RAM bytes tell that Mario stands on a platform sprite? Compare a state on the swing with one off it, a few frames of standing still after loading.
Usage: probe_riding_flag.py <on.state> <off.state>"""
import sys
from pathlib import Path

import numpy as np
import stable_retro as retro

from smwai import ram as R

GAME = "SuperMarioWorld-Snes-v0"
env = retro.make(GAME, state="YoshiIsland3", use_restricted_actions=retro.Actions.ALL, render_mode=None)
env.reset()
idle = np.zeros(len(env.buttons), dtype=np.int8)


def sample(path):
    env.unwrapped.em.set_state(Path(path).read_bytes())
    for _ in range(4):
        env.step(idle)
    ram = R.wram(env.unwrapped.get_ram())
    st = R.read_state(env.unwrapped.get_ram())
    return ram[:0x2000].copy(), st


a, sa = sample(sys.argv[1])
b, sb = sample(sys.argv[2])
print("on :", "x", sa.x, "y", sa.y, "air", sa.in_air, [(s.number, s.slot, s.x, s.y) for s in sa.sprites])
print("off:", "x", sb.x, "y", sb.y, "air", sb.in_air, [(s.number, s.slot, s.x, s.y) for s in sb.sprites])
for name, addr in (("$72 in air", 0x72), ("$77 blocked", 0x77), ("$1471 on sprite", 0x1471), ("$13EF", 0x13EF), ("$1470", 0x1470), ("$148F", 0x148F), ("$18B6", 0x18B6)):
    print(f"{name:18s} on {a[addr]:3d}  off {b[addr]:3d}")
diff = [(i, a[i], b[i]) for i in range(0x100) if a[i] != b[i]]
print("zero page bytes that differ:", [(f"${i:02X}", x, y) for i, x, y in diff])
