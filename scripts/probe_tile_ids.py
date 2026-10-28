"""Print the Map16 ids (high:low) of a rectangle of tiles. Usage: probe_tile_ids.py <state> <tile_x0> <tile_x1> <row0> <row1>"""
import sys

from smwai import ram as R
from smwai.env import EnvConfig, SmwEnv

level, x0, x1, r0, r1 = sys.argv[1], *(int(a) for a in sys.argv[2:6])
env = SmwEnv(EnvConfig(levels=[level]))
env.reset()
w = R.wram(env._env.unwrapped.get_ram())
print("      " + " ".join(f"{tx:>5d}" for tx in range(x0, x1 + 1)))
for ty in range(r0, r1 + 1):
    cells = []
    for tx in range(x0, x1 + 1):
        off = (tx // 16) * R.TILES_PER_PAGE + ty * 16 + tx % 16
        cells.append(f"{int(w[R.MAP16_HIGH + off]):x}:{int(w[R.MAP16_LOW + off]):02x}")
    print(f"row {ty:2d} " + " ".join(f"{c:>5s}" for c in cells))
