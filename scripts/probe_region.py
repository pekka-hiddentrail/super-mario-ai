"""Print the Map16 tiles of a region ('.' empty, '#' solid as the network sees it, otherwise the tile id).
Usage: probe_region.py <state> <x_from_px> <x_to_px> [row_from row_to]"""
import sys

from smwai import ram as R
from smwai.env import EnvConfig, SmwEnv

level, x0, x1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r0, r1 = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (14, 26)
env = SmwEnv(EnvConfig(levels=[level]))
env.reset()
w = R.wram(env._env.unwrapped.get_ram())
print(f"{level}, tiles x {x0 // 16}..{x1 // 16}, rows {r0}..{r1}  (legend: . empty sky, # seen as solid by the network, hex = other tile)")
ids = {}
for ty in range(r0, r1 + 1):
    cells = []
    for tx in range(x0 // 16, x1 // 16 + 1):
        off = (tx // 16) * R.TILES_PER_PAGE + ty * 16 + tx % 16
        hi, lo = int(w[R.MAP16_HIGH + off]), int(w[R.MAP16_LOW + off])
        if hi == 1:
            cells.append("#")
        elif hi == 0 and lo == 0x25:
            cells.append(".")
        else:
            cells.append("?")
            ids[(hi, lo)] = ids.get((hi, lo), 0) + 1
    print(f"{ty:2d} " + "".join(cells))
print("non-empty tiles the network does NOT treat as solid ('?'):", {f"{h:x}:{l:02x}": n for (h, l), n in sorted(ids.items(), key=lambda kv: -kv[1])[:12]})
