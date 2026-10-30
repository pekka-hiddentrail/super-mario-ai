"""With sprite_platforms on: stand Mario on a checkerboard platform (0x55 in YoshiIsland3, moves sideways) and print what the env makes of it:
riding_platform, the platform grid, and the stuck counter while he is carried. Memory writes set the test up. Usage: check_sprite_platforms.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from probe_sprites import level_sprites
from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv, input_layout

cfg = EnvConfig(levels=["YoshiIsland3"], frame_skip=1, inputs="collect", sprite_platforms=True, stuck_frames=10**6, max_frames=10**6)
env = SmwEnv(cfg)
env.reset()
raw = env._env.unwrapped
mem = raw.data.memory
none = np.zeros(len(BUTTONS), dtype=bool)
ram = R.wram(raw.get_ram())
lx, ly = [(x, y) for n, x, y in level_sprites(R.u8(ram, 0xCE) | R.u8(ram, 0xCF) << 8 | R.u8(ram, 0xD0) << 16) if n == 0x55][0]


def put(x, y):
    mem.assign(0x7E0094, "<u2", x)
    mem.assign(0x7E0096, "<u2", y)
    mem.assign(0x7E007B, "|u1", 0)
    mem.assign(0x7E007D, "|u1", 0)


pos = 0
while pos < lx - 40:
    pos = min(lx - 40, pos + 8)
    put(pos, ly - 60)
    env.step(none)
for _ in range(10):
    put(lx - 40, ly - 60)
    env.step(none)
sp = next(s for s in env.state.sprites if s.number == 0x55)
put(sp.x + 24, sp.y - 40)
platform_col = [n for n, _ in input_layout(cfg)].index("platform")
rows = []
for t in range(90):
    obs, *_ = env.step(none)
    st = env.state
    ride = R.riding_platform(st)
    rows.append((t, st.x, st.y, st.in_air, st.riding, ride is not None, round(ride.vx, 2) if ride else None, env.stuck))
for r in rows[::10]:
    print("t %2d  Mario (%d,%d) air %s riding flag %d  riding_platform %s  platform vx %s  stuck %d" % r)
cells = 169
start = sum(cells if k == "grid" else 1 for _, k in input_layout(cfg)[:platform_col])
grid = obs[start:start + cells].reshape(13, 13)
print("platform grid cells set:", int(grid.sum()), "| enemy grid cells set:", int(obs[cells:2 * cells].sum()))
