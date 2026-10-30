"""Which Map16 tiles are the big (Yoshi) coins? Counts the level's tiles, then puts Mario on the first tile of every rare non-solid kind (memory write: the test
set-up, no genome is involved) and looks at the coin counters. Usage: probe_coin_tiles.py [state ...]"""
import sys
from collections import Counter

import numpy as np
import stable_retro as retro

from smwai import ram as R

GAME = "SuperMarioWorld-Snes-v0"
for state in sys.argv[1:] or ["YoshiIsland1", "YoshiIsland2"]:
    env = retro.make(GAME, state=state, use_restricted_actions=retro.Actions.ALL, render_mode=None)
    env.reset()
    idle = np.zeros(len(env.buttons), dtype=np.int8)
    mem = env.unwrapped.data.memory
    base = env.unwrapped.em.get_state()
    ram = R.wram(env.unwrapped.get_ram())
    seen = {}
    for page in range(0, 12):
        for ty in range(27):
            for c in range(16):
                off = page * R.TILES_PER_PAGE + ty * 16 + c
                tile = (int(ram[R.MAP16_HIGH + off]), int(ram[R.MAP16_LOW + off]))
                seen.setdefault(tile, []).append((page * 256 + c * 16, ty * 16))
    rare = {t: p for t, p in seen.items() if t[0] == 0 and t != (0, 0x25) and 0 < len(p) <= 24}
    print(f"\n== {state}: {len(seen)} kinds of tile; rare non-solid ones: {sorted((f'{h}{l:02X}', len(p)) for (h, l), p in rare.items())}")
    for (h, l), pos in sorted(rare.items()):
        gained = []
        for x, y in pos[:2]:
            env.unwrapped.em.set_state(base)
            env.step(idle)
            r0 = R.wram(env.unwrapped.get_ram())
            before = (R.u8(r0, R.YOSHI_COINS), R.u8(r0, R.COINS))
            for _ in range(3):
                mem.assign(0x7E0094, "<u2", x)
                mem.assign(0x7E0096, "<u2", max(0, y - 8))
                mem.assign(0x7E007B, "|u1", 0)
                mem.assign(0x7E007D, "|u1", 0)
                env.step(idle)
            r1 = R.wram(env.unwrapped.get_ram())
            gained.append((R.u8(r1, R.YOSHI_COINS) - before[0], R.u8(r1, R.COINS) - before[1]))
        flag = "  <== big coin" if any(g[0] > 0 for g in gained) else ""
        print(f"  tile {h}{l:02X}: {len(pos)}x, first at {pos[0]}, (big coins, coins) gained {gained}{flag}")
    env.close()
