"""Find two equal tiles side by side with a pipe tile 2-3 rows below (the 'eye blocks' over a pipe) in the levels' tile maps, and list the sprites at that place.
Usage: probe_find_blocks.py [state ...]"""
import os
import sys

import stable_retro as retro

sys.path.insert(0, os.path.dirname(__file__))
from probe_sprites import level_sprites
from smwai import ram as R

GAME = "SuperMarioWorld-Snes-v0"
for state in sys.argv[1:] or [f"YoshiIsland{i}" for i in (1, 2, 3, 4)]:
    env = retro.make(GAME, state=state, use_restricted_actions=retro.Actions.ALL, render_mode=None)
    env.reset()
    ram = R.wram(env.unwrapped.get_ram())
    entries = level_sprites(R.u8(ram, 0xCE) | R.u8(ram, 0xCF) << 8 | R.u8(ram, 0xD0) << 16)

    def tile(tx, ty):
        off = (tx // 16) * R.TILES_PER_PAGE + ty * 16 + tx % 16
        return int(ram[R.MAP16_HIGH + off]) << 8 | int(ram[R.MAP16_LOW + off])

    for ty in range(1, 22):
        for tx in range(1, 12 * 16 - 4):
            col = [tile(tx, ty + d) for d in range(4)]
            if len(set(col)) == 1 and col[0] not in (0, 0x25) and all(tile(tx + 3, ty + d) == col[0] for d in range(4)):
                eyes = (tile(tx + 1, ty), tile(tx + 2, ty))
                if eyes[0] == eyes[1] and eyes[0] != col[0]:
                    sp = [(n, x, y) for n, x, y in entries if abs(x - tx * 16) < 96 and abs(y - ty * 16) < 96]
                    print(f"{state}: walls tile 0x{col[0]:03X} at tile ({tx},{ty}) = pixel ({tx * 16},{ty * 16}); top middle 0x{eyes[0]:03X} x2; rows below the middle: "
                          f"{[(hex(tile(tx + 1, ty + d)), hex(tile(tx + 2, ty + d))) for d in (1, 2, 3)]}; sprites near: {sp}")
    env.close()
