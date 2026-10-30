"""List the sprites of a level: type number and position, read from the level's sprite list in the ROM (pointer at $CE-$D0), and check it against the sprites
alive at the start. Usage: probe_sprites.py [state ...]   (default YoshiIsland1..4)"""
import sys
from collections import Counter, defaultdict

import stable_retro as retro

from smwai import ram as R

GAME = "SuperMarioWorld-Snes-v0"
rom = open(retro.data.get_romfile_path(GAME), "rb").read()
if len(rom) % 0x8000 == 0x200:
    rom = rom[0x200:]                       # copier header


def lorom(addr):
    return (addr >> 16 & 0x7F) * 0x8000 + (addr & 0x7FFF)


def level_sprites(addr):
    """The entries of a level's sprite list: header byte, then 3 bytes (yyyyEEsY, xxxxssss, number) until 0xFF."""
    off, out = lorom(addr) + 1, []
    while rom[off] != 0xFF:
        b0, b1, num = rom[off], rom[off + 1], rom[off + 2]
        screen = (b0 >> 1 & 1) << 4 | (b1 & 15)
        out.append((num, screen * 256 + (b1 >> 4) * 16, ((b0 & 1) << 8) + (b0 >> 4) * 16))
        off += 3
    return out


if __name__ == "__main__":
    for state in sys.argv[1:] or [f"YoshiIsland{i}" for i in (1, 2, 3, 4)]:
        env = retro.make(GAME, state=state, use_restricted_actions=retro.Actions.ALL, render_mode=None)
        env.reset()
        ram = R.wram(env.unwrapped.get_ram())
        addr = R.u8(ram, 0xCE) | R.u8(ram, 0xCF) << 8 | R.u8(ram, 0xD0) << 16
        entries = level_sprites(addr)
        live = R.read_sprites(ram)
        print(f"\n== {state}: sprite list at ${addr:06X}, {len(entries)} entries")
        kinds = defaultdict(list)
        for num, x, y in entries:
            kinds[num].append(x)
        for num, xs in sorted(kinds.items()):
            print(f"  0x{num:02X} x{len(xs):<2} at x {sorted(xs)[:8]}{' ...' if len(xs) > 8 else ''}")
        print("  alive at start:", sorted((s.number, s.x, s.y) for s in live))
        listed = {(num, x) for num, x, y in entries}
        print("  alive ones found in the list (number, x):", sum((s.number, s.x) in listed or (s.number, s.x & ~15) in listed for s in live), "of", len(live))
        env.close()
