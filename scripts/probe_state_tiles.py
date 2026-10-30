"""Print the Map16 tiles around Mario in saved states (to find the tile numbers of coloured pipes).
Usage: probe_state_tiles.py <state file> [<state file> ...]"""
import sys

import stable_retro as retro

from smwai import ram as R

for path in sys.argv[1:]:
    env = retro.make("SuperMarioWorld-Snes-v0", state="YoshiIsland2", use_restricted_actions=retro.Actions.ALL, render_mode=None)
    env.reset()
    env.unwrapped.em.set_state(open(path, "rb").read())      # saves from play_manual (F5) are raw emulator states
    env.step([0] * 12)
    w = R.wram(env.unwrapped.get_ram())
    st = R.read_state(w)
    ram = env.unwrapped.get_ram()
    print(f"{path}: x {st.x} y {st.y} level byte $13BF={int(w[0x13BF])} in_air {st.in_air} powerup {st.powerup}")
    tx0, ty0 = (st.x + 8) // 16, (st.y + 32) // 16          # the tile under the feet
    for dy in range(-3, 4):
        row = []
        for dx in range(-5, 6):
            tx, ty = tx0 + dx, ty0 + dy
            off = (tx // 16) * R.TILES_PER_PAGE + ty * 16 + tx % 16
            row.append(f"{int(w[R.MAP16_HIGH + off]):x}{int(w[R.MAP16_LOW + off]):02x}")
        print(f"  ty {ty0 + dy:3d}: " + " ".join(row) + ("   <- feet row" if dy == 0 else ""))
    env.close()
