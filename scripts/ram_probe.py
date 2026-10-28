import numpy as np, stable_retro as retro
env = retro.make("SuperMarioWorld-Snes-v0", state="YoshiIsland2", use_restricted_actions=retro.Actions.ALL, render_mode=None)
env.reset()
a = np.zeros(12, dtype=np.int8); a[env.buttons.index("RIGHT")] = 1
for _ in range(120): env.step(a)
ram = env.unwrapped.get_ram()
print("ram len", len(ram))
x = int(ram[0x94]) | int(ram[0x95]) << 8
print("x@0x94", x, "lives(0xDBE)", ram[0xDBE], "coins(0xDBF)", ram[0xDBF], "powerup(0x19)", ram[0x19], "mode(0x100)", ram[0x100])
print("L1X", int(ram[0x1A]) | int(ram[0x1B]) << 8, "Y", int(ram[0x96]) | int(ram[0x97]) << 8)
print("tile row bytes at 0xC800+..", ram[0xC800:0xC800+32].tolist())
st = [int(ram[0x14C8+i]) for i in range(12)]; print("sprite status", st)
print("var x from data:", env.unwrapped.data.lookup_all() if hasattr(env.unwrapped.data, "lookup_all") else "")
from smwai.ram import read_state, tile_grid, wram
ram = wram(ram)
st = read_state(ram)
print(st.x, st.y, st.sprites)
g = tile_grid(ram, st)
for r in range(13):
    print("".join({0: ".", 1: "#", -1: "E", 2: "+"}[int(v)] if not (r == 6 and c == 6) else "M" for c, v in enumerate(g[r])))
