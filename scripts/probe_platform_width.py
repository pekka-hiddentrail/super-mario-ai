"""How wide is a platform sprite? Travel to the first sprite of a type, then drop Mario from 40 px above at many x offsets with no input and note where he lands
standing on a platform sprite ($1471 non-zero). Memory writes are the test set-up; no genome is involved.
Usage: probe_platform_width.py <state> <number> [index of the sprite in the level's list]"""
import os
import sys

import numpy as np
import stable_retro as retro

sys.path.insert(0, os.path.dirname(__file__))
from probe_sprites import level_sprites
from smwai import ram as R

GAME = "SuperMarioWorld-Snes-v0"
state, number = sys.argv[1], int(sys.argv[2], 0)
index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
env = retro.make(GAME, state=state, use_restricted_actions=retro.Actions.ALL, render_mode=None)
env.reset()
mem = env.unwrapped.data.memory
idle = np.zeros(len(env.buttons), dtype=np.int8)
base = env.unwrapped.em.get_state()
ram = R.wram(env.unwrapped.get_ram())
entries = level_sprites(R.u8(ram, 0xCE) | R.u8(ram, 0xCF) << 8 | R.u8(ram, 0xD0) << 16)


def cur():
    return R.wram(env.unwrapped.get_ram()), R.read_state(env.unwrapped.get_ram())


def put(x, y):
    mem.assign(0x7E0094, "<u2", x)
    mem.assign(0x7E0096, "<u2", y)
    mem.assign(0x7E007B, "|u1", 0)
    mem.assign(0x7E007D, "|u1", 0)


lx, ly = [(x, y) for n, x, y in entries if n == number][index]
env.unwrapped.em.set_state(base)
env.step(idle)
pos = 0
while pos < lx - 40:
    pos = min(lx - 40, pos + 8)
    put(pos, max(0, ly - 60))
    env.step(idle)
for _ in range(10):
    put(lx - 40, max(0, ly - 60))
    env.step(idle)
sp0 = next((s for s in cur()[1].sprites if s.number == number and abs(s.x - lx) <= 160), None)
if sp0 is None:
    sys.exit("sprite not alive after travelling")
arrive = env.unwrapped.em.get_state()
print(f"sprite 0x{number:02X} at ({sp0.x},{sp0.y}) slot {sp0.slot}")
hits = []
for dx in range(-72, 129, 4):
    env.unwrapped.em.set_state(arrive)
    put(sp0.x + dx, sp0.y - 40)
    streak = 0
    for _ in range(60):
        env.step(idle)
        ram, st = cur()
        streak = streak + 1 if (not st.in_air and not st.dying) else 0
        if streak >= 4:
            break
    on = streak >= 4 and R.u8(ram, 0x1471) != 0
    hits.append((dx, on, st.y + 32 - sp0.y if on else None))
print("dx where he lands on the sprite:", [h[0] for h in hits if h[1]])
print("feet minus sprite y there:", sorted({h[2] for h in hits if h[1]}))
