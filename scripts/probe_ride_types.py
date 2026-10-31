"""Put Mario on each platform-like sprite type of a level and read what the game says about riding.
For every listed type: teleport Mario to the first sprite of that type (memory write: the test set-up, no genome is involved), check that the sprite is alive,
drop Mario from above at several x offsets with no input, and print the riding flag $1471, where his feet end up against the sprite, and how the sprite and Mario move
while he stands still. Usage: probe_ride_types.py <state> <number> [number ...]   (numbers like 0x5D)"""
import sys

import numpy as np
import stable_retro as retro

from smwai import ram as R
sys.path.insert(0, __import__("os").path.dirname(__file__))
from probe_sprites import level_sprites

GAME = "SuperMarioWorld-Snes-v0"
state, numbers = sys.argv[1], [int(a, 0) for a in sys.argv[2:]]
env = retro.make(GAME, state=state, use_restricted_actions=retro.Actions.ALL, render_mode=None)
env.reset()
mem = env.unwrapped.data.memory
idle = np.zeros(len(env.buttons), dtype=np.int8)
base = env.unwrapped.em.get_state()
ram = R.wram(env.unwrapped.get_ram())
entries = level_sprites(R.u8(ram, 0xCE) | R.u8(ram, 0xCF) << 8 | R.u8(ram, 0xD0) << 16)


def cur():
    ram = R.wram(env.unwrapped.get_ram())
    return ram, R.read_state(env.unwrapped.get_ram())


def put(x, y):
    mem.assign(0x7E0094, "<u2", x)
    mem.assign(0x7E0096, "<u2", y)
    mem.assign(0x7E007B, "|u1", 0)
    mem.assign(0x7E007D, "|u1", 0)


def travel(x, y):
    """Carry Mario along the level to x (8 px per frame, memory writes) so that the screen scrolls and the sprites there load. False if he dies on the way."""
    env.unwrapped.em.set_state(base)
    env.step(idle)
    pos = 0
    while pos < x:
        pos = min(x, pos + 8)
        put(pos, y)
        env.step(idle)
        if cur()[1].dying:
            return False
    for _ in range(10):
        put(x, y)
        env.step(idle)
    return True


def near(number, x):
    return [s for s in cur()[1].sprites if s.number == number and abs(s.x - x) <= 160]


for number in numbers:
    listed = [(x, y) for n, x, y in entries if n == number]
    print(f"\n== {state} sprite 0x{number:02X}: listed at {listed[:6]}")
    for lx, ly in listed[:3]:
        if not travel(max(0, lx - 40), max(0, ly - 60)):
            print(f"  listed ({lx},{ly}): Mario died on the way")
            continue
        alive = near(number, lx)
        if not alive:
            print(f"  listed ({lx},{ly}): not alive after travelling (sprites {[(s.number, s.x, s.y) for s in cur()[1].sprites]})")
            continue
        arrive = env.unwrapped.em.get_state()
        sp0 = alive[0]
        print(f"  listed ({lx},{ly}): alive at ({sp0.x},{sp0.y}) slot {sp0.slot}")
        found = None
        for height in (40, 70):
            for dx in range(-24, 41, 8):
                env.unwrapped.em.set_state(arrive)
                sp = near(number, lx)
                sp = sp[0] if sp else sp0
                put(sp.x + dx, sp.y - height)
                streak = 0
                for t in range(150):
                    env.step(idle)
                    ram, st = cur()
                    streak = streak + 1 if (not st.in_air and not st.dying) else 0
                    if streak >= 6:
                        break
                flag = R.u8(ram, 0x1471)
                if streak >= 6 and flag:
                    found = (dx, height, flag, st, ram)
                    break
            if found:
                break
        if not found:
            print(f"  could not land on it with $1471 set (last: Mario {st.x},{st.y} air {st.in_air} $1471={R.u8(ram, 0x1471)})")
            continue
        dx, height, flag, st, ram = found
        sp = next(s for s in st.sprites if s.number == number and s.slot == sp0.slot)
        print(f"  landed: dx {dx}, from {height} px up; $1471 = {flag}; Mario ({st.x},{st.y}); sprite ({sp.x},{sp.y}); feet-sprite y {st.y + 32 - sp.y:+d}; $77 {R.u8(ram, 0x77)}")
        track = []
        for t in range(120):
            env.step(idle)
            ram, st = cur()
            sp = next((s for s in st.sprites if s.number == number and s.slot == sp0.slot), None)
            track.append((st.x, st.y, sp.x if sp else None, sp.y if sp else None, R.u8(ram, 0x1471), int(st.in_air)))
        flags = sorted({r[4] for r in track})
        print(f"  120 frames standing still: $1471 values {flags}, in air on {sum(r[5] for r in track)} frames")
        ok = track[0][2] is not None and track[-1][2] is not None
        print(f"  Mario moved ({track[-1][0] - track[0][0]:+d},{track[-1][1] - track[0][1]:+d}), sprite moved {(track[-1][2] - track[0][2], track[-1][3] - track[0][3]) if ok else 'gone'}")
        break
