"""Put Mario on the swinging block (sprite 0x5F) and watch what the block does while he stands still on it.
Loads a saved state, writes Mario's position just above the block (measurement only: the memory write is how the test is set up, no genome is involved),
lets him drop, saves that state, then presses nothing for N frames: prints the block's and Mario's positions and writes a video.
Usage: place_on_swing.py <state file> <out state> <out mp4> [frames to watch]"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv

state_file, out_state, out_mp4 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
watch = int(sys.argv[4]) if len(sys.argv) > 4 else 300
SWING = 0x5F
env = SmwEnv(EnvConfig(levels=["YoshiIsland3"], frame_skip=1, stuck_frames=10**6, max_frames=10**6))
env.reset()
env._env.unwrapped.em.set_state(state_file.read_bytes())
none = np.zeros(len(BUTTONS), dtype=bool)
env.step(none)
mem = env._env.unwrapped.data.memory


def swing(st):
    return [(s.x, s.y) for s in st.sprites if s.number == SWING]


def on_block(st):
    return (not st.in_air) and bool(swing(st)) and R.on_platform(R.platform_blocks(st), st.x + 8, st.y + 33)


sw = swing(env.state)
print("swinging block at", sw, "Mario", (env.state.x, env.state.y))
if not sw:
    sys.exit("no swinging block in this state")
sx, sy = sw[0]
sx, sy = sx - 81, sy - 87           # the chain stands upright in the rest position: the platform is at the top, 81 px left of and 87 px above the sprite's position (read off the picture, see runs/_analysis/swing_sheet.png)
base = env._env.unwrapped.em.get_state()


def drop(dx, height):
    """Put Mario `height` px above the block and `dx` px to the side, let him fall with no input. True when he ends up standing at the block's height."""
    env.reset()
    env._env.unwrapped.em.set_state(base)
    env.step(none)
    mem.assign(0x7E0094, "<u2", sx + dx)
    mem.assign(0x7E0096, "<u2", sy - height)
    mem.assign(0x7E007B, "|u1", 0)
    mem.assign(0x7E007D, "|u1", 0)
    for t in range(120):
        _, _, term, trunc, _ = env.step(none)
        st = env.state
        if term or trunc or st.dying:
            return False
        if not st.in_air and abs(st.y + 32 - sy) <= 24 and abs(st.x + 8 - sx - 8) <= 24:
            return True
    return False


for height in (80, 50, 110):
    for dx in range(-40, 41, 4):
        if drop(dx, height):
            print(f"landed with dx {dx}, {height} px above")
            break
    else:
        continue
    break
else:
    sys.exit("did not land on the block; last state Mario %s block %s" % ((env.state.x, env.state.y), swing(env.state)))
t = 0
print(f"standing at the block: Mario {(env.state.x, env.state.y)}, block {swing(env.state)}")
out_state.write_bytes(env._env.unwrapped.em.get_state())
env.capture = [env.frame.copy()]
TABLES = {"$C2": 0xC2, "$151C": 0x151C, "$1528": 0x1528, "$1534": 0x1534, "$1540": 0x1540, "$154C": 0x154C, "$1558": 0x1558, "$1564": 0x1564, "$1570": 0x1570, "$157C": 0x157C,
          "$1594": 0x1594, "$15AC": 0x15AC, "$15C4": 0x15C4, "$15D0": 0x15D0, "$15F6": 0x15F6, "$1602": 0x1602, "$160E": 0x160E, "$1626": 0x1626, "$163E": 0x163E}
slot = [i for i in range(12) if int(R.wram(env._env.unwrapped.get_ram())[0x9E + i]) == SWING][0]
tabs = []
rows = []
for f in range(watch):
    if f == 600:
        env.capture = None                       # the video only shows the first 600 frames
    env.step(none)
    st = env.state
    rows.append((f + 1, swing(st), st.x, st.y, int(st.in_air), int(on_block(st)), st.lives))
    wr = R.wram(env._env.unwrapped.get_ram())
    tabs.append([int(wr[a + slot]) for a in TABLES.values()])
print("frame after landing | block (x, y) | Mario x y air on-block lives")
for r in rows[:: max(1, watch // 30)][:10]:
    print(f"{r[0]:5d} | {r[1]} | {r[2]} {r[3]} {r[4]} {r[5]} {r[6]}")
import math

CX, CY = 350.5, 216.5                   # centre of the circle Mario rides (from one lap: top / bottom / left / right extremes)
ang, prev, turned = [], None, 0.0
for r in rows:
    a = math.degrees(math.atan2(r[3] - CY, r[2] - CX))     # screen y points down: increasing angle = clockwise
    if prev is not None:
        d = (a - prev + 180) % 360 - 180
        turned += d
    ang.append(turned)
    prev = a
changing = [i for i in range(len(TABLES)) if len({t[i] for t in tabs}) > 1]
print("memory tables that change while riding:", [list(TABLES)[i] for i in changing])
print("frame | " + " | ".join(list(TABLES)[i] for i in changing))
for f in range(0, len(tabs), 40):
    print(f"{f + 1:5d} | " + " | ".join(f"{tabs[f][i]:5d}" for i in changing))
print("frame | turns (clockwise positive) | angle speed (deg per frame, 12-frame mean)")
for i in range(0, len(rows), 12):
    j = min(i + 12, len(rows) - 1)
    print(f"{rows[i][0]:5d} | {ang[i] / 360:+6.2f} | {(ang[j] - ang[i]) / max(1, j - i):+6.2f}")
w = imageio.get_writer(str(out_mp4), fps=60, codec="libx264", quality=7, macro_block_size=None)
for i, fr in enumerate(env.capture or []):
    img = Image.fromarray(fr).resize((fr.shape[1] * 3, fr.shape[0] * 3), Image.NEAREST)
    d = ImageDraw.Draw(img)
    r = rows[min(i, len(rows)) - 1] if i else None
    d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    d.text((6, 5), f"frame {i} | block {r[1] if r else sw} | Mario {(r[2], r[3]) if r else ''} | on block {r[5] if r else 1}", fill=(255, 255, 255))
    w.append_data(np.asarray(img))
w.close()
print("video:", out_mp4)
