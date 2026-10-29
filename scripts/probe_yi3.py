"""Yoshi Island 3 around x 400-800: terrain, and what moves there (sprites at several moments of a recorded run).
Usage: probe_yi3.py <run-dir>"""
import json
import sys
from pathlib import Path

from smwai import config, train
from smwai import ram as R

run = Path(sys.argv[1])
rec = max((json.loads(f.read_text()) for f in (run / "best").glob("gen_*.json")), key=lambda r: r["fitness"])
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
env.reset()
w = R.wram(env._env.unwrapped.get_ram())
print("terrain, tiles x 24..60 (px 384..960), rows 8..26 ('#' solid as the network sees it, '.' empty, 'o' other tile)")
for ty in range(8, 27):
    row = ""
    for tx in range(24, 61):
        off = (tx // 16) * R.TILES_PER_PAGE + ty * 16 + tx % 16
        hi, lo = int(w[R.MAP16_HIGH + off]), int(w[R.MAP16_LOW + off])
        row += "#" if hi == 1 else "." if (hi == 0 and lo == 0x25) else "o"
    print(f"{ty:2d} {row}")
print("   " + "".join(str((tx // 4) % 10) if tx % 4 == 0 else " " for tx in range(24, 61)) + "   (tile x / 4, last digit)")
print()
shown = 0
for i, mask in enumerate(rec["actions"]):
    env.step(train._from_mask(mask))
    if i % 6 == 0 and 250 <= env.state.x <= 560:
        st = env.state
        sprites = [(hex(s.number), s.kind, s.x, s.y) for s in st.sprites if abs(s.x - st.x) < 200]
        print(f"step {i:3d} Mario x {st.x} y {st.y} in_air {st.in_air}  sprites near: {sprites}")
        shown += 1
        if shown > 14:
            break
