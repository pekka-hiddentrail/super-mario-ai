"""Where is the swinging block (sprite 0x5F) on screen compared with its sprite position? Replays a recorded Yoshi Island 3 run.
Usage: probe_swing.py <run-dir> <out-dir>"""
import json
import sys
from pathlib import Path

from PIL import Image

from smwai import config, train
from smwai import ram as R

run, out = Path(sys.argv[1]), Path(sys.argv[2])
rec = max((json.loads(f.read_text()) for f in (run / "best").glob("gen_*.json")), key=lambda r: r["fitness"])
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
env.reset()
out.mkdir(parents=True, exist_ok=True)
for i, mask in enumerate(rec["actions"]):
    env.step(train._from_mask(mask))
    if i in (36, 42, 48):
        w = R.wram(env._env.unwrapped.get_ram())
        cam = (R.s16(w, R.LAYER1_X), R.s16(w, R.LAYER1_Y))
        for k in range(12):
            if w[R.SPRITE_STATUS + k] and w[R.SPRITE_NUMBER + k] == 0x5F:
                sx = int(w[R.SPRITE_X_LO + k]) | int(w[R.SPRITE_X_HI + k]) << 8
                sy = int(w[R.SPRITE_Y_LO + k]) | int(w[R.SPRITE_Y_HI + k]) << 8
                tabs = {hex(a): int(w[a + k]) for a in (0x151C, 0x1528, 0x1534, 0x1540, 0x1558, 0xC2, 0xAA, 0xB6)}
                print(f"step {i}: camera {cam}, sprite 0x5F world ({sx},{sy}) -> screen ({sx - cam[0]},{sy - cam[1]}) tables {tabs}")
        Image.fromarray(env.frame).resize((512, 448), Image.NEAREST).save(out / f"swing_step{i}.png")
