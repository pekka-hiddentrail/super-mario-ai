"""How does the game score the goal bar? Replays a cleared run and prints, for the last steps before and
after the clear flag: score, pending score, bonus stars, Mario's y, and the goal tape sprite's tables.
Usage: probe_goal.py <run-dir>   (first cleared generation of that run)"""
import json
import sys
from pathlib import Path

import numpy as np

from smwai import config, train
from smwai import ram as R

SCORE, PENDING, BONUS_STARS = 0x0F34, 0x0F40, 0x1900
TABLES = {"151C": 0x151C, "1528": 0x1528, "1534": 0x1534, "1540": 0x1540, "1558": 0x1558, "1594": 0x1594, "187B": 0x187B}


def score(w):
    return int(w[SCORE]) | int(w[SCORE + 1]) << 8 | int(w[SCORE + 2]) << 16


run = Path(sys.argv[1])
rec = next(r for r in (json.loads(f.read_text()) for f in sorted((run / "best").glob("gen_*.json"))) if r["cleared"])
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
env.reset()
rows = []
for i, mask in enumerate(rec["actions"]):
    _, _, term, trunc, info = env.step(train._from_mask(mask))
    w = R.wram(env._env.unwrapped.get_ram())
    slot = next((k for k in range(12) if w[R.SPRITE_NUMBER + k] == 0x7B and w[R.SPRITE_STATUS + k]), None)
    rows.append((i, score(w), int(w[PENDING]) | int(w[PENDING + 1]) << 8, int(w[BONUS_STARS]), env.state.y,
                 None if slot is None else (int(w[R.SPRITE_Y_LO + slot]) | int(w[R.SPRITE_Y_HI + slot]) << 8),
                 None if slot is None else {k: int(w[a + slot]) for k, a in TABLES.items()}, env.state.level_cleared))
    if term or trunc:
        break
print("recorded steps:", len(rec["actions"]))
for r in rows[-6:]:
    print("step %d score %d pending %d stars %d mario_y %d tape_y %s tables %s cleared %s" % r)
print("after the clear flag (noop):")
for k in range(0, 200):
    env.frame = env._env.step(np.zeros(12, dtype=np.int8))[0]
    if k in (0, 4, 10, 30, 60, 120, 199):
        w = R.wram(env._env.unwrapped.get_ram())
        print("  +%d frames: score %d pending %d stars %d" % (k + 1, score(w), int(w[PENDING]) | int(w[PENDING + 1]) << 8, int(w[BONUS_STARS])))
