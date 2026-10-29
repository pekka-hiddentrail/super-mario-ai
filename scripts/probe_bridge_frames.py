"""Frames of the turn-block bridge at several moments of its cycle, with the bridge sprites' positions and extension value.
Usage: probe_bridge_frames.py <state-file> <out-dir>"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from smwai import ram as R
from smwai.env import EnvConfig, SmwEnv

env = SmwEnv(EnvConfig(levels=["YoshiIsland3"]))
env.reset()
env._env.unwrapped.em.set_state(Path(sys.argv[1]).read_bytes())
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
pad = np.zeros(12, dtype=np.int8)
for t in range(0, 130):
    frame = env._env.step(pad)[0]
    if t in (0, 50, 66, 80, 96, 120):
        w = R.wram(env._env.unwrapped.get_ram())
        cam_x, cam_y = R.s16(w, R.LAYER1_X), R.s16(w, R.LAYER1_Y)
        info = []
        for k in range(12):
            if w[R.SPRITE_STATUS + k] and w[R.SPRITE_NUMBER + k] == 0x5A:
                sx = int(w[R.SPRITE_X_LO + k]) | int(w[R.SPRITE_X_HI + k]) << 8
                sy = int(w[R.SPRITE_Y_LO + k]) | int(w[R.SPRITE_Y_HI + k]) << 8
                info.append(f"slot {k}: world ({sx},{sy}) screen ({sx - cam_x},{sy - cam_y}) ext151C={int(w[0x151C + k])} state={int(w[0xC2 + k])} t1540={int(w[0x1540 + k])}")
        print(f"t={t}: camera ({cam_x},{cam_y}) Mario ({R.s16(w, R.PLAYER_X)},{R.s16(w, R.PLAYER_Y)}); " + "; ".join(info))
        Image.fromarray(frame).resize((512, 448), Image.NEAREST).save(out / f"bridge_t{t:03d}.png")
