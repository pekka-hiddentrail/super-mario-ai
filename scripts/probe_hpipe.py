"""How does entering the horizontal pipe at x ~2793 of Yoshi Island 4 look in memory?
Replays the best recorded run to a few points near its end, then holds RIGHT (with small hops) and watches $88/$89/$141A.
Usage: probe_hpipe.py <run-dir> [generation]"""
import json
import sys
from pathlib import Path

import numpy as np

from smwai import config, train
from smwai import ram as R
from smwai.env import BUTTONS

run = Path(sys.argv[1])
gen = int(sys.argv[2]) if len(sys.argv) > 2 else max(
    (json.loads(f.read_text()) for f in (run / "best").glob("gen_*.json")), key=lambda r: r["fitness"])["generation"]
rec = json.loads((run / "best" / f"gen_{gen:04d}.json").read_text())
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
seen = set()
for back in range(0, 70, 7):
    env.reset()
    for mask in rec["actions"][: len(rec["actions"]) - back]:
        env.step(train._from_mask(mask))
    x0, y0 = env.state.x, env.state.y
    for t in range(160):
        a = np.zeros(len(BUTTONS), dtype=bool)
        a[BUTTONS.index("RIGHT")] = True
        a[BUTTONS.index("B")] = (t % 12) < 3 and back % 2 == 0
        _, _, term, trunc, info = env.step(a)
        w = R.wram(env._env.unwrapped.get_ram())
        cur = (int(w[0x88]), int(w[0x89]), int(w[0x141A]))
        key = (cur[1], cur[2])
        if cur[0] > 0 and key not in seen:
            seen.add(key)
            print(f"back {back}: start x,y {x0},{y0}; t={t} x={info['x']} y={info['y']} timer {cur[0]} action {cur[1]} rooms {cur[2]} "
                  f"in_air {env.state.in_air}")
        if term or trunc:
            break
    print(f"back {back}: start x,y {x0},{y0} -> after 160 frames x={info['x']} y={info['y']} in_air={env.state.in_air}")
