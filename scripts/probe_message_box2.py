"""After a run reaches a message box: what happens with no input, and which button closes it?
Usage: probe_message_box2.py <run-dir> <gen>"""
import json
import sys
from pathlib import Path

import numpy as np

from smwai import config, train
from smwai import ram as R

run, gen = Path(sys.argv[1]), int(sys.argv[2])
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
rec = json.loads((run / "best" / f"gen_{gen:04d}.json").read_text())
print("buttons:", rec["buttons"], "cleared:", rec["cleared"], "steps:", len(rec["actions"]))


def trace(label, mask, steps=400):
    env.reset()
    for m in rec["actions"]:
        env.step(train._from_mask(m))
    last = None
    for i in range(steps):
        env.stuck = 0                                   # the stuck rule would end the episode: look past it
        _, _, term, trunc, info = env.step(train._from_mask(mask))
        w = R.wram(env._env.unwrapped.get_ram())
        cur = (int(w[0x1426]), int(w[0x1B88]), int(w[0x9D]), int(w[0x13D4]), int(w[0x1DF9]), env.state.x, env.state.in_level, env.state.level_cleared, env.stuck)
        if cur != last:
            print(f"{label} +{i} 1426={cur[0]} 1B88={cur[1]} 9D={cur[2]} 13D4={cur[3]} x={cur[5]} in_level={cur[6]} cleared={cur[7]} stuck={cur[8]} term={term} trunc={trunc}")
            last = cur
        if term:
            print(label, "episode ended at +%d" % i)
            break


for name, mask in [("noop", 0), ("press-A", 1 << rec["buttons"].index("A")), ("press-B", 1 << rec["buttons"].index("B")),
                   ("press-Start", 1 << rec["buttons"].index("START")) if "START" in rec["buttons"] else ("noop2", 0)]:
    trace(name, mask)
