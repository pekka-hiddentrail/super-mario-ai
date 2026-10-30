"""Replay one recorded attempt and print what Mario does near the end. Usage: probe_pipe_slope.py <run-dir> <gen> [first-step]"""
import json
import sys
from pathlib import Path

from smwai import config, train

run, gen = Path(sys.argv[1]), int(sys.argv[2])
first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
rec = json.loads((run / "best" / f"gen_{gen:04d}.json").read_text())
print("buttons", rec["buttons"], "steps", len(rec["actions"]))
env.reset()
for i, m in enumerate(rec["actions"]):
    _, _, term, trunc, info = env.step(train._from_mask(m))
    s = env.state
    if i >= first:
        print(i, "mask", m, "x", s.x, "y", s.y, "vx", s.x_speed, "vy", s.y_speed, "air", s.in_air, "stuck", env.stuck, "frames", env.frames)
    if term or trunc:
        print("ended", i, "term", term, "trunc", trunc)
        break
