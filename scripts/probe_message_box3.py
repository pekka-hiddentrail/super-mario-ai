"""Check skip_messages: replay a run that ends at a message box and see that it goes on. Usage: probe_message_box3.py <run-dir> <gen>"""
import json
import sys
from pathlib import Path

from smwai import config, train

run, gen = Path(sys.argv[1]), int(sys.argv[2])
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
rec = json.loads((run / "best" / f"gen_{gen:04d}.json").read_text())
env.reset()
for i, m in enumerate(rec["actions"]):
    _, _, term, trunc, info = env.step(train._from_mask(m))
    if env.state.message or i in (294, 295, 296):
        print(i, "x", env.state.x, "message", env.state.message, "frames", env.frames, "stuck", env.stuck, "trunc", trunc)
    if term or trunc:
        break
print("ended at step", i, "of", len(rec["actions"]), "x", env.state.x, "trunc", trunc, "term", term)
