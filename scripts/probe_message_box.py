"""Does a run hit a message box? Replays the best runs of a run folder and prints $1426 (trigger), $1B88 (box state) and $9D (freeze) whenever they change.
Usage: probe_message_box.py <run-dir> [gen ...]"""
import json
import sys
from pathlib import Path

from smwai import config, train
from smwai import ram as R

run = Path(sys.argv[1])
gens = [int(g) for g in sys.argv[2:]] or sorted(int(f.stem.split("_")[1]) for f in (run / "best").glob("gen_*.json"))
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
for g in gens:
    rec = json.loads((run / "best" / f"gen_{g:04d}.json").read_text())
    env.reset()
    last = None
    for i, mask in enumerate(rec["actions"]):
        _, _, term, trunc, info = env.step(train._from_mask(mask))
        w = R.wram(env._env.unwrapped.get_ram())
        cur = (int(w[0x1426]), int(w[0x1B88]), int(w[0x9D]), int(w[0x13D4]))
        if cur != last:
            print(f"gen {g} step {i} x {env.state.x} y {env.state.y} mask {mask} 1426={cur[0]} 1B88={cur[1]} 9D={cur[2]} 13D4={cur[3]} stuck={env.stuck}")
            last = cur
        if term or trunc:
            break
