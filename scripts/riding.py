"""How often does the best run of each given run stand on a moving platform (bridge or swinging block)?
Prints the totals, and with --trace the platform's position and speed per step.
Usage: riding.py [--trace] <run-dir> [<run-dir> ...]"""
import json
import sys
from pathlib import Path

from smwai import config, train
from smwai import ram as R

args = [a for a in sys.argv[1:] if a != "--trace"]
trace = "--trace" in sys.argv
for run in map(Path, args):
    recs = [json.loads(f.read_text()) for f in sorted((run / "best").glob("gen_*.json"))]
    rec = max(recs, key=lambda r: r["fitness"])
    train.use_config(config.load(run / "config.toml").env)
    env = train._get_env()
    env.reset()
    near = riding = 0
    for i, mask in enumerate(rec["actions"]):
        env.step(train._from_mask(mask))
        st = env.state
        blocks = R.platform_blocks(st)
        on = (not st.in_air) and R.on_platform(blocks, st.x + 8, st.y + 33)
        p = R.nearest_platform(st)
        near += p is not None and abs(p[0]) < 160 and abs(p[1]) < 100
        riding += on
        if trace and p is not None and abs(p[0]) < 160:
            print(f"  step {i} mario ({st.x},{st.y}) air {int(st.in_air)} platform dx {p[0]} dy {p[1]} v ({p[2]:.2f},{p[3]:.2f}) on {int(on)}")
    print(f"{run.name}: {len(rec['actions'])} steps, platform in reach for {near}, standing on a platform for {riding}")
