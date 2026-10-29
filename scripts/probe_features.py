"""Print the jump-option inputs and what is around Mario for a range of steps of a recorded run.
Usage: probe_features.py <run-dir> <generation> <first-step> <last-step>"""
import json
import sys
from pathlib import Path

from smwai import config, jumpview, train
from smwai import ram as R

run, gen, lo, hi = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rec = json.loads((run / "best" / f"gen_{gen:04d}.json").read_text())
cfg = config.load(run / "config.toml").env
train.use_config(cfg)
env = train._get_env()
env.reset()
for i, mask in enumerate(rec["actions"]):
    env.step(train._from_mask(mask))
    if lo <= i <= hi:
        w = R.wram(env._env.unwrapped.get_ram())
        st = env.state
        f = jumpview.features(w, st, ride=cfg.inputs in ("ride", "ride2"), ceilings=cfg.inputs == "ride2").reshape(6, 5)
        under = R.solid_at(w, st.x + 8, st.y + jumpview.FEET + 1)
        print(f"step {i} x {st.x} y {st.y} air {int(st.in_air)} vx {st.x_speed} vy {st.y_speed} solid_under_feet {under}")
        print("   A1/A10/A25/B1/B10/B25 [land enemy coins dx dy]:", " ".join(str(list(map(int, r))) for r in f))
        print("   sprites:", [(s.number, s.kind, s.x, s.y) for s in st.sprites])
