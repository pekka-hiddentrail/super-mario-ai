"""Which Map16 tiles are under Mario at a given x in a recorded run? Usage: probe_ground.py <run-dir> <gen> <x>"""
import json
import sys
from pathlib import Path

from smwai import config, train
from smwai import ram as R

run, gen, target_x = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rec = json.loads((run / "best" / f"gen_{gen:04d}.json").read_text())
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
env.reset()
for mask in rec["actions"]:
    env.step(train._from_mask(mask))
    if env.state.x >= target_x:
        break
w = R.wram(env._env.unwrapped.get_ram())
st = env.state
print("mario x,y", st.x, st.y)
tx = (st.x + 8) // 16
for dy in range(-1, 10):
    ty = st.y // 16 + dy
    off = (tx // 16) * R.TILES_PER_PAGE + ty * 16 + tx % 16
    print(f"row {dy:+d} (tile y {ty}): high {int(w[R.MAP16_HIGH + off])} low {int(w[R.MAP16_LOW + off]):02x}")
