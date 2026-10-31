"""Replay one recorded attempt up to a step and print the Map16 tiles around Mario. Usage: probe_pipe_tiles.py <run-dir> <gen> <step>"""
import json
import sys
from pathlib import Path

from smwai import config, ram as R, train

run, gen, stop = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
rec = json.loads((run / "best" / f"gen_{gen:04d}.json").read_text())
env.reset()
for m in rec["actions"][:stop + 1]:
    env.step(train._from_mask(m))
s = env.state
ram = R.wram(env._env.unwrapped.get_ram())
print("mario x", s.x, "y", s.y, "powerup", s.powerup)
tx0, ty0 = s.x // 16, s.y // 16
for ty in range(ty0 - 4, ty0 + 4):
    row = []
    for tx in range(tx0 - 4, tx0 + 6):
        off = (tx // 16) * R.TILES_PER_PAGE + ty * 16 + tx % 16
        row.append("%d%02X" % (ram[R.MAP16_HIGH + off], ram[R.MAP16_LOW + off]) if hasattr(R, "MAP16_LOW") else "?")
    print("ty", ty, " ".join(row), "<- mario row" if ty == ty0 else "")
