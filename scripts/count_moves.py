"""How much do the best runs slide (run + duck), ride Yoshi, or use a cape? Replays the best generation of each run.
Usage: count_moves.py <run-dir> [<run-dir> ...]"""
import json
import sys
from pathlib import Path

from smwai import config, train
from smwai import ram as R

DUCK, FLIGHT = 0x73, 0x1407            # ducking flag; cape flight flag (value non-zero while flying/gliding)

print(f"{'run':44s} {'steps':>5s} {'DOWN held':>10s} {'sliding':>8s} {'on Yoshi':>9s} {'cape':>5s} {'flying':>7s}")
for run in map(Path, sys.argv[1:]):
    recs = [json.loads(f.read_text()) for f in sorted((run / "best").glob("gen_*.json"))]
    rec = max(recs, key=lambda r: r["fitness"])
    train.use_config(config.load(run / "config.toml").env)
    env = train._get_env()
    env.reset()
    n = down = slide = yoshi = cape = fly = 0
    for mask in rec["actions"]:
        env.step(train._from_mask(mask))
        w = R.wram(env._env.unwrapped.get_ram())
        st = env.state
        n += 1
        down += bool(mask & (1 << 5))                                    # DOWN is bit 5 of the button mask
        slide += bool(w[DUCK]) and not st.in_air and abs(st.x_speed) >= 8
        yoshi += st.on_yoshi
        cape += st.powerup == 2
        fly += bool(w[FLIGHT])
    pct = lambda v: f"{100 * v / max(1, n):.0f}%"
    print(f"{run.name:44s} {n:5d} {pct(down):>10s} {pct(slide):>8s} {pct(yoshi):>9s} {pct(cape):>5s} {pct(fly):>7s}")
