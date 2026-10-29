"""What does a recorded run spend its time doing? Splits the best run into quarters and reports buttons, ducking, jumping and
how much Mario moves. Usage: run_behaviour.py <run-dir> [<run-dir> ...]"""
import json
import sys
from pathlib import Path

from smwai import config, train
from smwai import ram as R

DUCK = 0x73
for run in map(Path, sys.argv[1:]):
    recs = [json.loads(f.read_text()) for f in sorted((run / "best").glob("gen_*.json"))]
    rec = max(recs, key=lambda r: r["fitness"])
    train.use_config(config.load(run / "config.toml").env)
    env = train._get_env()
    env.reset()
    rows = []
    for mask in rec["actions"]:
        env.step(train._from_mask(mask))
        w = R.wram(env._env.unwrapped.get_ram())
        st = env.state
        rows.append((bool(mask & 32), bool(w[DUCK]), st.in_air, st.x, st.y, bool(mask & 1), bool(mask & 2)))
    n = len(rows)
    print(f"\n{run.name}: best generation {rec['generation']}, {n} steps, ended at x {rows[-1][3]} y {rows[-1][4]}, "
          f"events {sorted({e[1] for e in rec.get('events', [])})}")
    print(f"  {'quarter':8s} {'DOWN':>5s} {'ducking':>8s} {'in air':>7s} {'jump btn':>9s} {'x range':>14s} {'y range':>12s}")
    for q in range(4):
        part = rows[q * n // 4:(q + 1) * n // 4]
        if not part:
            continue
        pct = lambda f: f"{100 * sum(1 for r in part if f(r)) / len(part):.0f}%"
        xs, ys = [r[3] for r in part], [r[4] for r in part]
        print(f"  {q + 1:<8d} {pct(lambda r: r[0]):>5s} {pct(lambda r: r[1]):>8s} {pct(lambda r: r[2]):>7s} "
              f"{pct(lambda r: r[5] or r[6]):>9s} {min(xs):>6d}..{max(xs):<6d} {min(ys):>5d}..{max(ys):<5d}")
