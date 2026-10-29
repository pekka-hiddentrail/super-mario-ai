"""Which Map16 tiles does Mario stand on, and which tile is in front of him when he is pushed to a halt against something?
Replays recorded runs. Usage: probe_standing.py <run-dir> [<run-dir> ...]"""
import json
import sys
from collections import Counter
from pathlib import Path

from smwai import config, train
from smwai import ram as R

stand, wall = Counter(), Counter()
for run in map(Path, sys.argv[1:]):
    recs = [json.loads(f.read_text()) for f in sorted((run / "best").glob("gen_*.json"))]
    rec = max(recs, key=lambda r: r["fitness"])
    train.use_config(config.load(run / "config.toml").env)
    env = train._get_env()
    env.reset()
    for mask in rec["actions"]:
        env.step(train._from_mask(mask))
        w = R.wram(env._env.unwrapped.get_ram())
        st = env.state
        cx, feet = st.x + 8, st.y + 32
        off = R._tile_offset(cx, feet + 2)
        if off is not None and not st.in_air and st.y_speed >= 0:
            stand[(int(w[R.MAP16_HIGH + off]), int(w[R.MAP16_LOW + off]))] += 1
        side = 9 if st.x_speed >= 0 else -9
        off2 = R._tile_offset(cx + side, feet - 8)
        if off2 is not None and st.x_speed == 0 and not st.in_air:      # standing still: something may block him
            wall[(int(w[R.MAP16_HIGH + off2]), int(w[R.MAP16_LOW + off2]))] += 1
fmt = lambda c: {f"{h:x}:{l:02x}": n for (h, l), n in c.most_common(12)}
print("tile under Mario's feet when standing:", fmt(stand))
print("tile ahead of his body when he stands still:", fmt(wall))
