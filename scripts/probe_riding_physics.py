"""What do Mario's own numbers say while he stands on a moving platform? Replays the best run of each given run folder and prints, for the
steps where the sprite model says Mario is on a platform (and the steps around it): in_air flag, y, change of y over the step, the y speed
the game reports, and the platform's own vertical speed. Usage: probe_riding_physics.py <run-dir> [<run-dir> ...]"""
import json
import sys
from pathlib import Path

from smwai import config, train
from smwai import ram as R

for run in map(Path, sys.argv[1:]):
    recs = [json.loads(f.read_text()) for f in sorted((run / "best").glob("gen_*.json"))]
    rec = max(recs, key=lambda r: r["fitness"])
    train.use_config(config.load(run / "config.toml").env)
    env = train._get_env()
    env.reset()
    prev, rows = env.state, []
    for i, mask in enumerate(rec["actions"]):
        env.step(train._from_mask(mask))
        st = env.state
        blocks = R.platform_blocks(st)
        on = R.on_platform(blocks, st.x + 8, st.y + 33)
        p = R.nearest_platform(st)
        rows.append((i, int(st.in_air), st.y, st.y - prev.y, st.y_speed, int(on), None if p is None else round(p[3], 2)))
        prev = st
    hits = [r[0] for r in rows if r[5]]
    print(f"{run.name}: {len(rows)} steps, {len(hits)} on a platform per the sprites")
    shown = set()
    for h in hits:
        for k in range(max(0, h - 2), min(len(rows), h + 3)):
            shown.add(k)
    for k in sorted(shown)[:60]:
        i, air, y, dy, vy, on, pvy = rows[k]
        print(f"  step {i:4d} in_air {air} y {y:4d} dy/step {dy:3d} y_speed {vy:4d} sprite-says-on {on} platform vy {pvy}")
