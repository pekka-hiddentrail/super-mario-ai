"""Save an emulator state from a replay of a run's best genome, at the first step where a sprite of the given number is near Mario
(a place to start a measurement or a manual play without playing there by hand). Usage: make_state_near.py <run-dir> <sprite number> <max dx> <out.state>"""
import json
import sys
from pathlib import Path

from smwai import config, train

run, number, maxdx, out = Path(sys.argv[1]), int(sys.argv[2], 0), int(sys.argv[3]), Path(sys.argv[4])
recs = [json.loads(f.read_text()) for f in (run / "best").glob("gen_*.json")]
rec = max(recs, key=lambda r: r["fitness"])
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
env.reset()
seen = []
for i, mask in enumerate(rec["actions"]):
    env.step(train._from_mask(mask))
    st = env.state
    near = [s for s in st.sprites if s.number == number and abs(s.x - st.x) <= maxdx]
    if st.sprites and any(s.number == number for s in st.sprites):
        seen.append((i, st.x, st.y, [(s.x, s.y) for s in st.sprites if s.number == number]))
    if near and not st.in_air:
        out.write_bytes(env._env.unwrapped.em.get_state())
        print(f"saved {out}: step {i}, Mario x {st.x} y {st.y}, sprite(s) {[(s.x, s.y) for s in near]}")
        break
else:
    print("no step with that sprite near Mario on the ground; sprite seen at steps:", seen[:5])
