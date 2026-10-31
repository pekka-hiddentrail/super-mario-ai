"""Step by step trace of a run's best genome: Mario's position and air state, the raw network outputs (0..1) and the node nudge in effect.
Usage: probe_node_trace.py <run-dir> [x-from] [x-to]"""
import json
import pickle
import sys
from pathlib import Path

import neat
import numpy as np

from smwai import config, deathmap, train
from smwai.env import BUTTONS

run = Path(sys.argv[1])
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 99999)
recs = [(json.loads(f.read_text()), f) for f in (run / "best").glob("gen_*.json")]
rec, f = max(recs, key=lambda r: r[0]["fitness"])
genome = pickle.load(open(f.with_suffix(".pkl"), "rb"))
cfg = config.load(run / "config.toml")
neat_cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(run / "neat.cfg"))
train.use_config(cfg.env)
env = train._get_env()
env.death_map = rec.get("death_map")
net = neat.nn.FeedForwardNetwork.create(genome, neat_cfg)
obs, info = env.reset()
print(f"{run.name} gen {rec['generation']}  columns: step x y air y_speed | raw outputs {' '.join(BUTTONS)} | nudge | pressed")
step = 0
while True:
    out = np.array(net.activate(obs.ravel().astype(np.float64)))
    st = env.state
    nud = deathmap.node_nudge((env.death_map or {}).get("nodes"), st.x, st.y) if cfg.env.death_nodes and not st.in_air else [0.0] * 8
    pressed = env.decide(out)
    if lo <= st.x <= hi:
        print(f"{step:4d} {st.x:5d} {st.y:4d} {int(st.in_air)} {st.y_speed:4d} | " + " ".join(f"{v:.2f}" for v in out) + " | "
              + " ".join(f"{v:+.2f}" for v in nud) + " | " + "".join(b[0] if p else "." for b, p in zip(BUTTONS, pressed)))
    obs, _, term, trunc, info = env.step(pressed)
    step += 1
    if term or trunc:
        print("end", info["x"], info["y"], "died" if info.get("died") else "stuck/cut")
        break
