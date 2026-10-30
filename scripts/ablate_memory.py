"""Does a genome react to the death/takeoff memory inputs? Replays the best genome of a run twice: with the memory it trained with
and with the memory blanked (all memory inputs 0). Prints how far each replay got and how many steps pressed different buttons.
Usage: ablate_memory.py <run-dir> [<run-dir> ...]"""
import json
import sys
from pathlib import Path

import neat
import numpy as np

from smwai import config, train

for run in map(Path, sys.argv[1:]):
    recs = [(json.loads(f.read_text()), f) for f in (run / "best").glob("gen_*.json")]
    rec, f = max(recs, key=lambda r: r[0]["fitness"])
    genome = __import__("pickle").load(open(f.with_suffix(".pkl"), "rb"))
    cfg = config.load(run / "config.toml")
    neat_cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(run / "neat.cfg"))
    train.use_config(cfg.env)
    env = train._get_env()
    out = {}
    for label, dm in (("memory", rec.get("death_map")), ("blank", None)):
        env.death_map = dm
        reward, actions, info = train.run_episode(genome, neat_cfg, record=True)
        out[label] = (reward, actions, info["max_x"])
    a, b = out["memory"][1], out["blank"][1]
    n = min(len(a), len(b))
    diff = sum(a[i] != b[i] for i in range(n))
    print(f"{run.name} (gen {rec['generation']}, memory in file: {'yes' if rec.get('death_map') else 'NO'}): "
          f"with memory x {out['memory'][2]} / {len(a)} steps / reward {out['memory'][0]:.0f};  blanked x {out['blank'][2]} / {len(b)} steps / reward {out['blank'][0]:.0f};  "
          f"{diff} of {n} steps press different buttons")
