"""Does a recorded genome still reproduce its recorded fitness with the current observation code?
Usage: check_genome.py <run-dir> <generation>"""
import json
import pickle
import sys
from pathlib import Path

import neat

from smwai import config, train

run, gen = Path(sys.argv[1]), int(sys.argv[2])
rec = json.loads((run / "best" / f"gen_{gen:04d}.json").read_text())
train.use_config(config.load(run / "config.toml").env)
ncfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                   neat.DefaultStagnation, str(run / "neat.cfg"))
genome = pickle.load(open(run / "best" / f"gen_{gen:04d}.pkl", "rb"))
total, _, info = train.run_episode(genome, ncfg)
print(f"{run.name} gen {gen}: recorded {rec['fitness']:.1f}, now {total:.1f}, max_x {info['max_x']} ->",
      "SAME" if abs(total - rec["fitness"]) < 1e-6 else "DIFFERENT")
