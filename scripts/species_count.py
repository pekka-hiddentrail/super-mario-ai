"""How many species does a compatibility threshold give? Counts them for a fresh population of a config.
Usage: species_count.py <config.toml> <pop_size> <threshold> [<threshold> ...]"""
import sys
import tempfile
from pathlib import Path

import neat

from smwai import config

path, pop = sys.argv[1], int(sys.argv[2])
for th in map(float, sys.argv[3:]):
    cfg = config.load(path)
    text = cfg.neat_text().replace(f"compatibility_threshold = {cfg.neat['compatibility_threshold']}", f"compatibility_threshold = {th}")
    text = text.replace(f"pop_size = {cfg.neat['pop_size']}", f"pop_size = {pop}")
    f = Path(tempfile.mkdtemp()) / "neat.cfg"
    f.write_text(text)
    ncfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(f))
    p = neat.Population(ncfg)
    p.species.speciate(ncfg, p.population, 0)
    sizes = sorted((len(s.members) for s in p.species.species.values()), reverse=True)
    print(f"threshold {th}: {len(sizes)} species, largest {sizes[:6]}")
