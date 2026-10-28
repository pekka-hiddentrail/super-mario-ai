"""NEAT training: parallel episode evaluation, per-generation best-run recording."""
import json
import pickle
import shutil
import time
from pathlib import Path

import neat
import numpy as np

from .env import BUTTONS, SmwEnv

_ENV_CFG = None   # set before the worker pool forks
_ENV = None       # one emulator per process (stable-retro allows only one)


def _get_env():
    global _ENV
    if _ENV is None:
        _ENV = SmwEnv(_ENV_CFG)
    return _ENV


def use_config(env_cfg):
    """Select the env config for this process (reuses the emulator if one exists)."""
    global _ENV_CFG
    _ENV_CFG = env_cfg
    if _ENV is not None:
        _ENV.cfg = env_cfg


def _to_mask(action):
    return int(sum(1 << i for i, on in enumerate(action) if on))


def _from_mask(mask):
    return np.array([(mask >> i) & 1 for i in range(len(BUTTONS))], dtype=bool)


def run_episode(genome, config, record=False):
    """Play one episode with the genome's network. Returns (reward, actions, info)."""
    env = _get_env()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    obs, info = env.reset()
    total, actions, path = 0.0, [], [[info["x"], info["y"]]]
    while True:
        action = np.array(net.activate(obs.ravel().astype(np.float64))) > 0.5
        if record:
            actions.append(_to_mask(action))
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        if record:
            path.append([info["x"], info["y"]])
        if term or trunc:
            info["path"] = path
            return total, actions, info


def replay_actions(masks):
    """Re-run a recorded action list headless; returns (reward, info) with info["path"]."""
    env = _get_env()
    _, info = env.reset()
    total, path = 0.0, [[info["x"], info["y"]]]
    for mask in masks:
        _, reward, term, trunc, info = env.step(_from_mask(mask))
        total += reward
        path.append([info["x"], info["y"]])
        if term or trunc:
            break
    info["path"] = path
    return total, info


def eval_genome(genome, config):
    return run_episode(genome, config)[0]


class RunReporter(neat.reporting.BaseReporter):
    """Writes metrics.csv and records the best run of every generation."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        (self.dir / "best").mkdir(parents=True, exist_ok=True)
        self.generation = -1
        self.gen_start = time.time()
        self.best_ever = -np.inf
        self.csv = self.dir / "metrics.csv"
        if not self.csv.exists():
            self.csv.write_text("generation,best,mean,species,max_x,seconds,deterministic\n")

    def start_generation(self, generation):
        self.generation = generation
        self.gen_start = time.time()

    def post_evaluate(self, config, population, species, best_genome):
        fits = [g.fitness for g in population.values()]
        best = max(population.values(), key=lambda g: g.fitness)
        reward, actions, info = run_episode(best, config, record=True)
        deterministic = abs(reward - best.fitness) < 1e-6
        stem = self.dir / "best" / f"gen_{self.generation:04d}"
        with open(stem.with_suffix(".pkl"), "wb") as f:
            pickle.dump(best, f)
        stem.with_suffix(".json").write_text(json.dumps({
            "generation": self.generation, "fitness": best.fitness, "max_x": info["max_x"],
            "cleared": info["cleared"], "buttons": BUTTONS, "actions": actions,
            "path": info["path"]}))
        if best.fitness > self.best_ever:
            self.best_ever = best.fitness
            with open(self.dir / "best_ever.pkl", "wb") as f:
                pickle.dump(best, f)
        secs = time.time() - self.gen_start
        n_species = len(species.species)
        with open(self.csv, "a") as f:
            f.write(f"{self.generation},{best.fitness:.1f},{np.mean(fits):.1f},{n_species},"
                    f"{info['max_x']},{secs:.1f},{int(deterministic)}\n")
        flag = "" if deterministic else "  (replay differs!)"
        print(f"gen {self.generation:4d}  best {best.fitness:8.1f}  mean {np.mean(fits):8.1f}  "
              f"max_x {info['max_x']:5d}  species {n_species:2d}  {secs:5.1f}s{flag}", flush=True)


def train(cfg, config_path, resume=None, run_dir=None, generations=None, workers=None):
    global _ENV_CFG
    _ENV_CFG = cfg.env
    if resume:
        run_dir = Path(resume).parent
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(run_dir or Path(cfg.runs_dir) / f"{cfg.name}-{stamp}")
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, run_dir / "config.toml")
        (run_dir / "neat.cfg").write_text(cfg.neat_text())
    neat_cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                           neat.DefaultStagnation, str(run_dir / "neat.cfg"))
    pop = neat.Checkpointer.restore_checkpoint(resume) if resume else neat.Population(neat_cfg)
    pop.add_reporter(RunReporter(run_dir))
    pop.add_reporter(neat.Checkpointer(cfg.checkpoint_every, filename_prefix=str(run_dir / "ckpt-")))
    n_workers = cfg.workers if workers is None else workers
    if n_workers > 1:
        # the pool must fork before this process creates its own emulator
        evaluate = neat.ParallelEvaluator(n_workers, eval_genome).evaluate
    else:
        def evaluate(genomes, config):
            for _, genome in genomes:
                genome.fitness = eval_genome(genome, config)
    print(f"run dir: {run_dir}  workers: {n_workers}", flush=True)
    try:
        pop.run(evaluate, generations or cfg.generations)
    except KeyboardInterrupt:
        print("\ninterrupted; results so far are in", run_dir)
    return run_dir
