"""NEAT training: parallel episode evaluation, per-generation best-run recording."""
import json
import pickle
import random
import shutil
import time
from pathlib import Path

import neat
import numpy as np

from . import neat_params
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


def eval_genome_full(genome, config):
    """(fitness, emulator frames used) so experiments can compare methods per frame of experience."""
    total, _, info = run_episode(genome, config)
    return total, info["frames"]


class PoolEvaluator:
    """Evaluates a generation on a worker pool (or serially) and counts the emulator frames used."""

    def __init__(self, n_workers, stats):
        import multiprocessing
        self.pool = multiprocessing.Pool(n_workers) if n_workers > 1 else None
        self.stats = stats          # shared with the reporter; a plain dict, so checkpoints can pickle it

    def evaluate(self, genomes, config):
        if self.pool is not None:
            jobs = [self.pool.apply_async(eval_genome_full, (g, config)) for _, g in genomes]
            results = [j.get() for j in jobs]
        else:
            results = [eval_genome_full(g, config) for _, g in genomes]
        for (_, genome), (fitness, _frames) in zip(genomes, results):
            genome.fitness = fitness
        self.stats["frames"] = sum(frames for _, frames in results)

    def close(self):
        if self.pool is not None:
            self.pool.terminate()


class RunReporter(neat.reporting.BaseReporter):
    """Writes metrics.csv and records the best run of every generation."""

    def __init__(self, run_dir, neat_cfg, stats=None, adapt=None):
        self.dir = Path(run_dir)
        self.neat_cfg = neat_cfg
        self.adapt = adapt or {}
        self.adapt_best, self.stall, self.boosted, self.adapt_base = -np.inf, 0, False, {}
        self.stats = stats if stats is not None else {}     # must stay picklable: NEAT checkpoints reporters
        (self.dir / "best").mkdir(parents=True, exist_ok=True)
        self.generation = -1
        self.gen_start = time.time()
        self.best_ever = -np.inf
        self.csv = self.dir / "metrics.csv"
        if not self.csv.exists():
            self.csv.write_text("generation,best,mean,species,max_x,seconds,deterministic,frames,total_frames\n")
        header = self.csv.read_text().splitlines()[0]
        self.with_frames = "total_frames" in header          # runs started before frame counting keep their format
        self.total_frames = 0
        for line in self.csv.read_text().splitlines()[1:]:
            if self.with_frames and line.count(",") >= 8:
                self.total_frames = int(float(line.split(",")[8]))

    def start_generation(self, generation):
        self.generation = generation
        self.gen_start = time.time()

    def _adapt_mutation(self, best_fitness):
        """When the best fitness stops improving, mutate harder until it improves again."""
        a = self.adapt
        if not a.get("enabled"):
            return
        if best_fitness > self.adapt_best + a.get("min_improvement", 0.0) + 1e-9:   # tiny gains are noise
            self.adapt_best, self.stall = best_fitness, 0
        else:
            self.stall += 1
        change, kind = None, None
        if self.stall >= a["stall_generations"] and not self.boosted:
            g = self.neat_cfg.genome_config
            self.adapt_base = {k: getattr(g, k) for k in
                               ("weight_mutate_power", "conn_add_prob", "node_add_prob", "weight_mutate_rate")}
            b = self.adapt_base
            change = {"weight_mutate_power": b["weight_mutate_power"] * a["power_factor"],
                      "conn_add_prob": min(1.0, b["conn_add_prob"] * a["structure_factor"]),
                      "node_add_prob": min(1.0, b["node_add_prob"] * a["structure_factor"]),
                      "weight_mutate_rate": min(1.0, b["weight_mutate_rate"] + a["rate_add"])}
            self.boosted, kind = True, "boost"
        elif self.stall == 0 and self.boosted:
            change, self.boosted, kind = dict(self.adapt_base), False, "restore"
        if change:
            neat_params.apply_neat(self.neat_cfg, change)
            with open(self.dir / "events.jsonl", "a") as f:
                f.write(json.dumps({"generation": self.generation, "adaptive": kind, "changed": change}) + "\n")
            print(f"gen {self.generation}: adaptive mutation {kind} (stalled {self.stall} generations)", flush=True)

    def post_evaluate(self, config, population, species, best_genome):
        fits = [g.fitness for g in population.values()]
        best = max(population.values(), key=lambda g: g.fitness)
        self._adapt_mutation(best.fitness)
        reward, actions, info = run_episode(best, config, record=True)
        deterministic = abs(reward - best.fitness) < 1e-6
        stem = self.dir / "best" / f"gen_{self.generation:04d}"
        with open(stem.with_suffix(".pkl"), "wb") as f:
            pickle.dump(best, f)
        stem.with_suffix(".json").write_text(json.dumps({
            "generation": self.generation, "fitness": best.fitness, "max_x": info["max_x"],
            "cleared": info["cleared"], "buttons": BUTTONS, "actions": actions,
            "path": info["path"], "events": info.get("events", [])}))
        if best.fitness > self.best_ever:
            self.best_ever = best.fitness
            with open(self.dir / "best_ever.pkl", "wb") as f:
                pickle.dump(best, f)
        secs = time.time() - self.gen_start
        n_species = len(species.species)
        frames = self.stats.get("frames", 0)
        self.total_frames += frames
        row = (f"{self.generation},{best.fitness:.1f},{np.mean(fits):.1f},{n_species},"
               f"{info['max_x']},{secs:.1f},{int(deterministic)}")
        if self.with_frames:
            row += f",{frames},{self.total_frames}"
        with open(self.csv, "a") as f:
            f.write(row + "\n")
        flag = "" if deterministic else "  (replay differs!)"
        print(f"gen {self.generation:4d}  best {best.fitness:8.1f}  mean {np.mean(fits):8.1f}  "
              f"max_x {info['max_x']:5d}  species {n_species:2d}  {secs:5.1f}s{flag}", flush=True)


def train(cfg, config_path, resume=None, run_dir=None, generations=None, workers=None):
    global _ENV_CFG
    _ENV_CFG = cfg.env
    if cfg.seed is not None:
        random.seed(cfg.seed)          # neat-python uses the global random module
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
    n_workers = cfg.workers if workers is None else workers
    # the pool must fork before this process creates its own emulator
    stats = {"frames": 0}
    evaluator = PoolEvaluator(n_workers, stats)
    pop.add_reporter(RunReporter(run_dir, pop.config, stats, cfg.adapt))
    pop.add_reporter(neat.Checkpointer(cfg.checkpoint_every, filename_prefix=str(run_dir / "ckpt-")))
    print(f"run dir: {run_dir}  workers: {n_workers}", flush=True)
    try:
        pop.run(evaluator.evaluate, generations or cfg.generations)
    except KeyboardInterrupt:
        print("\ninterrupted; results so far are in", run_dir)
    finally:
        evaluator.close()
    return run_dir
