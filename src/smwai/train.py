"""NEAT training: parallel episode evaluation, per-generation best-run recording."""
import json
import pickle
import random
import shutil
import time
from pathlib import Path

import neat
import numpy as np

from . import deathmap, neat_params, novelty
from .env import BUTTONS, SmwEnv
from .novelty import blend

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
    cells = {novelty.cell_id(info["x"], info["y"])}
    while True:
        action = env.decide(net.activate(obs.ravel().astype(np.float64)))
        if record:
            actions.append(_to_mask(action))
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        cells.add(novelty.cell_id(info["x"], info["y"]))
        if record:
            path.append([info["x"], info["y"]])
        if term or trunc:
            info["path"] = path
            info["cells"] = cells
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


def eval_genome_full(genome, config, death_map=None):
    """(fitness, emulator frames used, behaviour) so experiments can compare methods per frame of experience."""
    _get_env().death_map = death_map
    t0 = time.time()
    total, _, info = run_episode(genome, config)
    return total, info["frames"], {"seconds": round(time.time() - t0, 1), "print": novelty.fingerprint(info["cells"], info.get("events", [])),
                                   "end": [info["x"], info["y"], bool(info.get("cleared"))], "takeoff": info.get("takeoff"),
                                   "cells": sorted(info["cells"]) if _get_env().cfg.death_nodes else None}


class PoolEvaluator:
    """Evaluates a generation on a worker pool (or serially) and counts the emulator frames used."""

    def __init__(self, n_workers, stats):
        import multiprocessing
        self.pool = multiprocessing.Pool(n_workers) if n_workers > 1 else None
        self.stats = stats          # shared with the reporter; a plain dict, so checkpoints can pickle it

    def evaluate(self, genomes, config):
        if self.pool is not None:
            jobs = [self.pool.apply_async(eval_genome_full, (g, config, self.stats.get("death_map"))) for _, g in genomes]
            results = [j.get() for j in jobs]
        else:
            results = [eval_genome_full(g, config, self.stats.get("death_map")) for _, g in genomes]
        for (_, genome), (fitness, _frames, _behaviour) in zip(genomes, results):
            genome.fitness = fitness
        self.stats["frames"] = sum(r[1] for r in results)
        self.stats["behaviour"] = {gid: r[2] for (gid, _), r in zip(genomes, results)}

    def close(self):
        if self.pool is not None:
            self.pool.terminate()


class RunReporter(neat.reporting.BaseReporter):
    """Writes metrics.csv and records the best run of every generation."""

    def __init__(self, run_dir, neat_cfg, stats=None, adapt=None, novelty_cfg=None, death_inputs=False, live_cfg=None, death_nodes=False):
        self.dir = Path(run_dir)
        self.neat_cfg = neat_cfg
        self.adapt = adapt or {}
        self.nov = novelty_cfg or {}
        self.live = live_cfg or {}
        self.deaths = deathmap.DeathMap()                  # always kept (takeoffs.jsonl); the networks read it only with death/takeoff inputs
        self.memory_inputs = death_inputs or death_nodes
        self.nodes = deathmap.NodeMemory() if death_nodes else None
        self.archive = novelty.NoveltyArchive(self.nov.get("k", 5), self.nov.get("archive_max", 300), self.nov.get("add_per_gen", 3))
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
        _get_env().death_map = self.stats.get("death_map")          # the memory this generation's genomes had
        reward, actions, info = run_episode(best, config, record=True)
        deterministic = abs(reward - best.fitness) < 1e-6
        stem = self.dir / "best" / f"gen_{self.generation:04d}"
        with open(stem.with_suffix(".pkl"), "wb") as f:
            pickle.dump(best, f)
        stem.with_suffix(".json").write_text(json.dumps({
            "generation": self.generation, "fitness": best.fitness, "max_x": info["max_x"],
            "cleared": info["cleared"], "buttons": BUTTONS, "actions": actions,
            "path": info["path"], "events": info.get("events", []),
            "death_map": self.stats.get("death_map")}))
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
        self._log_slow()
        self._save_population(population, species)
        self._remember_deaths(population)              # before novelty selection turns fitness into a blended score
        extra = self._novelty(population, best)
        print(f"gen {self.generation:4d}  best {best.fitness:8.1f}  mean {np.mean(fits):8.1f}  "
              f"max_x {info['max_x']:5d}  species {n_species:2d}  {secs:5.1f}s{flag}{extra}", flush=True)

    def _save_population(self, population, species):
        """With [live] save_population: the whole generation (genome, species, fitness), for the live view."""
        if not self.live.get("save_population"):
            return
        members = [{"gid": gid, "species": species.get_species_id(gid), "fitness": g.fitness, "genome": g}
                   for gid, g in population.items()]
        tmp = self.dir / "population.tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"generation": self.generation, "members": members}, f)
        if self.live.get("save_history"):
            shutil.copy(tmp, self.dir / f"population_{self.generation:04d}.pkl")
        tmp.replace(self.dir / "population.pkl")

    def _log_slow(self, limit=30.0):
        """Some generations take minutes instead of seconds: note any single episode over `limit` seconds (slow.jsonl)."""
        for gid, b in (self.stats.get("behaviour") or {}).items():
            if b.get("seconds", 0) > limit:
                with open(self.dir / "slow.jsonl", "a") as f:
                    f.write(json.dumps({"generation": self.generation, "genome": gid, "seconds": b["seconds"], "end": b["end"]}) + "\n")
                print(f"   slow episode: genome {gid} took {b['seconds']}s", flush=True)

    def _remember_deaths(self, population):
        """Add where this generation's best runs ended to the death map the next generation's networks can read."""
        behaviour = self.stats.get("behaviour")
        if not behaviour:
            return
        raw = {gid: g.fitness for gid, g in population.items() if gid in behaviour}
        kept = self.deaths.update({gid: behaviour[gid]["end"] for gid in raw}, raw, {gid: behaviour[gid].get("takeoff") for gid in raw})
        if kept:                                       # [x, y, buttons mask, jump hold steps, run, fitness] of the fatal takeoffs
            with open(self.dir / "takeoffs.jsonl", "a") as f:
                f.write(json.dumps({"generation": self.generation, "takeoffs": kept}) + "\n")
        if self.nodes is not None:
            self._update_nodes(raw, behaviour, kept)
        if self.memory_inputs:
            self.stats["death_map"] = {**self.deaths.export(), **({deathmap.NODES: self.nodes.export()} if self.nodes is not None else {})}

    def _update_nodes(self, raw, behaviour, kept):
        """Age the death nodes by what the top attempts did around them, and add this generation's: the best attempt's fatal takeoff."""
        top = sorted(raw, key=raw.get, reverse=True)[: max(1, int(len(raw) * self.deaths.top_share))]
        cells = [[(c // 64, c % 64) for c in (behaviour[gid].get("cells") or ())] for gid in top]

        fatal = [behaviour[gid]["takeoff"] for gid in top if behaviour[gid].get("takeoff")]

        def passed(x, y):                         # a top attempt visited the node's area (same height band) and went on beyond it ...
            if any(deathmap.in_reach(x, y, t[0], t[1]) for t in fatal):
                return False                      # ... but not while attempts still take off from here and die (the node is still needed)
            cx, cy = x // 64, y // 64
            return any(any(a == cx and abs(b - cy) <= 1 for a, b in run) and any(a > cx for a, _ in run) for run in cells)

        self.nodes.update(kept[0] if kept else None, passed)

    def _novelty(self, population, best):
        """Score the generation's behaviour novelty and map coverage; with weight > 0 also turn fitness into the selection score."""
        behaviour = self.stats.get("behaviour")
        if not self.nov.get("enabled") or not behaviour:
            return ""
        prints = {gid: b["print"] for gid, b in behaviour.items() if gid in population}
        scores = self.archive.score(prints)
        raw = {gid: g.fitness for gid, g in population.items() if gid in scores}
        new_cells = self.archive.update(prints, scores)
        best_id = max(raw, key=raw.get)
        row = {"generation": self.generation, "novelty_mean": round(float(np.mean(list(scores.values()))), 3),
               "novelty_max": round(max(scores.values()), 3), "novelty_of_best": round(scores[best_id], 3),
               "coverage": len(self.archive.seen), "new_cells": new_cells, "archive": len(self.archive.prints)}
        with open(self.dir / "novelty.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        weight = self.nov.get("weight", 0.0)
        if weight > 0:                                   # selection now rewards being different as well as getting far
            for gid, score in blend(raw, scores, weight).items():
                population[gid].fitness = score
        return f"  novelty {row['novelty_mean']:.2f}  cells {row['coverage']} (+{new_cells})"


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
    if cfg.hybrid and cfg.hybrid.get("enabled"):
        from .hybrid import HybridEvaluator        # NEAT + PPO: PPO refines the best genomes of each generation
        evaluator = HybridEvaluator(n_workers, stats, cfg.env, cfg.hybrid, run_dir / "hybrid.jsonl")
    else:
        evaluator = PoolEvaluator(n_workers, stats)
    pop.add_reporter(RunReporter(run_dir, pop.config, stats, cfg.adapt, cfg.novelty,
                                  cfg.env.death_inputs or cfg.env.takeoff_inputs, cfg.live, cfg.env.death_nodes))
    pop.add_reporter(neat.Checkpointer(cfg.checkpoint_every, filename_prefix=str(run_dir / "ckpt-")))
    print(f"run dir: {run_dir}  workers: {n_workers}", flush=True)
    try:
        pop.run(evaluator.evaluate, generations or cfg.generations)
    except KeyboardInterrupt:
        print("\ninterrupted; results so far are in", run_dir)
    finally:
        evaluator.close()
    return run_dir
