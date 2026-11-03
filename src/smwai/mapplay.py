"""Mario plays the whole game: the levels and the map between them, on one emulator, in one video.

The game starts inside its first level (Yoshi's Island 2, the game's Start state). A level is played by what Mario has learnt of it: a trained genome from a finished run
(configs/map_levels.toml names the run of every level there is one for); a level nobody trained is played by random buttons for a while. When the level is cleared (or Mario
dies with lives left) the game returns to the map, and the map policy picks a direction and a number of steps, 1 to 3, then presses A. A level can be played again and again.
The map policy sees the game (Mario's map position, what he has just done); the first one, RandomMapPolicy, only throws dice. The run ends when the visits are used,
the lives are gone or Mario is stuck in a level he cannot leave.
"""
import json
import pickle
import random
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import neat
import numpy as np
from PIL import Image

from . import chain, config, video
from .curriculum import carry_file
from .experiment import merge
from .env import SmwEnv, view as env_view
from .overworld import DIRECTIONS, MODE_LEVEL, Game
from .viewer.replay import genome_graph
from . import ram as R

DEFAULT_LEVELS = "configs/map_levels.toml"
MAX_ATTEMPTS = 10      # a death loads the save made when the level began; after this many deaths in a row the level really kills Mario


class RandomMapPolicy:
    """A random direction and one to three steps in it."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def choose(self, game):
        return self.rng.choice(DIRECTIONS), self.rng.choice((1, 2, 3))


class _Forward:
    """What SmwEnv.capture needs (append): every emulator frame the level player steps goes to the game's video."""

    def __init__(self, game):
        self.game = game

    def append(self, frame):
        self.game.frames += 1
        if self.game.sink:
            self.game.sink(frame, self.game.caption)


class LevelPlayer:
    """The first generation of a finished run whose best run cleared the level: its genome plays, with the observation layout of that run."""

    def __init__(self, run_dir, generation=None, genome_path=None):
        run = Path(run_dir)
        self.cfg = config.load(run / "config.toml")
        self.neat = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(run / "neat.cfg"))
        cleared = [int(p.stem.split("_")[1]) for p in sorted((run / "best").glob("gen_*.json")) if json.loads(p.read_text()).get("cleared")]
        if generation is None and not cleared:
            raise SystemExit(f"{run}: no generation cleared the level")
        self.generation = generation if generation is not None else min(cleared)
        self.fitness = None
        if genome_path:                                            # a stretch's genome kept beside its checkpoint save
            with open(genome_path, "rb") as f:
                self.genome = pickle.load(f)
            self.run = run
            return
        if genome_path is None:
            self.fitness = json.loads((run / "best" / f"gen_{self.generation:04d}.json").read_text())["fitness"]
        with open(run / "best" / f"gen_{self.generation:04d}.pkl", "rb") as f:
            self.genome = pickle.load(f)
        self.run = run

    def play(self, game, name, attempt=1, budget=20000, label=None, frames=None):
        """Returns 'cleared', 'died' or 'stuck' (the genome can do nothing more and the level cannot be left), or 'on' when `frames` emulator frames have been played."""
        env = SmwEnv(self.cfg.env)
        env.capture = _Forward(game)
        obs, info = env.attach(game.env, game.obs)
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.neat)
        outcome, total = "stuck", 0.0
        graph = genome_graph(self.genome, self.neat)                 # the network picture is the one of the normal videos (video.NetworkPanel)
        outputs = set(graph["outputs"])
        graph["hidden"] = sorted(k for k in self.genome.nodes if k not in outputs)
        graph["layer"].update({str(k): 1 for k in graph["hidden"] if str(k) not in graph["layer"]})
        graph["bias"].update({str(k): round(self.genome.nodes[k].bias, 3) for k in graph["hidden"]})
        watch = graph["hidden"] + graph["outputs"]
        panel = video.NetworkPanel(graph, [{"name": nm, "kind": k} for nm, k in env.layout], env_view(self.cfg.env), 224 * video.SCALE + video.BANNER)
        head = label or f"{name}   {self.run.name}   generation {self.generation}   genome {self.genome.key}   attempt {attempt}"
        end = game.frames + frames if frames else None
        for _ in range(budget // self.cfg.env.frame_skip):
            out = net.activate(obs.ravel().astype(np.float64))
            action = env.decide(out)
            game.panel = panel.draw(obs.ravel().copy(), {k: float(net.values.get(k, 0.0)) for k in watch}, action)
            game.caption = (head, f"genome {self.genome.key}   " + (f"fitness {self.fitness:.0f}   " if self.fitness is not None else "") + f"x {info['x']}   reward {total:.0f}")
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            if end is not None and game.frames >= end:
                outcome = "on"
                break
            if term:
                outcome = "cleared" if info["cleared"] else "died"
                break
            if trunc and env.stuck >= self.cfg.env.stuck_frames:
                break
        game.obs = env.frame
        game.panel = None
        return outcome


LEARN_BASE = "configs/yi1_adaptive.toml"
LEARN_SETTINGS = {"env": {"inputs": "physics", "checkpoint_every": 300, "checkpoint_patience": 10}, "reward": {"opportunity_bonus": 2.0},
                  "neat": {"num_hidden": 8, "start_fraction": 0.025, "conn_delete_prob": 0.5, "compatibility_threshold": 4.5}, "adapt": {"nuke_after": 10}}


class Learner:
    """One population that goes through the whole game and keeps learning in every level, from the save made when the level began.

    Each visit to a level is a round of training on that save (the normal training: checkpoints, the nuke, the opportunity bonus), starting from the population the
    last level left (carry-N), ending with the first generation whose best run clears the level. The best run of the best generation, the chain of checkpoint runs
    leading to it, is then played in the real game from the same save: the earlier stretches as the recorded input, the last one by its genome with its network.
    A round that does not clear the level is followed by another one (the population goes on) until `rounds` are used."""

    def __init__(self, out_dir, seed, generations=40, pop=40, rounds=3):
        self.out, self.seed, self.generations, self.pop, self.rounds = Path(out_dir), seed, generations, pop, rounds
        self.carry = None

    def _train(self, start_state, run_dir, round_no, label):
        raw = merge(config.load_raw(LEARN_BASE), LEARN_SETTINGS)
        raw["neat"]["pop_size"] = self.pop
        raw["env"]["levels"] = [str(Path(start_state).resolve())]
        raw["name"] = label
        raw.setdefault("train", {}).update(seed=self.seed + round_no, generations=self.generations, workers=12, runs_dir=str(Path(run_dir).parent))
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir.parent / f"{run_dir.name}.toml"
        cfg_path.write_text(config.dump(raw))
        cmd = [sys.executable, "-W", "ignore", "-m", "smwai.cli", "train", "--config", str(cfg_path), "--run-dir", str(run_dir), "--stop-on-clear", "--carry-at-end"]
        if self.carry:
            cmd += ["--resume", str(self.carry)]
        # a fresh process: one emulator per process, and the worker pool forks before an emulator exists
        subprocess.run(cmd, stdout=(run_dir.parent / f"{run_dir.name}.log").open("w"), stderr=subprocess.STDOUT, check=False)
        self.carry = carry_file(run_dir) or self.carry

    def visit(self, game, name, index):
        """Learn the level and play it. Returns ('cleared' | 'not cleared', rounds used)."""
        lv = self.out / "levels" / f"{index:02d}-{name}"
        lv.mkdir(parents=True, exist_ok=True)
        start = lv / "start.state"
        start.write_bytes(game.level_save)
        for round_no in range(1, self.rounds + 1):
            run = lv / f"round{round_no}"
            game.caption = f"{name}: learning, round {round_no} of {self.rounds} ({self.generations} generations of {self.pop} at most)"
            self._train(start, run, round_no, f"{self.out.name}-L{index:02d}-{name}-r{round_no}")
            best = video.best_generation(run)
            if best is None:
                return "not cleared", round_no
            outcome = self._play(game, name, run, best["generation"], round_no)
            print(f"  {name}: round {round_no}, generation {best['generation']}: {outcome}", flush=True)
            if outcome == "cleared":
                return "cleared", round_no
            game.retry()                                       # the level begins again, the population goes on learning
        return "not cleared", self.rounds

    def _play(self, game, name, run, generation, round_no):
        """The chain of generation `generation` in the real game, from the level's save."""
        stretches = chain.stretches_of(run, generation)
        game.env.unwrapped.em.set_state(game.level_save)
        for n, (trace, genome_file) in enumerate(stretches, 1):
            last = n == len(stretches)
            label = f"{name}   round {round_no}   generation {generation}   stretch {n}/{len(stretches)}"
            for mask in trace[:2]:                              # the two idle frames every episode starts with
                game.frame_mask(mask)
            if genome_file is None:                             # an older run: no genome kept for this stretch, play its recorded input
                game.caption = (label, "the recorded input of its run")
                for mask in trace[2:]:
                    game.frame_mask(mask)
                continue
            player = LevelPlayer(run, generation, None if last else genome_file)
            outcome = player.play(game, name, 1, label=label, frames=None if last else len(trace) - 2)
            if last or outcome != "on":
                return outcome
        return "stuck"


def play_random(game, name, rng, attempt=1, budget=3000):
    """A level nobody trained: run right, jump now and then."""
    for i in range(budget):
        game.caption = f"{name}: no trained genome, random buttons   attempt {attempt}"
        if R.u8(game.wram(), R.MESSAGE) != 0:                 # a message box freezes the game until A is pressed anew (SmwEnv does this for the trained genomes)
            game.frame(*(("A",) if i % 2 == 0 else ()))
            continue
        pressed = ["RIGHT", "Y"] + (["B"] if (i // 24) % 2 == 0 and rng.random() < 0.9 else [])
        game.frame(*pressed)
        if game.u8(0x1493) != 0:
            return "cleared"
        if game.mode != MODE_LEVEL:
            return "died"
    return "stuck"


def _writer(out_dir):
    path = Path(out_dir) / "videos" / "map_run.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return video._video_writer(path), path


def run_map(seed=1, visits=6, levels_file=DEFAULT_LEVELS, runs_dir="runs", state="Start", verbose=True, learn=False, generations=40, pop=40, rounds=3):
    """Play: the start level, then `visits` times a choice on the map. Returns the log (also written to <run>/map_run.json with the video)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    mode = f"learn-s{seed}-v{visits}-{pop}x{generations}r{rounds}" if learn else f"genomes-s{seed}-v{visits}"     # e.g. 20261103-143712-map-learn-s1-v6-40x40r3
    out = Path(runs_dir) / f"{stamp}-map-{mode}"
    out.mkdir(parents=True, exist_ok=True)
    writer, video_path = _writer(out)
    registry = tomllib.loads(Path(levels_file).read_text(encoding="utf-8")).get("levels", {}) if Path(levels_file).exists() else {}
    players, rng, current = {}, random.Random(seed), [None]
    learner = Learner(out, seed, generations, pop, rounds) if learn else None
    levels_played = [0]
    policy = RandomMapPolicy(seed)
    height = 224 * video.SCALE + video.BANNER
    head = f"{out.name}   the game from its start"

    def sink(frame, caption):
        lines = list(caption) if isinstance(caption, tuple) else [head, caption]
        game_img = Image.fromarray(frame).resize((256 * video.SCALE, 224 * video.SCALE), Image.NEAREST)
        left = Image.new("RGB", (game_img.width, height))
        left.paste(game_img, (0, video.BANNER))
        video._caption(left, lines, None)
        canvas = Image.new("RGB", (game_img.width + video.PANEL_W, height), video.BG)      # always as wide as with the network, so the video keeps one size
        canvas.paste(left, (0, 0))
        if game.panel is not None:
            canvas.paste(game.panel, (game_img.width, 0))
        writer.append_data(np.asarray(canvas))

    log = []
    game = Game(state, sink)

    def play_level(name):
        if learner:
            levels_played[0] += 1
            outcome, used = learner.visit(game, name, levels_played[0])
            outcome = "cleared" if outcome == "cleared" else "stuck"
            log.append({"level": name, "outcome": outcome, "rounds": used, "frame": game.frames, "learned": True})
            if verbose:
                print(f"  {name}: {outcome} after {used} round(s) of learning (frame {game.frames})", flush=True)
            return outcome
        attempt = 1
        while True:
            if name in registry and name not in players:
                players[name] = LevelPlayer(registry[name])
            if name in registry:
                current[0] = players[name]
                outcome = players[name].play(game, name, attempt)
            elif current[0] is not None:                       # a level without a genome of its own: the network Mario has now plays it
                p = current[0]
                outcome = p.play(game, name, attempt, label=f"{name}   (no genome of its own) {p.run.name[-26:]}   generation {p.generation}   genome {p.genome.key}   attempt {attempt}")
            else:
                outcome = play_random(game, name, rng, attempt)
            if outcome == "died" and attempt < MAX_ATTEMPTS:
                attempt += 1
                game.retry()                                  # no life lost, no trip to the map: the level starts again
                continue
            break
        entry = {"level": name, "outcome": outcome, "attempts": attempt, "frame": game.frames, "trained": name in registry}
        log.append(entry)
        if verbose:
            print(f"  {name}: {outcome} after {attempt} attempt(s) (frame {game.frames})", flush=True)
        return outcome

    try:
        if play_level(game.level_name) in ("stuck",) or not game.wait_map():
            return _finish(out, log, game, writer, video_path, "ended in the first level")
        for visit in range(1, visits + 1):
            direction, steps = policy.choose(game)
            game.caption = f"map: {direction} x {steps}"
            moved = [game.walk(direction) for _ in range(steps)]
            name = game.enter()
            entry = {"visit": visit, "direction": direction, "steps": steps, "moved": moved, "position": list(game.position)}
            log.append(entry)
            if verbose:
                print(f"visit {visit}: {direction} x {steps}, moved {moved}, at {game.position}, " + (f"entered {name}" if name else "no level here"), flush=True)
            if not name:
                continue
            outcome = play_level(name)
            if outcome == "stuck":
                return _finish(out, log, game, writer, video_path, f"stuck in {name}")
            if not game.wait_map():
                return _finish(out, log, game, writer, video_path, "game over")
        return _finish(out, log, game, writer, video_path, "visits used")
    finally:
        try:
            writer.close()
        except Exception:
            pass


def _finish(out, log, game, writer, video_path, reason):
    summary = {"reason": reason, "frames": game.frames, "lives": game.lives, "log": log}
    (out / "map_run.json").write_text(json.dumps(summary, indent=1))
    print(f"map run ended: {reason}; {game.frames} frames; video {video_path}", flush=True)
    return summary
