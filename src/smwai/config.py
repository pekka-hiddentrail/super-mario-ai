"""Load a run config (TOML) into env dataclasses and generate the neat-python config text."""
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .env import BUTTONS, EnvConfig, RewardConfig, n_inputs

NEAT_TEMPLATE = """\
[NEAT]
fitness_criterion = max
fitness_threshold = 1e12
pop_size = {pop_size}
reset_on_extinction = True
no_fitness_termination = True

[DefaultGenome]
activation_default = sigmoid
activation_mutate_rate = 0.0
activation_options = sigmoid
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob = {conn_add_prob}
conn_delete_prob = {conn_delete_prob}
enabled_default = True
enabled_mutate_rate = 0.01
feed_forward = True
initial_connection = {initial_connection}
node_add_prob = {node_add_prob}
node_delete_prob = {node_delete_prob}
num_hidden = 0
num_inputs = {n_inputs}
num_outputs = {n_outputs}
response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 30
weight_min_value = -30
weight_mutate_power = {weight_mutate_power}
weight_mutate_rate = {weight_mutate_rate}
weight_replace_rate = 0.1

[DefaultSpeciesSet]
compatibility_threshold = {compatibility_threshold}

[DefaultStagnation]
species_fitness_func = max
max_stagnation = {max_stagnation}
species_elitism = 2

[DefaultReproduction]
elitism = {elitism}
survival_threshold = {survival_threshold}
min_species_size = {min_species_size}
"""

NEAT_DEFAULTS = dict(
    pop_size=100, initial_connection="partial_direct 0.05", conn_add_prob=0.5,
    conn_delete_prob=0.2, node_add_prob=0.2, node_delete_prob=0.1, weight_mutate_rate=0.8,
    weight_mutate_power=0.5, compatibility_threshold=3.0, max_stagnation=15,
    survival_threshold=0.2, elitism=2, min_species_size=2)


ADAPT_DEFAULTS = dict(
    enabled=False,          # adaptive mutation: mutate harder while the best fitness is stalled
    stall_generations=5,    # generations without a meaningful new best before boosting
    min_improvement=20.0,   # a new best counts only if it beats the last one by this much (fitness units, ~pixels)
    power_factor=2.0,       # weight_mutate_power multiplier while boosted
    structure_factor=1.5,   # conn_add_prob / node_add_prob multiplier while boosted
    rate_add=0.1)           # added to weight_mutate_rate while boosted (capped at 1)

HYBRID_DEFAULTS = dict(
    enabled=False,          # NEAT + PPO as one learner: PPO refines the best genomes of each generation (hybrid.py)
    elites=2,               # how many of the best genomes are refined per generation
    start_generation=2,     # first generation with refinement (NEAT alone finds a first working behaviour)
    every=1,                # refine every N generations (trigger = schedule)
    trigger="schedule",     # "schedule": every N generations; "stall": only while NEAT's best has stopped improving
    stall_generations=3,    # (trigger = stall) generations without a real gain before PPO is used
    min_improvement=20.0,   # (trigger = stall) what counts as a real gain in fitness
    n_envs=8,               # parallel emulators for the PPO rollouts
    steps=128,              # steps per emulator per PPO iteration
    iters=3,                # PPO iterations per refined genome (iters * steps * n_envs steps of fresh experience)
    epochs=4, minibatch=256, clip=0.2, ent_coef=0.003, gamma=0.99, lam=0.95,
    lr=0.003,               # learning rate on the genome's weights and biases
    critic_lr=0.001, explore_eps=0.05, reward_scale=0.01)

NOVELTY_DEFAULTS = dict(
    enabled=True,           # measure behaviour novelty and map coverage each generation (novelty.jsonl); costs next to nothing
    weight=0.0,             # share of novelty in the selection score (0: record only; 0.3: 70% fitness, 30% novelty)
    k=5, archive_max=300, add_per_gen=3)

PPO_DEFAULTS = dict(
    total_steps=500_000, n_envs=8, n_steps=256, batch_size=512, n_epochs=4, learning_rate=3e-4,
    gamma=0.99, ent_coef=0.01, clip_range=0.2, reward_scale=0.01, eval_every=20_480,
    net_arch=[256, 256])


@dataclass
class RunConfig:
    name: str
    env: EnvConfig
    generations: int
    workers: int
    checkpoint_every: int
    runs_dir: str
    neat: dict
    seed: int = None
    ppo: dict = None
    adapt: dict = None
    hybrid: dict = None
    novelty: dict = None

    @property
    def n_inputs(self):
        return n_inputs(self.env)

    def neat_text(self):
        return NEAT_TEMPLATE.format(n_inputs=self.n_inputs, n_outputs=len(BUTTONS), **self.neat)


def load_raw(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return repr(v)


def dump(raw):
    """Write a dict of scalar/list values in sections to TOML text (comments are not kept)."""
    out = [f"{k} = {_toml_value(v)}" for k, v in raw.items() if not isinstance(v, dict)]
    for section, body in raw.items():
        if isinstance(body, dict):
            out += ["", f"[{section}]"] + [f"{k} = {_toml_value(v)}" for k, v in body.items()]
    return "\n".join(out) + "\n"


def load(path):
    raw = load_raw(path)
    env = EnvConfig(**raw.get("env", {}), reward=RewardConfig(**raw.get("reward", {})))
    train = raw.get("train", {})
    return RunConfig(
        name=raw.get("name", "run"), env=env,
        generations=train.get("generations", 100), workers=train.get("workers", 4),
        checkpoint_every=train.get("checkpoint_every", 5), runs_dir=train.get("runs_dir", "runs"),
        neat={**NEAT_DEFAULTS, **raw.get("neat", {})}, seed=train.get("seed"),
        ppo={**PPO_DEFAULTS, **raw.get("ppo", {})}, adapt={**ADAPT_DEFAULTS, **raw.get("adapt", {})},
        hybrid={**HYBRID_DEFAULTS, **raw.get("hybrid", {})}, novelty={**NOVELTY_DEFAULTS, **raw.get("novelty", {})})
