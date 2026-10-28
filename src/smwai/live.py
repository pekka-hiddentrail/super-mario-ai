"""Live parameter overrides: the dashboard writes overrides.json, the trainer applies it
at the start of each generation and publishes it to the worker processes as active.json."""
import json
import os
from pathlib import Path

ENV_KEYS = {"stuck_frames": int, "max_frames": int}
REWARD_KEYS = {"progress": float, "clear_bonus": float, "time_penalty": float,
               "death_penalty": float, "powerup_bonus": float, "coin_bonus": float,
               "speed_bonus": float, "clear_time_bonus": float, "clear_time_ref": float,
               "yoshi_coin_bonus": float, "block_bonus": float}
# override key -> (section of neat.Config, attribute, type)
NEAT_KEYS = {
    "conn_add_prob": ("genome_config", float), "conn_delete_prob": ("genome_config", float),
    "node_add_prob": ("genome_config", float), "node_delete_prob": ("genome_config", float),
    "weight_mutate_rate": ("genome_config", float), "weight_mutate_power": ("genome_config", float),
    "compatibility_threshold": ("species_set_config", float),
    "max_stagnation": ("stagnation_config", int),
    "survival_threshold": ("reproduction_config", float), "elitism": ("reproduction_config", int),
}


def read_overrides(path):
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json_atomic(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def apply_env(env_cfg, overrides):
    """Update an EnvConfig (and its reward) in place; returns {name: new value} that changed."""
    changed = {}
    for key, cast in ENV_KEYS.items():
        if key in overrides.get("env", {}):
            new = cast(overrides["env"][key])
            if getattr(env_cfg, key) != new:
                setattr(env_cfg, key, new)
                changed[key] = new
    for key, cast in REWARD_KEYS.items():
        if key in overrides.get("reward", {}):
            new = cast(overrides["reward"][key])
            if getattr(env_cfg.reward, key) != new:
                setattr(env_cfg.reward, key, new)
                changed[key] = new
    return changed


def apply_neat(neat_cfg, overrides):
    """Update a neat.Config in place; returns {name: new value} that changed."""
    changed = {}
    for key, (section, cast) in NEAT_KEYS.items():
        if key in overrides.get("neat", {}):
            new = cast(overrides["neat"][key])
            target = getattr(neat_cfg, section)
            if getattr(target, key) != new:
                setattr(target, key, new)
                changed[key] = new
    return changed
