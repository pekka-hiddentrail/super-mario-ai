"""Change NEAT settings on a loaded neat.Config in place (used by adaptive mutation)."""

# setting -> (section of neat.Config, type)
NEAT_KEYS = {
    "conn_add_prob": ("genome_config", float), "conn_delete_prob": ("genome_config", float),
    "node_add_prob": ("genome_config", float), "node_delete_prob": ("genome_config", float),
    "weight_mutate_rate": ("genome_config", float), "weight_mutate_power": ("genome_config", float),
    "compatibility_threshold": ("species_set_config", float),
    "max_stagnation": ("stagnation_config", int),
    "survival_threshold": ("reproduction_config", float), "elitism": ("reproduction_config", int),
}


def apply_neat(neat_cfg, values):
    """Set the given NEAT settings; returns {name: new value} for the ones that changed."""
    changed = {}
    for key, (section, cast) in NEAT_KEYS.items():
        if key in values:
            new = cast(values[key])
            target = getattr(neat_cfg, section)
            if getattr(target, key) != new:
                setattr(target, key, new)
                changed[key] = new
    return changed
