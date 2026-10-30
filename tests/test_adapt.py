import neat

from smwai import config
from smwai.train import RunReporter


def _neat_cfg(tmp_path):
    cfg = config.load("configs/level1.toml")
    path = tmp_path / "neat.cfg"
    path.write_text(cfg.neat_text())
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(path))


def test_adapt_defaults_are_off():
    cfg = config.load("configs/level1.toml")
    assert cfg.adapt["enabled"] is False and cfg.adapt["stall_generations"] == 5


def test_adaptive_mutation_boosts_when_stalled_and_restores_on_improvement(tmp_path):
    ncfg = _neat_cfg(tmp_path)
    adapt = {**config.ADAPT_DEFAULTS, "enabled": True, "stall_generations": 3}
    rep = RunReporter(tmp_path / "run", ncfg, {"frames": 0}, adapt)
    g = ncfg.genome_config
    base_power, base_add = g.weight_mutate_power, g.conn_add_prob
    rep.generation = 0
    rep._adapt_mutation(100.0)                    # new best: no stall
    for gen in (1, 2):
        rep.generation = gen
        rep._adapt_mutation(100.0)                # stalled 1, 2: still normal
    assert g.weight_mutate_power == base_power
    rep.generation = 3
    rep._adapt_mutation(100.0)                    # stalled 3: boost
    assert rep.boosted
    assert g.weight_mutate_power == base_power * adapt["power_factor"]
    assert g.conn_add_prob == min(1.0, base_add * adapt["structure_factor"])
    rep.generation = 4
    rep._adapt_mutation(150.0)                    # improved: back to normal
    assert not rep.boosted
    assert g.weight_mutate_power == base_power and g.conn_add_prob == base_add
    log = (tmp_path / "run" / "events.jsonl").read_text().splitlines()
    assert [__import__("json").loads(line)["adaptive"] for line in log] == ["boost", "restore"]


def test_adaptive_mutation_disabled_changes_nothing(tmp_path):
    ncfg = _neat_cfg(tmp_path)
    rep = RunReporter(tmp_path / "run", ncfg, {"frames": 0}, {**config.ADAPT_DEFAULTS})
    before = ncfg.genome_config.weight_mutate_power
    for gen in range(10):
        rep.generation = gen
        rep._adapt_mutation(1.0)
    assert ncfg.genome_config.weight_mutate_power == before and not rep.boosted


def test_tiny_improvements_do_not_reset_the_stall_counter(tmp_path):
    ncfg = _neat_cfg(tmp_path)
    adapt = {**config.ADAPT_DEFAULTS, "enabled": True, "stall_generations": 3, "min_improvement": 20.0}
    rep = RunReporter(tmp_path / "run", ncfg, {"frames": 0}, adapt)
    for gen, fit in enumerate([100.0, 101.0, 103.0, 104.0]):    # +1..+3 per generation: noise
        rep.generation = gen
        rep._adapt_mutation(fit)
    assert rep.boosted, "creeping gains of a few pixels must not hide a plateau"
    rep.generation = 4
    rep._adapt_mutation(130.0)                                   # +26 over the reference: a real gain
    assert not rep.boosted


def test_nuke_rewrites_most_new_children_and_keeps_the_elites(tmp_path):
    import random
    ncfg = _neat_cfg(tmp_path)
    random.seed(3)
    pop = neat.Population(ncfg)
    for i, g in enumerate(pop.population.values()):
        g.fitness = float(i)                                      # a ranking, so there is a best genome
    pop.species.speciate(ncfg, pop.population, 0)
    best = max(pop.population.values(), key=lambda g: g.fitness)
    adapt = {**config.ADAPT_DEFAULTS, "enabled": True, "stall_generations": 3, "nuke_after": 4}
    rep = RunReporter(tmp_path / "run", ncfg, {"frames": 0}, adapt)
    rep.generation = 5
    rep.stall = 3
    rep._plan_nuke(best, pop.species, pop.population)
    assert rep.nuke is None                                       # not stalled long enough
    rep.stall = 4
    rep._plan_nuke(best, pop.species, pop.population)
    assert rep.nuke is not None and rep.nuke["a"].key == best.key
    old_keys = set(pop.population)
    children = pop.reproduction.reproduce(ncfg, pop.species, ncfg.pop_size, 5)
    pop.species.speciate(ncfg, children, 6)
    before = {k: (len(g.connections), sorted(c.weight for c in g.connections.values())[:3]) for k, g in children.items()}
    rep.end_generation(ncfg, children, pop.species)
    new_keys = [k for k in children if k not in old_keys]
    changed = [k for k in new_keys if (len(children[k].connections), sorted(c.weight for c in children[k].connections.values())[:3]) != before[k]]
    assert len(changed) >= int(0.7 * len(new_keys)) - 1             # about 70% of the children were rewritten ...
    assert all(children[k].fitness is None for k in changed)
    for k in old_keys & set(children):                               # ... the elites are untouched
        assert children[k].fitness is not None
    for g in children.values():                                      # every genome still makes a working network
        neat.nn.FeedForwardNetwork.create(g, ncfg)
    assert rep.nuke is None and rep.nuke_cool == adapt["nuke_cooldown"]
    events = [__import__("json").loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert events[-1]["nuke"]["second_parent"] in ("species", "fresh")
    assert events[-1]["nuke"]["replaced"] == int(0.7 * len(new_keys)) and events[-1]["nuke"]["children"] == len(new_keys)
    rep.stall = 9
    rep._plan_nuke(best, pop.species, children)
    assert rep.nuke is None                                          # cooling down: no second nuke right away


def test_nuke_is_off_by_default():
    assert config.ADAPT_DEFAULTS["nuke_after"] == 0
