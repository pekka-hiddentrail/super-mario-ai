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
