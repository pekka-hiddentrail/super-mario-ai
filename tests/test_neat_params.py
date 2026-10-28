import neat

from smwai import config, neat_params


def _neat_cfg(tmp_path):
    cfg = config.load("configs/level1.toml")
    path = tmp_path / "neat.cfg"
    path.write_text(cfg.neat_text())
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(path))


def test_apply_neat_changes_all_sections(tmp_path):
    ncfg = _neat_cfg(tmp_path)
    changed = neat_params.apply_neat(ncfg, {"conn_add_prob": 0.9, "compatibility_threshold": 1.5,
                                            "max_stagnation": 7, "survival_threshold": 0.4})
    assert ncfg.genome_config.conn_add_prob == 0.9
    assert ncfg.species_set_config.compatibility_threshold == 1.5
    assert ncfg.stagnation_config.max_stagnation == 7
    assert ncfg.reproduction_config.survival_threshold == 0.4
    assert len(changed) == 4
    assert neat_params.apply_neat(ncfg, {"conn_add_prob": 0.9}) == {}      # unchanged values are not reported
