import neat

from smwai import config, live
from smwai.env import EnvConfig


def _neat_cfg(tmp_path):
    cfg = config.load("configs/level1.toml")
    path = tmp_path / "neat.cfg"
    path.write_text(cfg.neat_text())
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(path))


def test_apply_env_changes_in_place():
    cfg = EnvConfig()
    changed = live.apply_env(cfg, {"env": {"stuck_frames": 200}, "reward": {"time_penalty": 0.5}})
    assert cfg.stuck_frames == 200 and cfg.reward.time_penalty == 0.5
    assert changed == {"stuck_frames": 200, "time_penalty": 0.5}
    assert live.apply_env(cfg, {"env": {"stuck_frames": 200}}) == {}


def test_apply_neat_changes_all_sections(tmp_path):
    ncfg = _neat_cfg(tmp_path)
    ov = {"neat": {"conn_add_prob": 0.9, "compatibility_threshold": 1.5, "max_stagnation": 7,
                   "survival_threshold": 0.4}}
    changed = live.apply_neat(ncfg, ov)
    assert ncfg.genome_config.conn_add_prob == 0.9
    assert ncfg.species_set_config.compatibility_threshold == 1.5
    assert ncfg.stagnation_config.max_stagnation == 7
    assert ncfg.reproduction_config.survival_threshold == 0.4
    assert len(changed) == 4


def test_read_write_overrides(tmp_path):
    f = tmp_path / "overrides.json"
    assert live.read_overrides(f) == {}
    live.write_json_atomic(f, {"reward": {"progress": 2.0}})
    assert live.read_overrides(f) == {"reward": {"progress": 2.0}}
    f.write_text("{not json")
    assert live.read_overrides(f) == {}
