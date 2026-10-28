import configparser
from pathlib import Path

from smwai import config

CFG = Path(__file__).parent.parent / "configs" / "level1.toml"


def test_load_level1():
    cfg = config.load(CFG)
    assert cfg.env.levels == ["YoshiIsland2"]
    assert cfg.n_inputs == 169
    assert cfg.neat["pop_size"] == 100


def test_neat_text_is_valid_ini():
    cfg = config.load(CFG)
    parser = configparser.ConfigParser()
    parser.read_string(cfg.neat_text())
    assert parser["DefaultGenome"]["num_inputs"] == "169"
    assert parser["DefaultGenome"]["num_outputs"] == "8"


def test_dump_roundtrip(tmp_path):
    raw = config.load_raw(CFG)
    raw["reward"]["time_penalty"] = 0.25
    out = tmp_path / "c.toml"
    out.write_text(config.dump(raw))
    again = config.load_raw(out)
    assert again == raw
    assert config.load(out).env.reward.time_penalty == 0.25


def test_input_counts_by_mode():
    from smwai.env import EnvConfig, n_inputs
    assert n_inputs(EnvConfig()) == 169
    assert n_inputs(EnvConfig(inputs="channels")) == 169 * 3 + 3


def test_collect_mode_has_coin_grid():
    from smwai.env import EnvConfig, input_layout, n_inputs
    cfg = EnvConfig(inputs="collect")
    assert [n for n, _ in input_layout(cfg)][:4] == ["solid", "enemy", "item", "coin"]
    assert n_inputs(cfg) == 169 * 4 + 3
