import json

import neat

from smwai import config
from smwai.viewer.replay import genome_graph
from smwai.viewer.server import list_runs


def _neat_cfg(tmp_path):
    cfg = config.load("configs/level1.toml")
    path = tmp_path / "neat.cfg"
    path.write_text(cfg.neat_text())
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(path))


def test_genome_graph_has_inputs_outputs_and_edges(tmp_path):
    ncfg = _neat_cfg(tmp_path)
    genome = next(iter(neat.Population(ncfg).population.values()))
    graph = genome_graph(genome, ncfg)
    assert len(graph["inputs"]) == 169 and len(graph["outputs"]) == 8
    assert graph["edges"], "initial genome should have some connections"
    assert all(e["to"] in graph["outputs"] or str(e["to"]) in graph["layer"] for e in graph["edges"])


def test_list_runs_reads_best_files(tmp_path):
    best = tmp_path / "run1" / "best"
    best.mkdir(parents=True)
    (best / "gen_0003.json").write_text(json.dumps(
        {"generation": 3, "fitness": 12.5, "max_x": 99, "cleared": False, "actions": [1, 2]}))
    runs = list_runs(tmp_path)
    assert runs == [{"name": "run1", "gens": [{"gen": 3, "fitness": 12.5, "max_x": 99, "cleared": False}]}]


def test_tile_char_classification():
    from smwai.viewer.mapdata import _tile_char
    assert _tile_char(0x25, 0) == "."
    assert _tile_char(0x30, 1) == "#"
    assert _tile_char(0x2B, 0) == "c"
    assert _tile_char(0x30, 0) == "o"


def test_run_path_uses_recorded_path(tmp_path):
    from smwai.viewer.mapdata import run_path
    best = tmp_path / "r" / "best"
    best.mkdir(parents=True)
    (best / "gen_0001.json").write_text(json.dumps(
        {"generation": 1, "fitness": 5.0, "max_x": 9, "cleared": True, "actions": [], "path": [[0, 1], [2, 3]]}))
    assert run_path(tmp_path / "r", 1)["path"] == [[0, 1], [2, 3]]
