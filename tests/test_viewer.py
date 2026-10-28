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


def test_list_videos_and_best_generation(tmp_path):
    from smwai.video import best_generation
    from smwai.viewer.server import list_videos
    run = tmp_path / "r1"
    (run / "best").mkdir(parents=True)
    (run / "videos").mkdir()
    for g, fit in [(1, 10.0), (2, 99.0), (3, 50.0)]:
        (run / "best" / f"gen_{g:04d}.json").write_text(json.dumps(
            {"generation": g, "fitness": fit, "max_x": g * 100, "cleared": g == 2, "actions": []}))
    assert best_generation(run)["generation"] == 2
    (run / "videos" / "best_gen_0002.mp4").write_bytes(b"x" * 10)
    vids = list_videos(tmp_path)
    assert vids[0]["generation"] == 2 and vids[0]["fitness"] == 99.0 and vids[0]["cleared"] is True


def test_run_reporter_is_picklable(tmp_path):
    """NEAT checkpoints pickle the reporters; a worker pool inside one crashed training at generation 5."""
    import pickle

    from smwai.train import RunReporter
    reporter = RunReporter(tmp_path / "run", None, {"frames": 123})
    assert pickle.loads(pickle.dumps(reporter)).stats["frames"] == 123


def test_tile_char_marks_big_coins_and_boxes():
    from smwai.viewer.mapdata import _tile_char
    assert _tile_char(0x2D, 0) == "Y" and _tile_char(0x2E, 0) == "Y"
    assert _tile_char(0x24, 1) == "?" and _tile_char(0x26, 1) == "?"
    assert _tile_char(0x32, 1) == "#"          # used block stays solid


def test_run_path_returns_recorded_events(tmp_path):
    from smwai.viewer.mapdata import run_path
    best = tmp_path / "r" / "best"
    best.mkdir(parents=True)
    ev = [[10, "coin", 100, 300], [20, "block", 120, 290]]
    (best / "gen_0001.json").write_text(json.dumps(
        {"generation": 1, "fitness": 5.0, "max_x": 9, "cleared": True, "actions": [], "path": [[0, 1]], "events": ev}))
    assert "events" not in run_path(tmp_path / "r", 1)
    assert run_path(tmp_path / "r", 1, events=True)["events"] == ev
