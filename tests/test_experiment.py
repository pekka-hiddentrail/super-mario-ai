import json

from smwai import experiment


def test_merge_is_recursive_and_does_not_mutate():
    base = {"env": {"inputs": "mixed", "radius": 6}, "neat": {"pop_size": 100}}
    out = experiment.merge(base, {"env": {"inputs": "collect"}})
    assert out["env"] == {"inputs": "collect", "radius": 6} and base["env"]["inputs"] == "mixed"


def test_plan_expands_variants_and_seeds_seed_major(tmp_path):
    spec = tmp_path / "s.toml"
    spec.write_text(
        'name = "t"\nbase = "configs/yi3.toml"\nseeds = [1, 2]\ngenerations = 3\npop_size = 20\n'
        '[[variant]]\nname = "a"\n[[variant]]\nname = "b"\n[variant.env]\ninputs = "collect"\n')
    _, _, runs = experiment.plan(spec, runs_dir=str(tmp_path / "runs"))
    assert [(r["variant"], r["seed"]) for r in runs] == [("a", 1), ("b", 1), ("a", 2), ("b", 2)]
    b = runs[1]["raw"]
    assert b["env"]["inputs"] == "collect" and b["neat"]["pop_size"] == 20
    assert b["train"]["seed"] == 1 and b["train"]["generations"] == 3
    assert runs[0]["raw"]["env"]["inputs"] == "mixed"


def test_summarize_run_finds_first_clear_and_ignores_duplicate_rows(tmp_path):
    run = tmp_path / "r"
    (run / "best").mkdir(parents=True)
    (run / "metrics.csv").write_text(
        "generation,best,mean,species,max_x,seconds,deterministic,frames,total_frames\n"
        "0,100,10,1,120,1,1,1000,1000\n"
        "1,900,50,1,800,1,1,2000,3000\n"
        "1,901,50,1,801,1,1,2000,9999\n"
        "2,5000,90,2,4800,1,1,2500,5500\n")
    for g, cleared in [(0, False), (1, False), (2, True)]:
        (run / "best" / f"gen_{g:04d}.json").write_text(json.dumps(
            {"generation": g, "fitness": 1.0, "max_x": 1, "cleared": cleared, "actions": [0] * 10}))
    s = experiment.summarize_run(run)
    assert s["gens"] == 3 and s["best_x"] == 4800 and s["cleared"]
    assert s["gens_to_clear"] == 3 and s["frames_to_clear"] == 5500 and s["clear_frames"] == 40


def test_format_table_handles_missing_values():
    text = experiment.format_table([{"variant": "v", "seeds": 2, "clears": 0, "best_x": 100.0,
                                     "best_fitness": 90.0, "frames_to_clear": None, "total_frames": 5000.0}])
    assert "| v | 2 | 0/2 | 100 | 90 | - | 5,000 |" in text


def test_runs_are_named_with_the_launch_stamp_and_latest_stamp_finds_it(tmp_path):
    spec = tmp_path / "s.toml"
    spec.write_text('name = "t"\nbase = "configs/yi3.toml"\ngenerations = 1\n[[variant]]\nname = "a"\n')
    runs_dir = str(tmp_path / "runs")
    _, out_dir, runs = experiment.plan(spec, runs_dir=runs_dir, stamp="20261030-1445")
    assert runs[0]["raw"]["name"] == "20261030-1445-t-a-s1"
    assert out_dir.name == "20261030-1445-t"
    out_dir.mkdir(parents=True)
    (out_dir.parent / "20261029-0900-t").mkdir()
    (out_dir.parent / "20261030-1445-tt").mkdir()          # another experiment whose name ends the same: not ours
    assert experiment.latest_stamp("t", runs_dir) == "20261030-1445"
    assert experiment.latest_stamp("nothing", runs_dir) is None
