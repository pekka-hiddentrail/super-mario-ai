import pytest

from smwai import curriculum, train

SPEC = "configs/experiments/curriculum_yi.toml"


def test_plan_is_one_run_per_seed_and_level():
    spec, out_dir, plan = curriculum.level_runs(SPEC, "runs", stamp="20261031-1200")
    assert [(r["seed"], r["n"], r["level"]) for r in plan[:2]] == [(1, 1, "YoshiIsland1"), (1, 2, "YoshiIsland2")]
    assert len(plan) == len(spec["seeds"]) * len(spec["levels"])
    first = plan[0]
    assert first["run_dir"].name == "20261031-1200-curriculum-yi-s1-L1-YoshiIsland1"
    assert first["raw"]["env"]["levels"] == ["YoshiIsland1"]
    assert first["raw"]["env"]["inputs"] == "physics"
    assert first["raw"]["neat"]["num_hidden"] == 16 and first["raw"]["neat"]["pop_size"] == 15
    assert first["raw"]["train"]["generations"] == 15
    assert plan[1]["raw"]["env"]["levels"] == ["YoshiIsland2"]


def test_carry_file_is_the_newest_population(tmp_path):
    assert curriculum.carry_file(tmp_path) is None
    for n in (4, 12, 9):
        (tmp_path / f"carry-{n}").write_text("x")
    assert curriculum.carry_file(tmp_path).name == "carry-12"


class _Reporter:
    cleared = False


def test_stop_on_clear_saves_and_stops_only_when_cleared(tmp_path):
    reporter = _Reporter()
    stop = train.StopOnClear(tmp_path, reporter)
    saved = []
    stop.saver.save_checkpoint = lambda config, population, species, generation: saved.append(generation)
    stop.start_generation(7)
    stop.end_generation(None, {}, None)                  # not cleared: the level goes on
    assert saved == []
    reporter.cleared = True
    with pytest.raises(train.LevelCleared):
        stop.end_generation(None, {}, None)
    assert saved == [8]                                  # the saved population is the one that starts generation 8
