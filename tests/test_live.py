import os
import time

from smwai.live import latest_run


def _gen(run, n, mtime, neat_cfg=True):
    best = run / "best"
    best.mkdir(parents=True, exist_ok=True)
    if neat_cfg:
        (run / "neat.cfg").write_text("x")
    f = best / f"gen_{n:04d}.pkl"
    f.write_text("x")
    os.utime(f, (mtime, mtime))


def test_latest_run_follows_the_newest_generation(tmp_path):
    now = time.time()
    _gen(tmp_path / "a", 3, now - 100)
    _gen(tmp_path / "b", 1, now - 50)
    _gen(tmp_path / "b", 2, now - 10)
    _gen(tmp_path / "ppo", 9, now, neat_cfg=False)          # runs without a NEAT network are skipped
    run, gen = latest_run(tmp_path)
    assert (run.name, gen) == ("b", 2)
    assert latest_run(tmp_path, only="a")[1] == 3
    assert latest_run(tmp_path / "missing") is None
