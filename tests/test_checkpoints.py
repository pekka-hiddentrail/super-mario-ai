from pathlib import Path
from types import SimpleNamespace

from smwai.env import EnvConfig, SmwEnv
from smwai.train import promote_checkpoint


class FakeEm:
    def get_state(self):
        return b"state"


def env(tmp_path, **state):
    e = SmwEnv.__new__(SmwEnv)
    e.cfg = EnvConfig(checkpoints=[300, 576], checkpoint_dir=str(tmp_path))
    e._env = SimpleNamespace(unwrapped=SimpleNamespace(em=FakeEm()))
    e.state = SimpleNamespace(**{"x": 310, "y": 120, "in_air": False, "dying": False, "pipe_timer": 0, "in_level": True, **state})
    e.events, e.frames, e._cp_next, e._cp_n, e.start_path, e._trace, e.base, e._total, e.new_cands = [], 100, 0, 0, None, [1, 2, 3], 0.0, 0.0, []
    return e


def test_standing_near_a_checkpoint_saves_a_candidate(tmp_path):
    e = env(tmp_path)
    e._checkpoint()
    assert [p.name.split("_f")[0] for p in (tmp_path / "cand").glob("*.state")] == ["cand_01_x300"]
    assert e._cp_next == 1 and e.events[-1][1] == "checkpoint"


def test_no_save_in_the_air_too_low_or_too_far(tmp_path):
    for kw in ({"in_air": True}, {"y": 250}, {"x": 330}, {"x": 276}):
        e = env(tmp_path, **kw)
        e._checkpoint()
        assert not (tmp_path / "cand").exists(), kw


def test_promote_takes_the_furthest_checkpoint_then_the_fewest_frames(tmp_path):
    (tmp_path / "cand").mkdir()
    for name in ("cand_01_x300_f0000200_1_1", "cand_01_x300_f0000100_2_1", "cand_02_x576_f0000900_3_1", "cand_02_x576_f0000700_4_1"):
        (tmp_path / "cand" / f"{name}.state").write_bytes(name.encode())
        (tmp_path / "cand" / f"{name}.json").write_text('{"start": null, "trace": [7]}')
    cfg = EnvConfig(checkpoints=[300, 576], checkpoint_dir=str(tmp_path))
    assert promote_checkpoint(cfg) == (2, 576, 700)
    saved = tmp_path / "checkpoint_02_x576.state"
    assert saved.read_bytes() == b"cand_02_x576_f0000700_4_1"
    assert Path((tmp_path / "current.txt").read_text()) == saved.resolve()
    assert (tmp_path / "checkpoint_01_x300.state").read_bytes() == b"cand_01_x300_f0000100_2_1"     # the earlier checkpoint keeps its fastest save too
    assert not list((tmp_path / "cand").glob("*.state"))
    assert (tmp_path / "checkpoint_02_x576.json").exists()
    assert promote_checkpoint(cfg) is None


def test_every_n_pixels_any_height_and_only_once_per_increment(tmp_path):
    e = env(tmp_path, x=905, y=330)
    e.cfg = EnvConfig(checkpoint_every=300, checkpoint_dir=str(tmp_path))
    e._checkpoint()                                         # 905 is past 3 * 300, on the ground, no height rule
    assert [p.name.split("_f")[0] for p in (tmp_path / "cand").glob("*.state")] == ["cand_03_x900"]
    assert e._cp_next == 3
    e.state.x = 910
    e._checkpoint()                                         # still the same increment: nothing new
    assert len(list((tmp_path / "cand").glob("*.state"))) == 1
    e.state.in_air, e.state.x = True, 1210
    e._checkpoint()                                         # in the air: wait until he stands
    assert len(list((tmp_path / "cand").glob("*.state"))) == 1
    e.state.in_air = False
    e._checkpoint()
    assert any(p.name.startswith("cand_04_x1200") for p in (tmp_path / "cand").glob("*.state"))


def test_a_checkpoint_that_leads_nowhere_is_given_up_for_the_one_before(tmp_path):
    from smwai.train import revert_checkpoint
    cfg = EnvConfig(checkpoints=[300, 576], checkpoint_dir=str(tmp_path))
    for number, cx in ((1, 300), (2, 576), (3, 915)):
        (tmp_path / f"checkpoint_{number:02d}_x{cx}.state").write_bytes(b"s")
    (tmp_path / "current.txt").write_text(str((tmp_path / "checkpoint_03_x915.state").resolve()))
    assert revert_checkpoint(cfg) == (3, 2)
    assert (tmp_path / "checkpoint_03_x915.bad").exists()
    assert Path((tmp_path / "current.txt").read_text()).name == "checkpoint_02_x576.state"
    assert revert_checkpoint(cfg) == (2, 1)
    assert revert_checkpoint(cfg) == (1, 0)                    # nothing left: the level's own start
    assert not (tmp_path / "current.txt").exists() and revert_checkpoint(cfg) is None


def test_a_start_save_carries_the_score_the_chain_had_collected(tmp_path):
    import json
    save = tmp_path / "checkpoint_02_x600.state"
    save.write_bytes(b"s")
    assert SmwEnv._start_score(str(save)) == 0.0                         # no json: a plain start
    save.with_suffix(".json").write_text(json.dumps({"start": None, "trace": [], "score": 812.5}))
    assert SmwEnv._start_score(str(save)) == 812.5
    assert SmwEnv._start_score(None) == 0.0
    e = env(tmp_path)
    e.base, e._total = 800.0, 12.5
    e._save_candidate(1, 300)
    side = next((tmp_path / "cand").glob("*.json"))
    assert json.loads(side.read_text())["score"] == 812.5
