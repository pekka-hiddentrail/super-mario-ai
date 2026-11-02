from types import SimpleNamespace

from smwai import env as E
from smwai import jumpview


def _env(bonus, hits, monkeypatch):
    e = E.SmwEnv.__new__(E.SmwEnv)
    e.cfg = SimpleNamespace(reward=E.RewardConfig(opportunity_bonus=bonus))
    e._env = SimpleNamespace(unwrapped=SimpleNamespace(get_ram=lambda: b""))
    e._chance, e._climbed, e.events, e.stuck, e.frames = None, set(), [], 99, 10
    monkeypatch.setattr(E.R, "wram", lambda ram: ram)
    monkeypatch.setattr(jumpview, "higher_landings", lambda ram, state, rise: hits)
    return e


def _st(in_air, y, x=100):
    return SimpleNamespace(in_air=in_air, y=y, x=x, rooms=0)


def _jump(e, takeoff_y, landing_y):
    e.state = _st(True, takeoff_y)
    e._opportunity(_st(False, takeoff_y))
    e.state = _st(False, landing_y)
    return e._opportunity(_st(True, landing_y + 10))


def test_pays_per_pixel_when_a_jump_to_something_higher_succeeds(monkeypatch):
    e = _env(2.0, [(120, 300)], monkeypatch)
    assert _jump(e, 200, 168) == 2.0 * 32            # landed 32 pixels higher
    assert e.stuck == 0 and e.events[-1][1] == "climb"
    assert _jump(e, 200, 168) == 0.0                 # the same landing tile pays once


def test_pays_nothing_without_a_platform_in_reach_or_without_a_climb(monkeypatch):
    assert _jump(_env(2.0, [], monkeypatch), 200, 168) == 0.0        # nothing higher in reach at takeoff: luck is not rewarded
    assert _jump(_env(2.0, [(120, 300)], monkeypatch), 200, 200) == 0.0   # the jump came down at the same height
    assert _jump(_env(0.0, [(120, 300)], monkeypatch), 200, 168) == 0.0   # off by default


def test_ledge_tiles_are_solid_to_stand_on_but_not_a_wall_or_ceiling(monkeypatch):
    ram = bytearray(E.R.MAP16_HIGH + 16)
    monkeypatch.setattr(E.R, "_tile_offset", lambda px, py: 3)
    ram[E.R.MAP16_HIGH + 3], ram[E.R.MAP16_LOW + 3] = 1, 0x01        # Map16 0x101: a ledge
    assert E.R.solid_at(ram, 0, 0) and E.R.ledge_at(ram, 0, 0)
    ram[E.R.MAP16_LOW + 3] = 0x49                                     # Map16 0x149: ground
    assert E.R.solid_at(ram, 0, 0) and not E.R.ledge_at(ram, 0, 0)


def test_ledge_bonus_pays_once_per_ledge(monkeypatch):
    e = _env(0.0, [], monkeypatch)
    e.cfg = SimpleNamespace(reward=E.RewardConfig(ledge_bonus=5.0), swing_model=False)
    e._ledges = set()
    keys = iter([(3, 20), (3, 20), None, (9, 14)])
    monkeypatch.setattr(E.R, "ledge_key", lambda ram, st, swing: next(keys))
    e.state = _st(False, 200)
    paid = [e._ledge(None) for _ in range(4)]
    assert paid == [5.0, 0.0, 0.0, 5.0]
    e.state = _st(True, 200)
    assert e._ledge(None) == 0.0                                      # in the air nothing is paid


def test_the_ledges_input_mode_has_its_own_grid_and_the_same_size_as_physics_plus_one_grid():
    physics = E.n_inputs(E.EnvConfig(inputs="physics"))
    ledges = E.n_inputs(E.EnvConfig(inputs="ledges"))
    rows, cols = E.R.grid_shape(E.view(E.EnvConfig()))
    assert ledges == physics + rows * cols
    assert ("ledge", "grid") in E.input_layout(E.EnvConfig(inputs="ledges"))
