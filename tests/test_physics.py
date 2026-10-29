"""The jump formula against apex heights measured in the emulator (scripts/probe_jump.py, flat ground)."""
import pytest

from smwai import physics

# (x_speed, frames the jump button is held, measured apex in pixels)
MEASURED = [(0, 1, 29), (0, 3, 33), (0, 6, 39), (0, 10, 45), (0, 15, 51), (0, 25, 55), (0, 40, 55),
            (36, 1, 35), (36, 10, 53), (36, 25, 67), (36, 40, 67)]


@pytest.mark.parametrize("vx,hold,apex", MEASURED)
def test_apex_matches_emulator(vx, hold, apex):
    predicted = -min(p.dy for p in physics.trajectory(vx, hold))
    assert abs(predicted - apex) <= 1.0


def test_longer_hold_jumps_higher_up_to_a_cap():
    heights = [-min(p.dy for p in physics.trajectory(0, h)) for h in (1, 5, 10, 20, 40)]
    assert heights == sorted(heights) and heights[-1] == pytest.approx(heights[-2], abs=1.5)


def test_landing_on_floor_and_pit():
    floor = lambda px, py: py >= 100           # flat floor at y=100, start standing on it
    frame, x, y = physics.landing(36, 10, floor, 0, 100)
    assert 30 < frame < 45 and y >= 100 and x > 50
    assert physics.landing(36, 10, lambda px, py: False, 0, 100) is None


def test_hold_for_apex():
    assert physics.hold_for_apex(0, 40) in range(5, 9)
    assert physics.hold_for_apex(0, 80) is None
