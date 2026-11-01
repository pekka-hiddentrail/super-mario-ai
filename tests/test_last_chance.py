import numpy as np

from smwai.env import BUTTONS, EnvConfig, SmwEnv


def env(stuck, chance=30):
    e = SmwEnv.__new__(SmwEnv)
    e.cfg = EnvConfig(stuck_frames=120, last_chance_frames=chance)
    e.stuck, e._chance_on = stuck, False
    return e


def pressed(action):
    return {b for b, on in zip(BUTTONS, action) if on}


def test_off_by_default_and_before_the_window():
    idle = np.zeros(len(BUTTONS), dtype=bool)
    assert pressed(env(119, chance=0)._last_chance(idle)) == set()
    assert pressed(env(80)._last_chance(idle)) == set()


def test_window_runs_right_and_jumps_with_a_fresh_press():
    e = env(92)
    held = np.zeros(len(BUTTONS), dtype=bool)
    held[BUTTONS.index("B")] = held[BUTTONS.index("LEFT")] = True       # the policy was already holding B and walking left
    first = pressed(e._last_chance(held))
    assert first == {"RIGHT", "Y"}                                      # B released for one step, LEFT dropped
    e.stuck = 96
    assert pressed(e._last_chance(held)) == {"B", "RIGHT", "Y"}         # then held: a new jump
    e.stuck = 0                                                         # progress: the window ends and re-arms
    assert pressed(e._last_chance(held)) == {"B", "LEFT"} and not e._chance_on
