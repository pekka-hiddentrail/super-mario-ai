import numpy as np

from smwai.env import BUTTONS, EnvConfig, SmwEnv


def env(stuck, chance=30, wall=True):
    e = SmwEnv.__new__(SmwEnv)
    e.cfg = EnvConfig(stuck_frames=120, last_chance_frames=chance)
    e.stuck, e._chance_on = stuck, False
    e._wall_ahead = lambda: wall
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


def test_no_wall_no_jump_until_one_is_there():
    e = env(100, wall=False)
    idle = np.zeros(len(BUTTONS), dtype=bool)
    assert pressed(e._last_chance(idle)) == set() and not e._chance_on
    e._wall_ahead = lambda: True
    assert pressed(e._last_chance(idle)) == {"RIGHT", "Y"} and e._chance_on
    e._wall_ahead = lambda: False                                       # in the air the wall is below him: it runs on
    assert pressed(e._last_chance(idle)) == {"B", "RIGHT", "Y"}


def test_no_spin_drops_the_a_button():
    e = SmwEnv.__new__(SmwEnv)
    e.cfg = EnvConfig(no_spin=True)
    seen = []
    e._last_chance = lambda a: (seen.append(a), (_ for _ in ()).throw(StopIteration))[0]
    held = np.ones(len(BUTTONS), dtype=bool)
    try:
        e.step(held)
    except StopIteration:
        pass
    assert not seen[0][BUTTONS.index("A")] and seen[0][BUTTONS.index("B")]
