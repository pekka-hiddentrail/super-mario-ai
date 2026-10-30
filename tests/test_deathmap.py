from smwai import deathmap as D
from smwai.env import EnvConfig, input_layout, n_inputs


def test_map_keeps_deaths_of_the_best_not_cleared_runs():
    m = D.DeathMap(top_share=0.5)
    m.update({0: [320, 0, False], 1: [64, 0, False], 2: [900, 0, True], 3: [500, 0, False]}, {0: 100, 1: 5, 2: 300, 3: 50})
    assert set(m.bins) == {320 // D.BIN}                 # top half = genomes 2 (cleared: ignored) and 0
    m.update({0: [320, 0, False]}, {0: 1})
    assert abs(m.bins[10] - 1.8) < 1e-9                  # old deaths fade by 0.8 per generation


def test_features_show_a_death_spot_ahead():
    bins = {"10": 2.0}
    assert D.features(None, 100) == [0, 0, 0, 0]
    assert D.features(bins, 320)[0] == 16 and D.features(bins, 320)[3] == 16       # standing on it
    assert D.features(bins, 320 - 3 * D.BIN)[1] > 0                              # a few bins before it
    far = D.features(bins, 320 - 8 * D.BIN)
    assert far[2] > 0 and far[3] == 0 and D.features(bins, 400) == [0, 0, 0, 0]  # nothing ahead once past it


def test_death_inputs_add_four_scalars():
    base = EnvConfig(inputs="ride")
    more = EnvConfig(inputs="ride", death_inputs=True)
    assert n_inputs(more) == n_inputs(base) + 4
    assert [n for n, _ in input_layout(more)][-4:] == list(D.NAMES)


def test_opposite_buttons_cancel():
    import numpy as np
    from smwai.env import BUTTONS, cancel_opposites
    a = np.zeros(len(BUTTONS), dtype=bool)
    for b in ("LEFT", "RIGHT", "UP", "A"):
        a[BUTTONS.index(b)] = True
    out = cancel_opposites(a)
    assert not out[BUTTONS.index("LEFT")] and not out[BUTTONS.index("RIGHT")]
    assert out[BUTTONS.index("UP")] and out[BUTTONS.index("A")] and a[BUTTONS.index("LEFT")]    # the input is not changed


def test_takeoff_table_and_features():
    m = D.DeathMap(top_share=1.0)
    kept = m.update({0: [320, 0, False], 1: [330, 0, False], 2: [100, 0, True]}, {0: 9, 1: 8, 2: 7},
                    {0: [300, 0, 0, 6, 1], 1: [310, 0, 0, 10, 1], 2: [90, 0, 0, 3, 0]})
    assert len(kept) == 2 and kept[0][:5] == [300, 0, 0, 6, 1] and kept[0][5] == 9.0      # the cleared run is no bad takeoff
    out = m.export()
    assert set(out["takeoff"]) == {"9"} and D.features(out, 320)[0] > 0                    # the takeoff table does not disturb the death features
    assert D.takeoff_features(None, 300) == [0, 0, 0, 0]
    here = D.takeoff_features(out, 300)
    assert here[0] == 16 and here[2] == 8 and here[3] == 16                               # mean hold (6 + 10) / 2, always running
    assert 0 < D.takeoff_features(out, 300 - 3 * D.BIN)[0] < 16
    assert D.takeoff_features(out, 300 - 10 * D.BIN) == [0, 0, 0, 0] and D.takeoff_features(out, 500) == [0, 0, 0, 0]


def test_takeoff_inputs_add_four_scalars():
    base = EnvConfig(inputs="ride", death_inputs=True)
    more = EnvConfig(inputs="ride", death_inputs=True, takeoff_inputs=True)
    assert n_inputs(more) == n_inputs(base) + 4
    assert [n for n, _ in input_layout(more)][-4:] == list(D.TAKEOFF_NAMES)


def test_env_follows_the_last_takeoff():
    import numpy as np
    from types import SimpleNamespace as S
    from smwai.env import BUTTONS, SmwEnv
    env = SmwEnv(EnvConfig())

    def act(*names):
        a = np.zeros(len(BUTTONS), dtype=bool)
        for n in names:
            a[BUTTONS.index(n)] = True
        return a

    def step(prev_air, air, x, *names):
        env.state = S(in_air=air, x=x, y=0)
        env._track_takeoff(S(in_air=prev_air, x=x - 8, y=0), act(*names))

    step(False, True, 108, "B", "Y", "RIGHT")            # takeoff: x is where Mario stood
    step(True, True, 116, "B", "Y")
    step(True, True, 124, "Y")                           # released: the hold stops counting
    step(True, True, 132, "B", "Y")
    t = env._takeoff
    assert (t["x"], t["hold"], t["run"]) == (100, 2, 1)
    step(True, False, 140)                               # landed: nothing to blame
    assert env._takeoff is None
    step(False, True, 148, "RIGHT")                      # walking off a ledge is a takeoff with no jump
    assert env._takeoff["hold"] == 0 and env._takeoff["run"] == 0


def test_ride_motion_shows_a_carried_mario():
    from types import SimpleNamespace as S
    from smwai.env import RIDE_INPUTS, ride_motion
    still = S(in_air=False, x=100, y=200, x_speed=0)
    assert ride_motion(still, None).tolist() == [0, 0]                                   # first step: nothing to compare with
    assert ride_motion(S(in_air=False, x=100, y=192, x_speed=0), still).tolist() == [-8, 0]     # a rising platform: up 8 px
    assert ride_motion(S(in_air=False, x=108, y=200, x_speed=0), still).tolist() == [0, 8]      # carried right although he does not walk
    assert ride_motion(S(in_air=False, x=108, y=200, x_speed=32), still).tolist() == [0, 0]     # walking 8 px in 4 frames explains the move
    assert ride_motion(S(in_air=True, x=108, y=150, x_speed=0), still).tolist() == [0, 0]       # in the air the jump physics apply
    assert n_inputs(EnvConfig(inputs="ride", ride_motion=True)) == n_inputs(EnvConfig(inputs="ride")) + 2
    assert [n for n, _ in input_layout(EnvConfig(inputs="ride", ride_motion=True))][-2:] == list(RIDE_INPUTS)


def test_a_bad_takeoff_fades_in_four_generations():
    m = D.DeathMap(top_share=1.0)
    m.update({0: [320, 0, False]}, {0: 1.0}, {0: [300, 0, 0, 6, 1]})
    weights = []
    for _ in range(5):
        m.update({}, {}, None)
        weights.append(m.takeoffs.get(300 // D.BIN, [0])[0])
    assert weights == [0.5, 0.25, 0.125, 0, 0]             # 0.0625 is below the cut: four generations
    assert not m.takeoffs                                  # gone after four generations without a new death there
