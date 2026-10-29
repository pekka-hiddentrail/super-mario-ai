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
