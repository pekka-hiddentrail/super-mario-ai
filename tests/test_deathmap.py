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


def test_node_nudge_window_height_and_cap():
    node = {"x": 300, "y": 200, "w": 1.0, "n": [0.5, -0.5, 0.2, 0, 0, 0, 0, 0]}
    assert D.node_nudge([node], 300, 200)[:3] == [D.NODE_CAP, -D.NODE_CAP, 0.2]        # summed nudge is limited to +-0.45 per button
    assert D.node_nudge([node], 268, 200)[2] == 0.2 and D.node_nudge([node], 267, 200) == [0.0] * 8        # 32 px before ...
    assert D.node_nudge([node], 308, 200)[2] == 0.2 and D.node_nudge([node], 309, 200) == [0.0] * 8        # ... to 8 px after
    assert D.node_nudge([node], 300, 248)[2] == 0.2 and D.node_nudge([node], 300, 249) == [0.0] * 8        # same height band only (a bypass above)
    assert D.node_nudge([{**node, "w": 0.5}], 300, 200)[2] == 0.1                      # the nudge shrinks with the node's weight
    assert D.node_nudge(None, 300, 200) == [0.0] * 8


def test_random_nudge_touches_one_to_all_buttons_mostly_four_or_five():
    import random
    rng = random.Random(5)
    counts = []
    for _ in range(400):
        n = D.random_nudge(rng)
        assert len(n) == 8 and all(v == 0.0 or 0.1 <= abs(v) <= 0.5 for v in n)
        counts.append(sum(v != 0.0 for v in n))
    assert min(counts) >= 1 and max(counts) <= 8
    assert sum(c in (4, 5) for c in counts) > 0.4 * len(counts)


def test_nodes_fade_by_half_or_slowly_when_they_work_and_leave():
    import random
    m = D.NodeMemory()
    m.update([300, 200, 0, 5, 1, 9.0], lambda x, y: False, random.Random(1))
    assert m.nodes[0][3] == 1.0 and len(m.nodes[0][2]) == 8
    weights = []
    for _ in range(5):
        m.update(None, lambda x, y: False)
        weights.append(m.nodes[0][3] if m.nodes else 0)
    assert weights == [0.5, 0.25, 0.125, 0.0625, 0]                                     # five generations, then gone
    m.update([300, 200, 0, 5, 1, 9.0], lambda x, y: False, random.Random(1))
    for _ in range(3):
        m.update(None, lambda x, y: True)
    assert abs(m.nodes[0][3] - 0.85 ** 3) < 1e-9                                        # a working node fades slowly
    for _ in range(D.NODE_MAX_AGE):
        m.update(None, lambda x, y: True)
    assert not m.nodes                                                                  # but not forever
    assert D.features({"nodes": [1], "10": 2.0}, 320)[0] == 16                          # the node list does not disturb the death features


def test_env_decide_applies_nodes_on_the_ground_only():
    import numpy as np
    from types import SimpleNamespace as S
    from smwai.env import SmwEnv
    env = SmwEnv(EnvConfig(death_nodes=True))
    env.death_map = {"nodes": [{"x": 300, "y": 200, "w": 1.0, "n": [-0.5, 0.5, 0, 0, 0, 0, 0, 0]}]}
    out = [0.7, 0.3, 0.9, 0.2, 0.1, 0.1, 0.1, 0.1]
    env.state = S(in_air=False, x=300, y=200)
    assert env.decide(out).tolist() == [False, True, True, False, False, False, False, False]   # 0.45 shifts the logit by 5.4: 0.7 stops, 0.3 presses
    fairly = [0.99, 0.01, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0]                                # outputs inside about 0.5%-99.5% flip ...
    assert env.decide(fairly).tolist() == [False, True, True, False, False, False, False, False]
    sure = [0.999, 0.001, 0.9999, 0.0001, 0.0, 0.0, 0.0, 0.0]                          # ... a network that is surer than that keeps its say
    assert env.decide(sure).tolist() == [True, False, True, False, False, False, False, False]
    env.state = S(in_air=True, x=300, y=200)
    assert env.decide(out).tolist() == [True, False, True, False, False, False, False, False]    # in the air: the plain network
    env.state = S(in_air=False, x=300, y=200)
    env.cfg.death_nodes = False
    assert env.decide(out).tolist() == [True, False, True, False, False, False, False, False]    # switch off: nothing changes
    assert EnvConfig().death_nodes is False


def test_a_node_where_attempts_still_die_fades_fast_even_if_they_pass_it(tmp_path):
    from smwai.novelty import cell_id
    from smwai.train import RunReporter
    rep = RunReporter(tmp_path / "run", None, {"frames": 0}, death_nodes=True)
    rep.nodes.nodes = [[285, 352, [0.1] * 8, 1.0, 0], [150, 352, [0.1] * 8, 1.0, 0]]
    run = sorted({cell_id(x, 352) for x in range(100, 560, 8)})             # an attempt that ran right through both spots and on to x 560
    behaviour = {1: {"cells": run, "takeoff": [292, 352, 0, 5, 1]}, 2: {"cells": run, "takeoff": None}}
    rep._update_nodes({1: 10.0, 2: 5.0}, behaviour, [])
    weights = {n[0]: n[3] for n in rep.nodes.nodes}
    assert weights[285] == 0.5                  # still where the best attempt takes off and dies: the node fails, passing it does not count
    assert weights[150] == 0.85                 # attempts pass it and take off elsewhere: it works
