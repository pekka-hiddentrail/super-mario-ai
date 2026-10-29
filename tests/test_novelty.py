from smwai import novelty as N


def test_fingerprint_uses_cells_and_event_kinds():
    fp = N.fingerprint([N.cell_id(10, 10), N.cell_id(100, 10)], [[5, "pipe", 10, 10], [6, "other", 0, 0]])
    assert N.cell_id(10, 10) in fp and N.cell_id(100, 10) in fp
    assert -1 - N.EVENT_KINDS.index("pipe") in fp and len(fp) == 3


def test_distance_is_zero_for_equal_and_one_for_disjoint():
    a, b = frozenset({1, 2}), frozenset({3, 4})
    assert N.distance(a, a) == 0.0 and N.distance(a, b) == 1.0
    assert abs(N.distance(frozenset({1, 2}), frozenset({2, 3})) - 2 / 3) < 1e-9


def test_archive_scores_copies_low_and_new_routes_high():
    arc = N.NoveltyArchive(k=2, add_per_gen=3)
    same = frozenset({1, 2, 3})
    prints = {0: same, 1: same, 2: frozenset({7, 8, 9})}
    scores = arc.score(prints)
    assert scores[2] > scores[0]
    assert arc.update(prints, scores) == 6                       # six cells nobody had reached
    assert arc.score({5: same})[5] < arc.score({5: frozenset({40, 41})})[5]    # a known route is less novel than a new one


def test_blend_keeps_best_fitness_first():
    out = N.blend({0: 100.0, 1: 10.0, 2: 50.0}, {0: 0.0, 1: 1.0, 2: 0.5}, 0.9)
    assert max(out, key=out.get) == 0
