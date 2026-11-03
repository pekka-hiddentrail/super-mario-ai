from smwai.mapplay import RandomMapPolicy
from smwai.overworld import DIRECTIONS, LEVELS


def test_every_shipped_level_has_a_translevel_number():
    assert len(LEVELS) == 24 and len(set(LEVELS.values())) == 24
    assert LEVELS[0x2A] == "YoshiIsland2" and LEVELS[0x27] == "YoshiIsland3"


def test_random_map_policy_picks_a_direction_and_one_to_three_steps_and_repeats_with_its_seed():
    a, b = RandomMapPolicy(7), RandomMapPolicy(7)
    picks = [a.choose(None) for _ in range(50)]
    assert picks == [b.choose(None) for _ in range(50)]
    assert all(d in DIRECTIONS and s in (1, 2, 3) for d, s in picks)
    assert {s for _, s in picks} == {1, 2, 3} and {d for d, _ in picks} == set(DIRECTIONS)
