import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from smwai import ppo
from smwai.env import BUTTONS


def test_combo_masks_are_unique_and_valid():
    assert len(set(ppo.COMBO_MASKS)) == len(ppo.COMBO_MASKS)
    assert all(0 <= m < 2 ** len(BUTTONS) for m in ppo.COMBO_MASKS)


def test_right_jump_combo_presses_right_and_b():
    names = list(ppo.COMBOS)
    bits = ppo._bits(ppo.COMBO_MASKS[names.index("right_jump")])
    pressed = {b for b, on in zip(BUTTONS, bits) if on}
    assert pressed == {"RIGHT", "B"}
    assert not ppo._bits(ppo.COMBO_MASKS[names.index("noop")]).any()
