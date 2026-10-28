"""Reward events that need the real game (skipped where stable-retro is not installed)."""
import numpy as np
import pytest

pytest.importorskip("stable_retro")

from smwai.env import BUTTONS, EnvConfig, RewardConfig, SmwEnv  # noqa: E402


def _run_pattern(env, jump_every, down_every, steps=700):
    env.reset()
    rewards = []
    for t in range(steps):
        a = np.zeros(len(BUTTONS), dtype=bool)
        a[BUTTONS.index("RIGHT")] = a[BUTTONS.index("Y")] = (t % 40) < 34
        a[BUTTONS.index("B")] = (t % jump_every) < 3
        a[BUTTONS.index("DOWN")] = (t % down_every) < 2
        _, r, term, trunc, info = env.step(a)
        rewards.append(r)
        if term or trunc:
            break
    return rewards


def test_pipe_pays_once_and_exit_does_not_pay():
    env = SmwEnv(EnvConfig(levels=["YoshiIsland2"], stuck_frames=900, reward=RewardConfig(pipe_bonus=100000.0)))
    rewards = _run_pattern(env, 14, 5)
    pipes = [e for e in env.events if e[1] == "pipe"]
    assert len(pipes) == 1 and 3800 < pipes[0][2] < 3950          # the pipe near x 3878 on Yoshi Island 2
    assert sum(r > 50000 for r in rewards) == 1


def test_no_pipe_reward_when_bonus_is_zero():
    env = SmwEnv(EnvConfig(levels=["YoshiIsland2"], stuck_frames=900))
    rewards = _run_pattern(env, 14, 5)
    assert max(rewards) < 1000                                      # the event is recorded but pays nothing
    assert any(e[1] == "pipe" for e in env.events)
