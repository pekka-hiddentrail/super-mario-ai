"""Does the pipe reward fire exactly once, and does climbing pay? Plays the pattern that found the pipe on Yoshi Island 2.
Usage: probe_pipe_reward.py"""
import numpy as np

from smwai.env import BUTTONS, EnvConfig, RewardConfig, SmwEnv

cfg = EnvConfig(levels=["YoshiIsland2"], stuck_frames=900,
                reward=RewardConfig(pipe_bonus=100000.0, height_bonus=1.0))
env = SmwEnv(cfg)
for jump_every, down_every in ((9, 7), (14, 5), (6, 11), (20, 3)):
    env.reset()
    total, pipe_rewards, climbs = 0.0, [], 0
    for t in range(1500):
        a = np.zeros(len(BUTTONS), dtype=bool)
        a[BUTTONS.index("RIGHT")] = a[BUTTONS.index("Y")] = (t % 40) < 34
        a[BUTTONS.index("B")] = (t % jump_every) < 3
        a[BUTTONS.index("DOWN")] = (t % down_every) < 2
        n_events = len(env.events)
        _, r, term, trunc, info = env.step(a)
        total += r
        for e in env.events[n_events:]:
            if e[1] == "pipe":
                pipe_rewards.append((t, e[2], e[3], round(r)))
            if e[1] == "climb":
                climbs += 1
        if term or trunc:
            break
    kinds = {}
    for e in env.events:
        kinds[e[1]] = kinds.get(e[1], 0) + 1
    print(f"pattern jump {jump_every} down {down_every}: ended at step {t}, max_x {info['max_x']}, total reward {round(total)}")
    print("   pipe events (step, x, y, reward that step):", pipe_rewards, " event counts:", kinds)
