"""Pipe at x 3878 on Yoshi Island 2 with room_progress on: after the pipe, does walking right in the bonus room earn progress,
and what happens when the room is left? Prints the reward per phase."""
import numpy as np

from smwai.env import BUTTONS, EnvConfig, RewardConfig, SmwEnv

env = SmwEnv(EnvConfig(levels=["YoshiIsland2"], stuck_frames=600, room_progress=True,
                       reward=RewardConfig(pipe_bonus=100000.0)))
env.reset()
phase, totals, last_room = "main", {"main": 0.0}, 0
for t in range(1800):
    a = np.zeros(len(BUTTONS), dtype=bool)
    in_pipe_phase = env.state.rooms == 0 and t < 700
    a[BUTTONS.index("RIGHT")] = a[BUTTONS.index("Y")] = (t % 40) < 34 if in_pipe_phase else True
    a[BUTTONS.index("B")] = (t % 14) < 3
    a[BUTTONS.index("DOWN")] = (t % 5) < 2 if in_pipe_phase else False
    _, r, term, trunc, info = env.step(a)
    phase = f"room{env.state.rooms}"
    totals[phase] = totals.get(phase, 0.0) + r
    if env.state.rooms != last_room:
        print(f"t={t}: entered room #{env.state.rooms} at x={info['x']} y={info['y']}  signature {env.room_sig}  room max {env.max_x}")
        last_room = env.state.rooms
    if term or trunc:
        break
print("ended at step", t, "cleared" if info.get("cleared") else "", "died" if info.get("died") else "", "max_x", info["max_x"])
print("reward per room visit:", {k: round(v) for k, v in totals.items()})
print("events:", [(e[1], e[2]) for e in env.events if e[1] in ("pipe",)])
