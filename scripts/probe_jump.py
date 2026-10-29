"""Measure Mario's jump physics from game memory, frame by frame (no emulator tricks: it only presses buttons).
For several A-hold lengths and run speeds it prints the Y-speed sequence ($7D), apex, air time and landing distance.
Usage: probe_jump.py [level-state]"""
import sys

import numpy as np

from smwai import config, ram as R, train
from smwai.env import EnvConfig

Y_SPEED, IN_AIR = 0x7D, 0x72


def s8(v):
    return v - 256 if v > 127 else v


def frame(env, held):
    pad = np.zeros(12, dtype=np.int8)
    for name in held:
        pad[env._pad[["A", "B", "X", "Y", "UP", "DOWN", "LEFT", "RIGHT"].index(name)]] = 1
    env.frame = env._env.step(pad)[0]
    w = R.wram(env._env.unwrapped.get_ram())
    return w, R.s16(w, R.PLAYER_X), R.s16(w, R.PLAYER_Y), s8(int(w[R.PLAYER_X_SPEED])), s8(int(w[Y_SPEED])), int(w[IN_AIR])


def trial(env, run_frames, hold, run, button="A"):
    env.reset()
    held = (["RIGHT", "Y"] if run else ["RIGHT"] if run_frames else [])
    for _ in range(run_frames):
        frame(env, held)
    for _ in range(20):                      # make sure Mario stands on the ground
        w, x, y, vx, vy, air = frame(env, held)
        if not air:
            break
    start_x, start_y, start_vx = x, y, vx
    seq, ys = [], [y]
    for t in range(120):
        pressed = held + ([button] if t < hold else [])
        w, x, y, vx, vy, air = frame(env, pressed)
        seq.append(vy)
        ys.append(y)
        if t > 2 and not air:
            break
    return {"hold": hold, "vx0": start_vx, "vy": seq, "apex": start_y - min(ys), "frames": len(seq), "dx": x - start_x}


if __name__ == "__main__":
    level = sys.argv[1] if len(sys.argv) > 1 else "YoshiIsland2"
    train.use_config(EnvConfig(levels=[level]))
    env = train._get_env()
    if "scan" in sys.argv:                  # takeoff speed against run speed
        seen = {}
        for run_frames in range(0, 90, 3):
            for run in (False, True):
                r = trial(env, run_frames, 40, run)
                seen.setdefault(abs(r["vx0"]), set()).add(r["vy"][0])
        for vx in sorted(seen):
            print("|vx0|=%d -> takeoff vy %s" % (vx, sorted(seen[vx])))
        sys.exit()
    for run_frames, run in ((0, False), (60, True)):
        for hold in (1, 3, 6, 10, 15, 25, 40):
            r = trial(env, run_frames, hold, run)
            print("run=%s hold=%d vx0=%d v0=%d apex=%dpx frames=%d dx=%d" % (run, hold, r["vx0"], r["vy"][0], r["apex"], r["frames"], r["dx"]))
            print("   vy:", r["vy"][:60])
