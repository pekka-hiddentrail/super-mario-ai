"""Can Mario ride the moving platform (sprite 0x5F) over the Yoshi Island 3 pit, and with what timing? Analysis only (not part of
the agent): replays a recorded run to the last ground contact before the first big jump, then tries jump timings from saved
states in two stages. Usage: probe_platform.py <run-dir>"""
import json
import sys
from pathlib import Path

import numpy as np

from smwai import config, train
from smwai import ram as R
from smwai.env import BUTTONS

run = Path(sys.argv[1])
rec = max((json.loads(f.read_text()) for f in (run / "best").glob("gen_*.json")), key=lambda r: r["fitness"])
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
env.reset()
em = env._env.unwrapped.em
snap, snap_xy = None, None
for mask in rec["actions"]:
    env.step(train._from_mask(mask))
    st = env.state
    if st.x > 400:
        break
    if st.x >= 200 and not st.in_air:
        snap, snap_xy = em.get_state(), (st.x, st.y)       # the last time on the ground before the first big jump
print("start of trials from x,y =", snap_xy)


def pad(names):
    p = np.zeros(12, dtype=np.int8)
    for n in names:
        p[env._pad[BUTTONS.index(n)]] = 1
    return p


def platform(w):
    for k in range(12):
        if w[R.SPRITE_STATUS + k] and w[R.SPRITE_NUMBER + k] == 0x5F:
            return (int(w[R.SPRITE_X_LO + k]) | int(w[R.SPRITE_X_HI + k]) << 8,
                    int(w[R.SPRITE_Y_LO + k]) | int(w[R.SPRITE_Y_HI + k]) << 8)
    return None


def jump_trial(state, button, hold, wait, max_frames=160):
    """From a saved state: wait, then run right with a jump. Returns (landing_state_or_None, (best_x, on_platform, where))."""
    em.set_state(state)
    was_air, best_x = False, 0
    for t in range(max_frames):
        keys = ["RIGHT", "Y"] if t >= wait else []
        if wait <= t < wait + hold:
            keys.append(button)
        env._env.step(pad(keys))
        w = R.wram(env._env.unwrapped.get_ram())
        st = R.read_state(w)
        best_x = max(best_x, st.x)
        pf = platform(w)
        if st.in_air:
            was_air = True
        if st.dying or st.y > 440:
            return None, (best_x, False, "fell")
        if was_air and not st.in_air:
            on_platform = bool(pf) and abs(st.x + 8 - (pf[0] + 8)) < 28 and 440 <= st.x <= 700
            return em.get_state(), (best_x, on_platform, (st.x, st.y))
    return None, (best_x, False, "no landing")


# stage 1: jump from the first ledge, wherever it lands safely
landings = []
for button in ("A", "B"):
    for hold in (10, 25):
        for wait in range(0, 60, 6):
            state, info = jump_trial(snap, button, hold, wait)
            if state is not None:
                landings.append((state, info[2], (button, hold, wait)))
uniq = {}
for state, xy, how in landings:
    uniq.setdefault((xy[0] // 8, xy[1] // 8), (state, xy, how))
print(f"stage 1: {len(landings)} safe landings, {len(uniq)} distinct spots:", sorted(v[1] for v in uniq.values()))

# stage 2: from each landing spot, jump again, with timings that may meet the moving platform
hits, best_overall = [], 0
for key, (state, xy, how) in uniq.items():
    for button in ("A", "B"):
        for hold in (10, 25):
            for wait in range(0, 120, 6):
                st2, info = jump_trial(state, button, hold, wait)
                best_overall = max(best_overall, info[0])
                if info[1] or info[0] > 640:
                    hits.append((xy, how, (button, hold, wait), info))
print(f"stage 2: best x {best_overall}; trials that rode the platform or passed x 640: {len(hits)}")
for h in hits[:10]:
    print("  from", h[0], "first jump", h[1], "second jump", h[2], "->", h[3])
