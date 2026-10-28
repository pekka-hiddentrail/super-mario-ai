"""How good are the jump predictions? From several on-ground moments of a run-right play, predict each standard jump
(jumpview.features), then really do it from the same saved state and compare the landing.
(Saving and restoring states is used only here, to check the model. The agent never does it.)
Usage: validate_jumps.py [level-state] [x_to_start_sampling]"""
import sys

import numpy as np

from smwai import jumpview
from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv

level = sys.argv[1] if len(sys.argv) > 1 else "YoshiIsland1"
env = SmwEnv(EnvConfig(levels=[level], inputs="physics"))
env.reset()
em = env._env.unwrapped.em


def pad_for(names):
    pad = np.zeros(12, dtype=np.int8)
    for n in names:
        pad[env._pad[BUTTONS.index(n)]] = 1
    return pad


def read():
    w = R.wram(env._env.unwrapped.get_ram())
    return w, R.read_state(w)


def real_jump(snapshot, button, hold):
    em.set_state(snapshot)
    w, st0 = read()
    cx0, feet0 = st0.x + 8, st0.y + 32
    for t in range(120):
        names = ["RIGHT", "Y"] + ([button] if t < hold else [])
        env._env.step(pad_for(names))
        w, st = read()
        if st.dying or st.y + 32 > jumpview.LEVEL_BOTTOM:
            return -1, 0, 0
        if t >= 3 and not st.in_air:
            return 1, round((st.x + 8 - cx0) / 16), round((st.y + 32 - feet0) / 16)
    return 0, 0, 0


samples = []
for jump_every in (6, 9, 14, 20):            # four ways of running right and jumping: collect on-ground moments from each
    env.reset()
    for t in range(900):
        a = np.zeros(len(BUTTONS), dtype=bool)
        a[BUTTONS.index("RIGHT")] = a[BUTTONS.index("Y")] = True
        a[BUTTONS.index("B")] = (t % jump_every) < 3
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
        w, st = read()
        if not st.in_air and t % 10 == 0 and t > 15:
            samples.append((em.get_state(), st.x, st.x_speed))
print(f"{level}: {len(samples)} on-ground moments sampled ")

same_outcome = within1 = total = 0
dx_err, dy_err, rows = [], [], []
for snapshot, x, xs in samples:
    em.set_state(snapshot)
    w, st = read()
    feats = jumpview.features(w, st).reshape(6, 5)
    for i, (button, hold) in enumerate([(b, h) for b in jumpview.BUTTONS_MODELLED for h in jumpview.HOLDS]):
        pred = feats[i]
        outcome, dx, dy = real_jump(snapshot, button, hold)
        total += 1
        same_outcome += int(pred[0] == outcome)
        if pred[0] == 1 and outcome == 1:
            dx_err.append(abs(pred[3] - dx))
            dy_err.append(abs(pred[4] - dy))
            within1 += int(abs(pred[3] - dx) <= 1 and abs(pred[4] - dy) <= 1)
        if pred[0] != outcome or (pred[0] == 1 and abs(pred[3] - dx) > 1):
            rows.append(f"  x {x} speed {xs} {button}{hold}: predicted land {pred[0]} dx {pred[3]} dy {pred[4]}  "
                        f"actual land {outcome} dx {dx} dy {dy}")
print(f"landing outcome (land / pit) matches: {same_outcome}/{total}")
both = len(dx_err)
if both:
    print(f"when both land: mean dx error {np.mean(dx_err):.2f} tiles, mean dy error {np.mean(dy_err):.2f} tiles, "
          f"within 1 tile in both: {within1}/{both}")
print("\n".join(rows[:12]))
