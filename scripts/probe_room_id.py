"""Does $0E,$0F identify the room? Runs the pattern that enters the pipe on Yoshi Island 2 and prints it before and after."""
import numpy as np

from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv

env = SmwEnv(EnvConfig(levels=["YoshiIsland2"], stuck_frames=900))
env.reset()
w = R.wram(env._env.unwrapped.get_ram())
print("start: $0E,$0F/$010C =", int(w[0x0E]), int(w[0x0F]), " $13BF =", int(w[0x13BF]), " $141A =", int(w[0x141A]))
last = None
for t in range(900):
    a = np.zeros(len(BUTTONS), dtype=bool)
    a[BUTTONS.index("RIGHT")] = a[BUTTONS.index("Y")] = (t % 40) < 34
    a[BUTTONS.index("B")] = (t % 14) < 3
    a[BUTTONS.index("DOWN")] = (t % 5) < 2
    _, _, term, trunc, info = env.step(a)
    w = R.wram(env._env.unwrapped.get_ram())
    cur = (int(w[0x0E]) | int(w[0x0F]) << 8, int(w[0x141A]), int(w[0x13BF]))
    if cur != last:
        print(f"t={t} x={info['x']} y={info['y']}  level/room id $0E,$0F={cur[0]} (hex {cur[0]:x})  rooms $141A={cur[1]}  translevel $13BF={cur[2]}")
        last = cur
    if term or trunc:
        break
