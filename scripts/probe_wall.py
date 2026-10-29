"""Is the dirt filler tile (Map16 0x03F) solid? Walk Mario right into the pillar at the start of Yoshi Island 3
(tiles 3..13, bodies made of 0x3F/0x41) and see where he stops."""
import numpy as np

from smwai import ram as R
from smwai.env import BUTTONS, EnvConfig, SmwEnv

env = SmwEnv(EnvConfig(levels=["YoshiIsland3"]))
obs, info = env.reset()
w = R.wram(env._env.unwrapped.get_ram())
print("start x,y", info["x"], info["y"])
tx = (info["x"] + 8) // 16
row = (info["y"] + 16) // 16
print("tiles to the right of Mario at body height, x tile -> id:",
      {t: f"{int(w[R.MAP16_HIGH + (t // 16) * R.TILES_PER_PAGE + row * 16 + t % 16]):x}:"
          f"{int(w[R.MAP16_LOW + (t // 16) * R.TILES_PER_PAGE + row * 16 + t % 16]):02x}" for t in range(tx + 1, tx + 20)})
pad = np.zeros(len(BUTTONS), dtype=bool)
pad[BUTTONS.index("RIGHT")] = pad[BUTTONS.index("Y")] = True
last = None
for t in range(120):
    _, _, term, trunc, info = env.step(pad)
    if t % 10 == 0 or term or trunc:
        print(f"step {t:3d} x {info['x']} y {info['y']} x_speed {env.state.x_speed}")
    if term or trunc:
        break
