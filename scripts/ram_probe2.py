import numpy as np, stable_retro as retro
env = retro.make("SuperMarioWorld-Snes-v0", state="YoshiIsland2", use_restricted_actions=retro.Actions.ALL, render_mode=None)
env.reset()
for _ in range(120): env.step(np.zeros(12, dtype=np.int8))
b = env.unwrapped.data.memory.blocks
print({hex(k): len(v) for k, v in b.items()})
ram = env.unwrapped.get_ram()
for k, v in b.items():
    arr = np.frombuffer(v, dtype=np.uint8)
    print(hex(k), len(arr), "match ram head:", np.array_equal(arr[:256], ram[:256]))
