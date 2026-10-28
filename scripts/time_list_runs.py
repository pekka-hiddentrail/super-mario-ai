"""Where does listing the runs spend its time? (first call builds the indexes, later calls reuse them)"""
import time
from pathlib import Path

from smwai.viewer.server import _gen_summaries, list_runs

for attempt in (1, 2, 3):
    t = time.time()
    runs = list_runs("runs")
    print(f"list_runs #{attempt}: {time.time() - t:.2f}s for {len(runs)} runs, {sum(len(r['gens']) for r in runs)} generations")
t = time.time()
bests = list(Path("runs").glob("*/best"))
print(f"glob */best: {time.time() - t:.2f}s ({len(bests)} runs)")
t = time.time()
n = sum(len(list(b.glob("gen_*.json"))) for b in bests)
print(f"glob gen files: {time.time() - t:.2f}s ({n} files)")
t = time.time()
for b in bests:
    for f in b.glob("gen_*.json"):
        f.stat()
print(f"stat all: {time.time() - t:.2f}s")
