"""Save a few frames near the end of a recorded run to see what stopped it.
Usage: dump_frames.py <run> <generation> <out_dir> [n_frames_before_end ...]"""
import base64
import json
import sys
import urllib.request
from pathlib import Path

run, gen, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
offsets = [int(a) for a in sys.argv[4:]] or [0, 10, 25, 50]
out.mkdir(parents=True, exist_ok=True)
data = json.load(urllib.request.urlopen(f"http://localhost:8765/api/replay?run={run}&gen={gen}", timeout=600))
steps = data["steps"]
print("steps", len(steps), "max_x", data["max_x"], "died", data["died"], "cleared", data["cleared"])
for off in offsets:
    i = max(0, len(steps) - 1 - off)
    s = steps[i]
    path = out / f"{run}_gen{gen}_end-{off}.jpg"
    path.write_bytes(base64.b64decode(s["frame"]))
    print(path, "step", i, "x", s["x"], "y", s["y"], "buttons", s["pressed"])
