"""Tally over all rooms-pipes runs (any seeds): how many took a pipe, cleared the level, or both. Usage: pipe_summary.py"""
import glob
import json
from pathlib import Path

rows = {}
for run in sorted(glob.glob("runs/exp-rooms-pipes*-*-rooms-s*")):
    recs = [json.loads(f.read_text()) for f in sorted(Path(run, "best").glob("gen_*.json"))]
    if not recs:
        continue
    level = "yi2" if "-yi2-" in run else "yi4"
    took_pipe = any(e[1] == "pipe" for r in recs for e in r.get("events", []))
    cleared = any(r["cleared"] for r in recs)
    both = any(r["cleared"] and any(e[1] == "pipe" for e in r.get("events", [])) for r in recs)
    best = max(recs, key=lambda r: r["fitness"])
    rows.setdefault(level, []).append((Path(run).name, took_pipe, cleared, both, round(best["fitness"]), best["max_x"]))
for level, items in rows.items():
    n = len(items)
    print(f"{level}: {n} runs  took a pipe {sum(i[1] for i in items)}/{n}  cleared {sum(i[2] for i in items)}/{n}  "
          f"cleared AFTER a pipe in the same run {sum(i[3] for i in items)}/{n}")
    for i in items:
        print(f"    {i[0]}: pipe={i[1]} cleared={i[2]} cleared-after-pipe={i[3]} best fitness {i[4]} max_x {i[5]}")
