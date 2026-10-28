"""How often did the best runs of a set of runs collect each kind of thing? Usage: count_events.py <run-name-glob> ..."""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

for pattern in sys.argv[1:]:
    for run in sorted(glob.glob(f"runs/{pattern}")):
        recs = [json.loads(f.read_text()) for f in sorted(Path(run, "best").glob("gen_*.json"))]
        if not recs:
            continue
        pipes_in_any = sum(1 for r in recs if any(e[1] == "pipe" for e in r.get("events", [])))
        best = max(recs, key=lambda r: r["fitness"])
        print(f"{Path(run).name}: best fitness {best['fitness']:.0f} max_x {best['max_x']} events of the best run "
              f"{dict(Counter(e[1] for e in best.get('events', [])))}; generations whose best run entered a pipe: {pipes_in_any}/{len(recs)}")
