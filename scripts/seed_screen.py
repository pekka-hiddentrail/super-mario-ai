"""Per seed: how far and how high did the runs of an experiment get? Usage: seed_screen.py <experiment name> [--high 100]
A seed 'climbed' when a best run of any generation got Mario above y = 100 (the high platforms of Yoshi's Island 3)."""
import json
import sys
from pathlib import Path

name = sys.argv[1]
high = int(sys.argv[sys.argv.index("--high") + 1]) if "--high" in sys.argv else 100
by = {}
for run in sorted(Path("runs").glob(f"exp-{name}-*")):
    variant, seed = run.name[len(f"exp-{name}-"):].rsplit("-s", 1)
    gens = [json.loads(f.read_text()) for f in sorted((run / "best").glob("gen_*.json"))]
    if not gens:
        continue
    by.setdefault(variant, []).append((int(seed), max(g["max_x"] for g in gens), min(min(p[1] for p in g["path"]) for g in gens), max(g["fitness"] for g in gens)))
for variant, rows in by.items():
    rows.sort()
    climbed = sum(r[2] < high for r in rows)
    xs = sorted(r[1] for r in rows)
    print(f"{variant}: {len(rows)} seeds, climbed {climbed}, median best x {xs[len(xs) // 2]}, mean best x {sum(xs) / len(xs):.0f}")
    print("  " + "  ".join(f"s{s}: x{x} y{y}" for s, x, y, _ in rows))
