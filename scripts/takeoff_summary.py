"""Where did the best runs take off for their fatal fall? Reads runs/<run>/takeoffs.jsonl (written by every training run) and
prints a histogram per 32 px, with the mean jump hold and the share of takeoffs with Y held.
Usage: takeoff_summary.py <run folder> [<run folder> ...] [--from GEN] [--top 12]"""
import json
import sys
from collections import defaultdict
from pathlib import Path

args = sys.argv[1:]
first = int(args[args.index("--from") + 1]) if "--from" in args else 0
top = int(args[args.index("--top") + 1]) if "--top" in args else 12
runs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--from", "--top"))]
for run in runs:
    f = Path(run) / "takeoffs.jsonl"
    if not f.exists():
        print(run, ": no takeoffs.jsonl (run trained before takeoffs were recorded)")
        continue
    bins = defaultdict(lambda: [0, 0, 0, 0])            # count, hold sum, run sum, jump-less (walked off)
    for line in f.read_text().splitlines():
        row = json.loads(line)
        if row["generation"] < first:
            continue
        for x, _y, _mask, hold, run_, _fit in row["takeoffs"]:
            b = bins[x // 32]
            b[0] += 1; b[1] += hold; b[2] += run_; b[3] += hold == 0
    total = sum(b[0] for b in bins.values())
    print(f"{Path(run).name}: {total} fatal takeoffs of top runs (from generation {first})")
    for b, (n, h, r, walk) in sorted(bins.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"  x {b * 32:5d}-{b * 32 + 31:5d}  {n:4d} ({100 * n / total:4.1f}%)  mean hold {h / n:4.1f} steps  Y held {100 * r / n:3.0f}%  no jump {100 * walk / n:3.0f}%")
