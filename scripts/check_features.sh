#!/usr/bin/env bash
# End-to-end check that the features work together on a real explore-mode run (pipe + per-room progress):
# replay API (network graph, layout), path API (events, path), level map, video list, dashboard page.
run="${1:-exp-rooms-pipes-3k-yi4-rooms-s1}"
gen="${2:-6}"
. ~/venv/bin/activate
python - "$run" "$gen" <<'PY'
import json, sys, urllib.request
run, gen = sys.argv[1], sys.argv[2]
def get(path, timeout=300):
    with urllib.request.urlopen("http://localhost:8765" + path, timeout=timeout) as r:
        return json.load(r)
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
runs = get("/api/runs")
check("run list newest first and contains the run", any(r["name"] == run for r in runs), f"{len(runs)} runs")
rep = get(f"/api/replay?run={run}&gen={gen}")
names = [l["name"] for l in rep["layout"]]
check("replay: network graph present", rep["graph"] and len(rep["graph"]["inputs"]) > 800, f"{len(rep['graph']['inputs'])} inputs, {len(rep['steps'])} steps")
check("replay: explore layout has the pipe grid and jump options", "pipe" in names and "A10:land" in names, f"{len(names)} input groups")
pth = get(f"/api/path?run={run}&gen={gen}&events=1")
kinds = sorted({e[1] for e in pth["events"]})
check("path: recorded with events", len(pth["path"]) > 50, f"{len(pth['path'])} points, events {kinds}")
lvl = get(f"/api/level?run={run}")
check("level map: pipes marked", any("P" in row for row in lvl["rows"]), f"{lvl['pages']} screens")
vids = get("/api/videos")
check("video list answers", isinstance(vids, list), f"{len(vids)} videos")
page = urllib.request.urlopen("http://localhost:8501", timeout=30).status
check("dashboard answers", page == 200)
print("ALL PASS" if ok else "SOME CHECKS FAILED")
PY
