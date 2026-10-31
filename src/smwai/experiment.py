"""Experiment runner: several variants x seeds under the same budget, summarised in one table.

A spec (see configs/experiments/) names a base config and a list of variants; each variant
overrides some settings of the base. Every (variant, seed) is trained in its own process and
lands in runs/<stamp>-<experiment>-<variant>-s<seed>/ (stamp = start time, YYYYMMDD-HHMM), so the dashboard shows it like any other run.
"""
import copy
import csv
import json
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

from . import config


def merge(base, over):
    out = copy.deepcopy(base)          # never share nested dicts between runs
    for key, value in over.items():
        out[key] = merge(out.get(key, {}), value) if isinstance(value, dict) else copy.deepcopy(value)
    return out


def latest_stamp(name, runs_dir="runs"):
    """The stamp of the newest earlier launch of experiment `name` (None if there is none)."""
    found = sorted(d.name[: -len(name) - 1] for d in (Path(runs_dir) / "_experiments").glob(f"*-{name}")
                   if d.name[:-len(name) - 1].replace("-", "").isdigit())
    return found[-1] if found else None


def plan(spec_path, runs_dir="runs", stamp=None):
    """stamp: start time of this launch (default now); pass an earlier one to continue or summarise that launch."""
    spec = tomllib.loads(Path(spec_path).read_text(encoding="utf-8"))
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M")
    base = config.load_raw(spec["base"])
    if "pop_size" in spec:                  # the spec-wide default; a variant may still override it
        base["neat"]["pop_size"] = spec["pop_size"]
    out_dir = Path(runs_dir) / "_experiments" / f"{stamp}-{spec['name']}"
    runs = []
    for seed in spec.get("seeds", [1]):          # seed-major: a stopped experiment still compares evenly
        for variant in spec["variant"]:
            overrides = {k: v for k, v in variant.items() if isinstance(v, dict)}
            raw = merge(base, overrides)
            raw["name"] = f"{stamp}-{spec['name']}-{variant['name']}-s{seed}"
            raw.setdefault("train", {}).update(
                seed=seed, generations=spec["generations"], workers=spec.get("workers", 12),
                runs_dir=runs_dir)
            runs.append({"variant": variant["name"], "seed": seed, "raw": raw,
                         "config_path": out_dir / f"{variant['name']}-s{seed}.toml",
                         "run_dir": Path(runs_dir) / raw["name"]})
    return spec, out_dir, runs


def _rows(run_dir):
    f = Path(run_dir) / "metrics.csv"
    if not f.exists():
        return []
    seen = {}
    for row in csv.DictReader(f.open()):
        seen.setdefault(int(row["generation"]), row)       # first row per generation wins
    return [seen[g] for g in sorted(seen)]


def summarize_run(run_dir):
    rows = _rows(run_dir)
    out = {"gens": len(rows), "best_fitness": None, "best_x": None, "cleared": False,
           "gens_to_clear": None, "frames_to_clear": None, "clear_frames": None, "total_frames": None}
    if not rows:
        return out
    out["best_fitness"] = max(float(r["best"]) for r in rows)
    out["best_x"] = max(int(r["max_x"]) for r in rows)
    out["total_frames"] = int(float(rows[-1].get("total_frames") or 0)) or None
    for r in rows:
        rec = json.loads((Path(run_dir) / "best" / f"gen_{int(r['generation']):04d}.json").read_text())
        if rec["cleared"]:
            out["cleared"] = True
            out["gens_to_clear"] = int(r["generation"]) + 1
            out["frames_to_clear"] = int(float(r.get("total_frames") or 0)) or None
            out["clear_frames"] = len(rec["actions"]) * 4
            break
    return out


def aggregate(runs):
    by_variant = {}
    for run in runs:
        by_variant.setdefault(run["variant"], []).append(summarize_run(run["run_dir"]))
    table = []
    for name, items in by_variant.items():
        def mean(key):
            vals = [i[key] for i in items if i[key] is not None]
            return sum(vals) / len(vals) if vals else None
        table.append({"variant": name, "seeds": len(items), "clears": sum(i["cleared"] for i in items),
                      "best_x": mean("best_x"), "best_fitness": mean("best_fitness"),
                      "frames_to_clear": mean("frames_to_clear"), "total_frames": mean("total_frames")})
    table.sort(key=lambda r: (-(r["best_x"] or 0), r["frames_to_clear"] or 1e18))
    return table


def format_table(table):
    def f(v, spec):
        return "-" if v is None else format(v, spec)
    lines = ["| variant | seeds | cleared | mean best x | mean best fitness | mean frames to clear | mean total frames |",
             "|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['variant']} | {r['seeds']} | {r['clears']}/{r['seeds']} | {f(r['best_x'], '.0f')} | "
                     f"{f(r['best_fitness'], '.0f')} | {f(r['frames_to_clear'], ',.0f')} | {f(r['total_frames'], ',.0f')} |")
    return "\n".join(lines)


def run_experiment(spec_path, runs_dir="runs", summary_only=False, stamp=None):
    if summary_only and not stamp:         # rebuild the table of the latest launch
        stamp = latest_stamp(tomllib.loads(Path(spec_path).read_text(encoding="utf-8"))["name"], runs_dir)
    spec, out_dir, runs = plan(spec_path, runs_dir, stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, run in enumerate(runs, 1):
        done = len(_rows(run["run_dir"])) >= spec["generations"]
        if summary_only or done:
            print(f"[{i}/{len(runs)}] {run['run_dir'].name}: {'done already' if done else 'skipped'}", flush=True)
            continue
        run["config_path"].write_text(config.dump(run["raw"]))
        log = out_dir / f"{run['variant']}-s{run['seed']}.log"
        print(f"[{i}/{len(runs)}] training {run['run_dir'].name} ...", flush=True)
        # a fresh process per run: one emulator per process, and no state leaks between variants
        subprocess.run([sys.executable, "-W", "ignore", "-m", "smwai.cli", "train", "--config",
                        str(run["config_path"]), "--run-dir", str(run["run_dir"])],
                       stdout=log.open("w"), stderr=subprocess.STDOUT, check=False)
        s = summarize_run(run["run_dir"])
        print(f"      gens {s['gens']}  best x {s['best_x']}  cleared {s['cleared']}  "
              f"elapsed {time.time() - t0:.0f}s", flush=True)
    table = aggregate(runs)
    text = f"# Experiment {spec['name']}\n\nbase: {spec['base']}, {spec['generations']} generations, " \
           f"seeds {spec.get('seeds', [1])}\n\n{format_table(table)}\n"
    (out_dir / "summary.md").write_text(text)
    (out_dir / "summary.json").write_text(json.dumps(table, indent=1))
    print("\n" + format_table(table))
    return table
