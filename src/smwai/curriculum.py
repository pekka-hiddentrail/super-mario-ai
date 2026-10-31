"""One population through several levels: when a level is cleared, the same population moves on to the next.

A spec (see configs/experiments/curriculum_yi.toml) names a base config, the levels in order and, per level, the generations and
genomes. For every seed the population trains on the first level until its best run clears it (or the generations run out), then
the population NEAT saved that generation (with its network structure, species and innovation numbers) is trained on the next
level, and so on. A level that is not cleared in time ends the seed. Every level is a normal run folder,
runs/<stamp>-<name>-s<seed>-L<n>-<level>/, so videos and the viewer work as usual; the generation numbers keep counting across levels.
"""
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from . import config
from .experiment import merge, summarize_run


def level_runs(spec_path, runs_dir="runs", stamp=None):
    """The planned run of every (seed, level): name, config, run folder."""
    spec = tomllib.loads(Path(spec_path).read_text(encoding="utf-8"))
    stamp = stamp or time.strftime("%Y%m%d-%H%M")
    base = config.load_raw(spec["base"])
    if "pop_size" in spec:
        base["neat"]["pop_size"] = spec["pop_size"]
    base = merge(base, {k: v for k, v in spec.items() if k in ("env", "neat", "reward", "adapt") and isinstance(v, dict)})
    out_dir = Path(runs_dir) / "_experiments" / f"{stamp}-{spec['name']}"
    plan = []
    for seed in spec.get("seeds", [1]):
        for n, level in enumerate(spec["levels"], 1):
            raw = merge(base, {"env": {"levels": [level]}})
            raw["name"] = f"{stamp}-{spec['name']}-s{seed}-L{n}-{level}"
            raw.setdefault("train", {}).update(seed=seed, generations=spec["generations"], workers=spec.get("workers", 12), runs_dir=runs_dir)
            plan.append({"seed": seed, "n": n, "level": level, "raw": raw,
                         "config_path": out_dir / f"s{seed}-L{n}-{level}.toml", "run_dir": Path(runs_dir) / raw["name"]})
    return spec, out_dir, plan


def carry_file(run_dir):
    """The population StopOnClear saved when the level was cleared (None if it was not)."""
    found = sorted(Path(run_dir).glob("carry-*"), key=lambda p: int(p.name.split("-")[-1]))
    return found[-1] if found else None


def run_curriculum(spec_path, runs_dir="runs", stamp=None):
    spec, out_dir, plan = level_runs(spec_path, runs_dir, stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, rows = time.time(), []
    for seed in spec.get("seeds", [1]):
        previous = None
        for run in (r for r in plan if r["seed"] == seed):
            run["config_path"].write_text(config.dump(run["raw"]))
            cmd = [sys.executable, "-W", "ignore", "-m", "smwai.cli", "train", "--config", str(run["config_path"]),
                   "--run-dir", str(run["run_dir"]), "--stop-on-clear"]
            if previous:
                cmd += ["--resume", str(previous)]
            print(f"seed {seed} level {run['n']} {run['level']}: training {run['run_dir'].name} ...", flush=True)
            # a fresh process per level: one emulator per process, and the worker pool forks before an emulator exists
            subprocess.run(cmd, stdout=(out_dir / f"s{seed}-L{run['n']}-{run['level']}.log").open("w"), stderr=subprocess.STDOUT, check=False)
            s = summarize_run(run["run_dir"])
            previous = carry_file(run["run_dir"])
            rows.append({"seed": seed, "level": run["level"], "gens": s["gens"], "best_x": s["best_x"], "cleared": s["cleared"]})
            print(f"      gens {s['gens']}  best x {s['best_x']}  cleared {s['cleared']}  elapsed {time.time() - t0:.0f}s", flush=True)
            if not previous:
                break                                  # not cleared in time: this seed stops here
    lines = ["| seed | level | generations | best x | cleared |", "|---|---|---|---|---|"]
    lines += [f"| {r['seed']} | {r['level']} | {r['gens']} | {r['best_x']} | {'yes' if r['cleared'] else 'no'} |" for r in rows]
    text = f"# Curriculum {spec['name']}\n\nbase: {spec['base']}, levels {spec['levels']}, {spec['generations']} generations per level\n\n" + "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text)
    print("\n" + "\n".join(lines))
    return rows
