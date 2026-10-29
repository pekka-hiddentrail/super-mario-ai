"""Streamlit dashboard: training curves, start/stop runs, replay, path map and videos."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from smwai import config

ROOT = Path(os.environ.get("SMWAI_ROOT", ".")).resolve()
RUNS = ROOT / "runs"
PROCS = RUNS / "_procs"
VIEWER_PORT = int(os.environ.get("SMWAI_VIEWER_PORT", "8765"))

st.set_page_config(page_title="SMW AI", layout="wide")


# ---------- helpers ----------
@st.cache_resource
def ensure_viewer():
    """Start the replay/map server once (separate process: it owns an emulator)."""
    import socket
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", VIEWER_PORT)) == 0:
            return "already running"
    subprocess.Popen([sys.executable, "-m", "smwai.cli", "view", "--runs-dir", str(RUNS),
                      "--port", str(VIEWER_PORT)], cwd=ROOT,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return "started"


def run_names():
    """Runs with metrics, newest (most recently active) first."""
    runs = [p for p in RUNS.glob("*") if (p / "metrics.csv").exists()]
    return [p.name for p in sorted(runs, key=lambda p: (p / "metrics.csv").stat().st_mtime, reverse=True)]


def metrics(name):
    return pd.read_csv(RUNS / name / "metrics.csv")


def is_running(name):
    f = RUNS / name / "metrics.csv"
    return time.time() - f.stat().st_mtime < 180


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def procs():
    out = []
    for f in PROCS.glob("*.json"):
        info = json.loads(f.read_text())
        info["alive"] = alive(info["pid"])
        out.append(info)
    return out


def start_training(cfg_path, workers, generations, algo="neat"):
    PROCS.mkdir(parents=True, exist_ok=True)
    name = Path(cfg_path).stem
    log = PROCS / f"{name}.log"
    if algo == "ppo":      # here `workers` = parallel emulators, `generations` = total training steps
        cmd = [sys.executable, "-m", "smwai.cli", "train-ppo", "--config", str(cfg_path),
               "--envs", str(workers), "--steps", str(generations)]
    else:
        cmd = [sys.executable, "-m", "smwai.cli", "train", "--config", str(cfg_path),
               "--workers", str(workers), "--generations", str(generations)]
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                         start_new_session=True)
    (PROCS / f"{name}.json").write_text(json.dumps(
        {"name": name, "pid": p.pid, "config": str(cfg_path), "log": str(log)}))


def viewer_frame(path, height):
    components.iframe(f"http://localhost:{VIEWER_PORT}{path}", height=height, scrolling=True)


def param_inputs(raw, prefix):
    """Number inputs for the tunable settings; edits `raw` in place. Returns (generations, workers)."""
    c1, c2, c3 = st.columns(3)
    generations, workers = raw["train"].get("generations", 100), raw["train"].get("workers", 4)
    with c1:
        st.markdown("**Training**")
        if True:
            generations = st.number_input("Generations", 1, 10000, int(generations), key=prefix + "gens")
            workers = st.number_input("Parallel emulators", 1, 32, int(workers), key=prefix + "workers")
            raw["neat"]["pop_size"] = st.number_input("Population", 10, 2000, int(raw["neat"]["pop_size"]),
                                                      key=prefix + "pop")
        if True:
            modes = ["mixed", "channels", "collect", "physics", "explore", "ride", "ride2"]
            raw["env"]["inputs"] = st.selectbox("Network inputs", modes, key=prefix + "inputs",
                                                index=modes.index(raw["env"].get("inputs", "mixed")),
                                                help="mixed: one tile grid. channels: solid / enemy / item grids plus water, "
                                                     "Yoshi, power-up. collect: + coins. physics: + speeds and what each "
                                                     "standard jump would do. explore: + a pipe grid. ride: + a moving-platform grid "
                                                     "(bridges, swinging blocks count as ground, not enemies). ride2: the same, and jumps "
                                                     "from below a platform bump the head.")
            raw["env"]["room_progress"] = st.checkbox(
                "Progress per room (pipes and bonus rooms)", value=bool(raw["env"].get("room_progress", False)),
                key=prefix + "rooms", help="Each room keeps its own furthest x, so a bonus room is not a dead end.")
            window = st.text_input("View window [left, right, up, down] in tiles (empty = 6 every way)",
                                   value=",".join(map(str, raw["env"].get("window") or [])), key=prefix + "window")
            parts = [p.strip() for p in window.replace("[", "").replace("]", "").split(",") if p.strip()]
            if len(parts) == 4 and all(p.lstrip("-").isdigit() for p in parts):
                raw["env"]["window"] = [int(p) for p in parts]
            else:
                raw["env"].pop("window", None)
            adapt = raw.setdefault("adapt", {})
            adapt["enabled"] = st.checkbox("Adaptive mutation (mutate harder while stuck)",
                                           value=bool(adapt.get("enabled", False)), key=prefix + "adapt")
            if adapt["enabled"]:
                adapt["stall_generations"] = int(st.number_input("...after N generations without a real gain", 1, 50,
                                                                  int(adapt.get("stall_generations", 3)), key=prefix + "astall"))
                adapt["min_improvement"] = float(st.number_input("...a gain counts from (fitness)", 0.0, 1000.0,
                                                                 float(adapt.get("min_improvement", 20.0)), key=prefix + "amin"))
        raw["env"]["stuck_frames"] = st.number_input("Give up after N frames without progress", 30, 2000,
                                                      int(raw["env"]["stuck_frames"]), key=prefix + "stuck")
    with c2:
        st.markdown("**Reward**")
        for key, label in [("progress", "Per pixel of progress"), ("clear_bonus", "Clear bonus"),
                           ("time_penalty", "Penalty per frame"), ("death_penalty", "Death penalty"),
                           ("powerup_bonus", "Power-up bonus"), ("coin_bonus", "Coin bonus"),
                           ("yoshi_coin_bonus", "Big (Yoshi) coin bonus"), ("block_bonus", "Box hit bonus"),
                           ("goal_bar_bonus", "Goal bar bonus (per star: higher hit = more)"),
                           ("height_bonus", "Climb bonus (per pixel higher than ever before)"),
                           ("pipe_bonus", "Pipe bonus (entering a pipe, once per pipe)"),
                           ("water_penalty", "Water penalty (per frame in water)"),
                           ("duck_penalty", "Duck penalty (per frame crouching or sliding)"),
                           ("speed_bonus", "Speed bonus (per px/frame running)"),
                           ("clear_time_bonus", "Fast-clear bonus (shrinks with time)"),
                           ("clear_time_ref", "...reaches zero at N frames")]:
            raw["reward"][key] = st.number_input(label, value=float(raw["reward"].get(key, 3000.0 if key == "clear_time_ref" else 0.0)),
                                                 step=0.1, format="%.2f", key=prefix + key)
    with c3:
        st.markdown("**NEAT**")
        for key, label in [("conn_add_prob", "Add connection"), ("conn_delete_prob", "Delete connection"),
                           ("node_add_prob", "Add node"), ("node_delete_prob", "Delete node"),
                           ("weight_mutate_rate", "Mutate weights"), ("survival_threshold", "Survival fraction")]:
            raw["neat"][key] = st.number_input(label, 0.0, 1.0, float(raw["neat"][key]), step=0.05,
                                               key=prefix + key)
        raw["neat"]["compatibility_threshold"] = st.number_input(
            "Species threshold (lower = more species)", 0.5, 20.0, float(raw["neat"]["compatibility_threshold"]),
            step=0.5, key=prefix + "compat")
        raw["neat"]["max_stagnation"] = st.number_input("Drop species after N stale generations", 1, 200,
                                                         int(raw["neat"]["max_stagnation"]), key=prefix + "stag")
    return int(generations), int(workers)


# ---------- layout ----------
ensure_viewer()
st.title("Super Mario World AI")
tab_runs, tab_train, tab_replay, tab_map, tab_video = st.tabs(
    ["Runs", "Train", "Replay", "Path map", "Videos"])

with tab_runs:
    names = run_names()
    if not names:
        st.info("No runs yet. Start one in the Train tab.")
    else:
        picked = st.multiselect("Runs to compare", names, default=names[:2])
        auto = st.toggle("Auto-refresh every 10 s", value=True)

        @st.fragment(run_every=10 if auto else None)
        def curves():
            if not picked:
                return
            frames = {n: metrics(n).set_index("generation") for n in picked}
            for col, title in [("best", "Best fitness per generation"), ("mean", "Mean fitness"),
                               ("max_x", "Furthest x of the best run"), ("species", "Species"),
                               ("seconds", "Seconds per generation")]:
                st.subheader(title)
                st.line_chart(pd.DataFrame({n: f[col] for n, f in frames.items()}))
            st.subheader("Latest generation")
            st.dataframe(pd.DataFrame([{"run": n, "running": is_running(n), "generation": int(f.index[-1]),
                                        **f.iloc[-1][["best", "mean", "max_x", "species"]].to_dict()}
                                       for n, f in frames.items()]), hide_index=True)
        curves()
        with st.expander("Settings used by a run"):
            which = st.selectbox("Run", picked or names, key="cfgrun")
            st.code((RUNS / which / "config.toml").read_text(), language="toml")

with tab_train:
    st.caption("Pick a base config, change the values, and start a run. Each run keeps its own copy of the settings.")
    bases = sorted(p for p in (ROOT / "configs").glob("*.toml") if not p.name.startswith("ppo"))
    base = st.selectbox("Base config", bases, format_func=lambda p: p.name)
    raw = config.load_raw(base)
    with st.form("train"):
        name = st.text_input("Run name", value=f"{raw.get('name', 'run')}-custom")
        generations, workers = param_inputs(raw, "t_")
        go = st.form_submit_button("Start training", type="primary")
    if go:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name).strip("-") or "run"
        raw["name"] = safe
        raw["train"]["generations"], raw["train"]["workers"] = generations, workers
        out = ROOT / "configs" / "custom" / f"{safe}.toml"
        out.parent.mkdir(exist_ok=True)
        out.write_text(config.dump(raw))
        start_training(out, workers, generations)
        st.success(f"Started {safe}. Watch it in the Runs tab (the first generation takes a minute).")

    with st.expander("Start a PPO run instead (stable-baselines3, CPU)"):
        ppo_bases = sorted((ROOT / "configs").glob("ppo*.toml"))
        if not ppo_bases:
            st.write("No configs/ppo*.toml found.")
        else:
            pbase = st.selectbox("PPO base config", ppo_bases, format_func=lambda p: p.name, key="pbase")
            praw = config.load_raw(pbase)
            pp = {**config.PPO_DEFAULTS, **praw.get("ppo", {})}
            with st.form("ppo"):
                pname = st.text_input("Run name", value=f"{praw.get('name', 'ppo')}-custom", key="pname")
                c1, c2, c3 = st.columns(3)
                psteps = c1.number_input("Training steps", 4096, 20_000_000, int(pp["total_steps"]), step=50_000)
                penvs = c1.number_input("Parallel emulators", 1, 32, int(pp["n_envs"]), key="penvs")
                pp["learning_rate"] = c2.number_input("Learning rate", 1e-6, 1e-1, float(pp["learning_rate"]),
                                                      format="%.5f", step=1e-4)
                pp["ent_coef"] = c2.number_input("Entropy bonus (exploration)", 0.0, 0.5, float(pp["ent_coef"]), step=0.005,
                                                 format="%.3f")
                pp["gamma"] = c3.number_input("Discount (gamma)", 0.8, 0.9999, float(pp["gamma"]), format="%.4f")
                pp["reward_scale"] = c3.number_input("Reward scale", 0.0001, 1.0, float(pp["reward_scale"]),
                                                     format="%.4f")
                for key, label in [("progress", "Per pixel of progress"), ("clear_bonus", "Clear bonus"),
                                   ("powerup_bonus", "Power-up bonus"), ("coin_bonus", "Coin bonus"),
                                   ("time_penalty", "Penalty per frame"), ("death_penalty", "Death penalty")]:
                    praw["reward"][key] = c2.number_input(label, value=float(praw["reward"].get(key, 0.0)),
                                                          step=0.1, format="%.2f", key="p_" + key)
                pgo = st.form_submit_button("Start PPO", type="primary")
            if pgo:
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in pname).strip("-") or "ppo"
                praw["name"], praw["ppo"] = safe, pp
                out = ROOT / "configs" / "custom" / f"{safe}.toml"
                out.parent.mkdir(exist_ok=True)
                out.write_text(config.dump(praw))
                start_training(out, int(penvs), int(psteps), algo="ppo")
                st.success(f"Started {safe}. Each row in the Runs curves is one evaluation round (not a NEAT generation).")

    st.subheader("Runs started from here")
    for info in procs():
        c1, c2, c3 = st.columns([3, 1, 1])
        c1.write(f"**{info['name']}** - {'running' if info['alive'] else 'finished'}")
        if info["alive"] and c2.button("Stop", key=f"stop-{info['name']}"):
            os.kill(info["pid"], signal.SIGINT)   # trainer saves what it has and exits
            st.rerun()
        if c3.checkbox("Log", key=f"log-{info['name']}"):
            st.code("\n".join(Path(info["log"]).read_text().splitlines()[-15:]) or "(empty)")

with tab_replay:
    names = [n for n in run_names() if (RUNS / n / "best").exists()]
    if names:
        viewer_frame(f"/?run={st.selectbox('Run', names, key='rp')}", 1050)

with tab_video:
    st.caption("One video per run: the generation with the highest fitness at the time of rendering. "
               "Runs still in progress get a new video when you render again with 'Re-render'.")
    c1, c2 = st.columns(2)
    force = c2.checkbox("Re-render existing videos", value=False)
    if c1.button("Render best-run videos"):
        with st.spinner("Rendering (about a minute per run)..."):
            cmd = [sys.executable, "-m", "smwai.cli", "video", "--runs-dir", str(RUNS)] + (["--force"] if force else [])
            res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        st.code((res.stdout + res.stderr)[-1500:] or "(no output)")
    viewer_frame("/videos", 1400)

with tab_map:
    names = [n for n in run_names() if (RUNS / n / "best").exists()]
    if names:
        viewer_frame(f"/map?run={st.selectbox('Run', names, key='mp')}", 800)
