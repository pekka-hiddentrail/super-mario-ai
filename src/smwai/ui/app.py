"""Streamlit dashboard: training curves, start/stop runs with edited settings, replay and path map."""
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
    return sorted((p.name for p in RUNS.glob("*") if (p / "metrics.csv").exists()), reverse=True)


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
        info["file"] = f
        out.append(info)
    return out


def start_training(cfg_path, workers, generations):
    PROCS.mkdir(parents=True, exist_ok=True)
    name = Path(cfg_path).stem
    log = PROCS / f"{name}.log"
    cmd = [sys.executable, "-m", "smwai.cli", "train", "--config", str(cfg_path),
           "--workers", str(workers), "--generations", str(generations)]
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                         start_new_session=True)
    (PROCS / f"{name}.json").write_text(json.dumps(
        {"name": name, "pid": p.pid, "config": str(cfg_path), "log": str(log)}))


def viewer_frame(path, height):
    components.iframe(f"http://localhost:{VIEWER_PORT}{path}", height=height, scrolling=True)


# ---------- layout ----------
ensure_viewer()
st.title("Super Mario World AI")
tab_runs, tab_train, tab_replay, tab_map = st.tabs(["Runs", "Train", "Replay", "Path map"])

with tab_runs:
    names = run_names()
    if not names:
        st.info("No runs yet. Start one in the Train tab.")
    else:
        picked = st.multiselect("Runs to compare", names, default=names[:2])
        live = st.toggle("Auto-refresh every 10 s", value=True)

        @st.fragment(run_every=10 if live else None)
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
    bases = sorted((ROOT / "configs").glob("*.toml"))
    base = st.selectbox("Base config", bases, format_func=lambda p: p.name)
    raw = config.load_raw(base)
    with st.form("train"):
        name = st.text_input("Run name", value=f"{raw.get('name', 'run')}-custom")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Training**")
            generations = st.number_input("Generations", 1, 10000, int(raw["train"].get("generations", 100)))
            workers = st.number_input("Parallel emulators", 1, 32, int(raw["train"].get("workers", 4)))
            raw["neat"]["pop_size"] = st.number_input("Population", 10, 2000, int(raw["neat"]["pop_size"]))
            raw["env"]["stuck_frames"] = st.number_input("Give up after N frames without progress", 30, 2000,
                                                          int(raw["env"]["stuck_frames"]))
        with c2:
            st.markdown("**Reward**")
            for key, label in [("progress", "Per pixel of progress"), ("clear_bonus", "Clear bonus"),
                               ("time_penalty", "Penalty per frame"), ("death_penalty", "Death penalty"),
                               ("powerup_bonus", "Power-up bonus"), ("coin_bonus", "Coin bonus")]:
                raw["reward"][key] = st.number_input(label, value=float(raw["reward"].get(key, 0.0)), step=0.1,
                                                     format="%.2f")
        with c3:
            st.markdown("**NEAT**")
            for key, label in [("conn_add_prob", "Add connection"), ("conn_delete_prob", "Delete connection"),
                               ("node_add_prob", "Add node"), ("node_delete_prob", "Delete node"),
                               ("weight_mutate_rate", "Mutate weights"), ("survival_threshold", "Survival fraction")]:
                raw["neat"][key] = st.number_input(label, 0.0, 1.0, float(raw["neat"][key]), step=0.05)
            raw["neat"]["compatibility_threshold"] = st.number_input(
                "Species threshold (lower = more species)", 0.5, 20.0, float(raw["neat"]["compatibility_threshold"]), step=0.5)
            raw["neat"]["max_stagnation"] = st.number_input("Drop species after N stale generations", 1, 200,
                                                             int(raw["neat"]["max_stagnation"]))
        go = st.form_submit_button("Start training", type="primary")
    if go:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name).strip("-") or "run"
        raw["name"] = safe
        raw["train"]["generations"], raw["train"]["workers"] = int(generations), int(workers)
        out = ROOT / "configs" / "custom" / f"{safe}.toml"
        out.parent.mkdir(exist_ok=True)
        out.write_text(config.dump(raw))
        start_training(out, int(workers), int(generations))
        st.success(f"Started {safe}. Watch it in the Runs tab (the first generation takes a minute).")

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

with tab_map:
    names = [n for n in run_names() if (RUNS / n / "best").exists()]
    if names:
        viewer_frame(f"/map?run={st.selectbox('Run', names, key='mp')}", 800)
