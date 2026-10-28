"""Command line interface: smwai train | replay."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="smwai")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train with NEAT")
    t.add_argument("--config", default="configs/level1.toml")
    t.add_argument("--generations", type=int)
    t.add_argument("--workers", type=int)
    t.add_argument("--run-dir")
    t.add_argument("--resume", help="path to a ckpt-N file in a run dir")

    pp = sub.add_parser("train-ppo", help="train with PPO (stable-baselines3)")
    pp.add_argument("--config", default="configs/ppo_level1.toml")
    pp.add_argument("--steps", type=int, help="total training steps")
    pp.add_argument("--envs", type=int, help="parallel emulators")
    pp.add_argument("--run-dir")

    r = sub.add_parser("replay", help="re-run a recorded generation headless and check it")
    r.add_argument("run_dir")
    r.add_argument("generation", type=int)

    v = sub.add_parser("view", help="start the browser replay viewer")
    v.add_argument("--runs-dir", default="runs")
    v.add_argument("--port", type=int, default=8765)

    u = sub.add_parser("ui", help="start the Streamlit dashboard")
    u.add_argument("--port", type=int, default=8501)
    u.add_argument("--viewer-port", type=int, default=8765)

    args = ap.parse_args(argv)
    if args.cmd == "ui":
        import os
        import subprocess
        import sys
        app = Path(__file__).parent / "ui" / "app.py"
        env = {**os.environ, "SMWAI_VIEWER_PORT": str(args.viewer_port), "SMWAI_ROOT": os.getcwd()}
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port",
                                str(args.port), "--server.headless", "true",
                                "--browser.gatherUsageStats", "false"], env=env)
    if args.cmd == "view":
        from .viewer.server import serve
        return serve(args.runs_dir, args.port)
    from . import config, train

    if args.cmd == "train-ppo":
        from .ppo import train_ppo
        return train_ppo(config.load(args.config), args.config, run_dir=args.run_dir,
                         steps=args.steps, n_envs=args.envs)
    if args.cmd == "train":
        cfg = config.load(args.config)
        train.train(cfg, args.config, resume=args.resume, run_dir=args.run_dir,
                    generations=args.generations, workers=args.workers)
    else:
        run = Path(args.run_dir)
        train._ENV_CFG = config.load(run / "config.toml").env
        rec = json.loads((run / "best" / f"gen_{args.generation:04d}.json").read_text())
        reward, info = train.replay_actions(rec["actions"])
        print(f"recorded fitness {rec['fitness']:.1f}, replayed {reward:.1f}, max_x {info['max_x']}")


if __name__ == "__main__":
    main()
