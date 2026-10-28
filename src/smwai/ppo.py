"""PPO (stable-baselines3) on the same environment, logging into the NEAT-style run folder.

Each "generation" in metrics.csv is one evaluation round: every `eval_every` training steps the
current policy plays one deterministic episode, which is recorded (buttons + path) exactly like a
NEAT generation's best run, so the replay viewer, path map and dashboard curves work unchanged.
"""
import json
import shutil
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import train
from .env import BUTTONS, SmwEnv

# Discrete actions: named button combinations (PPO picks one per step)
COMBOS = {
    "noop": [], "right": ["RIGHT"], "right_jump": ["RIGHT", "B"], "right_run": ["RIGHT", "Y"],
    "right_run_jump": ["RIGHT", "Y", "B"], "right_spin": ["RIGHT", "A"], "jump": ["B"],
    "left": ["LEFT"], "left_jump": ["LEFT", "B"], "down": ["DOWN"],
}
COMBO_MASKS = [sum(1 << BUTTONS.index(b) for b in buttons) for buttons in COMBOS.values()]


def _bits(mask):
    return np.array([(mask >> i) & 1 for i in range(len(BUTTONS))], dtype=bool)


class ComboEnv(gym.Env):
    """SmwEnv with discrete button combos, float observations and a scaled reward."""

    def __init__(self, env_cfg, reward_scale):
        self.inner = SmwEnv(env_cfg)
        self.scale = reward_scale
        self.action_space = spaces.Discrete(len(COMBO_MASKS))
        n = self.inner.observation_space.shape[0]
        self.observation_space = spaces.Box(-16.0, 16.0, (n,), dtype=np.float32)
        self.ep_return = 0.0

    def reset(self, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed, options=options)
        self.ep_return = 0.0
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, term, trunc, info = self.inner.step(_bits(COMBO_MASKS[int(action)]))
        self.ep_return += reward
        if term or trunc:
            info["ep_return"], info["ep_max_x"] = self.ep_return, info["max_x"]
        return obs.astype(np.float32), reward * self.scale, term, trunc, info


def _make_env(env_cfg, scale):
    def factory():
        return ComboEnv(env_cfg, scale)
    return factory


def play_episode(model, env_cfg):
    """One deterministic episode in this process; returns (return, info, button-mask list, path)."""
    train.use_config(env_cfg)
    env = train._get_env()
    obs, info = env.reset()
    total, masks, path = 0.0, [], [[info["x"], info["y"]]]
    while True:
        action, _ = model.predict(obs.astype(np.float32), deterministic=True)
        mask = COMBO_MASKS[int(action)]
        masks.append(mask)
        obs, reward, term, trunc, info = env.step(_bits(mask))
        total += reward
        path.append([info["x"], info["y"]])
        if term or trunc:
            info["path"] = path
            return total, info, masks


def train_ppo(cfg, config_path, run_dir=None, steps=None, n_envs=None):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    p = cfg.ppo
    steps = steps or p["total_steps"]
    n_envs = n_envs or p["n_envs"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir or Path(cfg.runs_dir) / f"{cfg.name}-{stamp}")
    (run_dir / "best").mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.toml")
    torch.set_num_threads(2)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    # the worker processes must be created before this process opens its own emulator
    vec = SubprocVecEnv([_make_env(cfg.env, p["reward_scale"]) for _ in range(n_envs)], start_method="fork")
    csv = run_dir / "metrics.csv"
    csv.write_text("generation,best,mean,species,max_x,seconds,deterministic,frames,total_frames\n")

    class Recorder(BaseCallback):
        def __init__(self):
            super().__init__()
            self.returns, self.round, self.last_eval, self.t0 = [], 0, 0, time.time()
            self.best = -np.inf

        def _on_step(self):
            for info in self.locals["infos"]:
                if "ep_return" in info:
                    self.returns.append(info["ep_return"])
            if self.num_timesteps - self.last_eval >= p["eval_every"]:
                self.last_eval = self.num_timesteps
                self._evaluate()
            return True

        def _evaluate(self):
            ret, info, masks = play_episode(self.model, cfg.env)
            n = self.round
            (run_dir / "best" / f"gen_{n:04d}.json").write_text(json.dumps({
                "generation": n, "fitness": ret, "max_x": info["max_x"], "cleared": info["cleared"],
                "buttons": BUTTONS, "actions": masks, "path": info["path"],
                "events": info.get("events", []),
                "timesteps": self.num_timesteps}))
            if ret > self.best:
                self.best = ret
                self.model.save(str(run_dir / "best_ever"))
            mean = float(np.mean(self.returns[-30:])) if self.returns else 0.0
            secs = time.time() - self.t0
            with open(csv, "a") as f:
                fs = cfg.env.frame_skip
                f.write(f"{n},{ret:.1f},{mean:.1f},0,{info['max_x']},{secs:.1f},1,"
                        f"{p['eval_every'] * fs},{self.num_timesteps * fs}\n")
            print(f"round {n:4d}  steps {self.num_timesteps:8d}  eval {ret:8.1f}  train mean {mean:8.1f}  "
                  f"max_x {info['max_x']:5d}  {'CLEARED' if info['cleared'] else ''}", flush=True)
            self.round += 1

    model = PPO("MlpPolicy", vec, learning_rate=p["learning_rate"], n_steps=p["n_steps"],
                batch_size=p["batch_size"], n_epochs=p["n_epochs"], gamma=p["gamma"],
                ent_coef=p["ent_coef"], clip_range=p["clip_range"], seed=cfg.seed, device="cpu",
                policy_kwargs={"net_arch": p["net_arch"]}, verbose=0)
    print(f"run dir: {run_dir}  envs: {n_envs}  steps: {steps}", flush=True)
    try:
        model.learn(total_timesteps=steps, callback=Recorder())
    except KeyboardInterrupt:
        print("\ninterrupted; results so far are in", run_dir)
    finally:
        vec.close()
    return run_dir
