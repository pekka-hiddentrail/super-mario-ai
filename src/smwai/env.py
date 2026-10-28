"""Gymnasium environment for Super Mario World on stable-retro.

Observation: (2r+1, 2r+1) int8 grid around Mario (1 solid, -1 enemy, 2 power-up).
Action: MultiBinary over BUTTONS (each output of a NEAT network maps to one button).
Reward: progress to the right (new max x) plus optional bonuses, all configurable.
"""
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import jumpview
from . import ram as R

GAME = "SuperMarioWorld-Snes-v0"
BUTTONS = ["A", "B", "X", "Y", "UP", "DOWN", "LEFT", "RIGHT"]


@dataclass
class RewardConfig:
    progress: float = 1.0          # per pixel of new max x
    time_penalty: float = 0.0      # per emulator frame
    death_penalty: float = 0.0
    clear_bonus: float = 1000.0
    powerup_bonus: float = 0.0     # per power-up state increase
    coin_bonus: float = 0.0
    speed_bonus: float = 0.0       # per (pixel/frame of rightward speed) per frame, i.e. rewards running
    clear_time_bonus: float = 0.0  # on clear: this * max(0, 1 - frames / clear_time_ref); faster = more
    clear_time_ref: float = 3000.0 # frames at which the clear-time bonus has shrunk to zero
    yoshi_coin_bonus: float = 0.0  # per big (Yoshi) coin
    block_bonus: float = 0.0       # per ? block hit (any block that turns into a used block)
    goal_bar_bonus: float = 0.0    # per bonus star won at the goal bar: hitting the bar higher pays more


@dataclass
class EnvConfig:
    levels: list = field(default_factory=lambda: ["YoshiIsland2"])
    frame_skip: int = 4
    radius: int = 6                # view: this many tiles every way around Mario ...
    window: list = None            # ... or [left, right, up, down] in tiles (overrides radius), e.g. [4, 8, 4, 8]
    inputs: str = "mixed"          # mixed: one tile grid | channels: solid/enemy/item + scalars | collect: + coins | physics: + jump options
    stuck_frames: int = 120       # no new max x for this many frames -> truncate
    max_frames: int = 20000
    reward: RewardConfig = field(default_factory=RewardConfig)


def view(cfg):
    """The (left, right, up, down) window around Mario that the network sees, in tiles."""
    return R.window(cfg.window if cfg.window else cfg.radius)


def input_layout(cfg):
    """Ordered input groups as (name, kind); kind 'grid' = (2r+1)^2 cells, 'scalar' = one value."""
    scalars = [("in_water", "scalar"), ("on_yoshi", "scalar"), ("powerup", "scalar")]
    if cfg.inputs == "physics":     # collect + Mario's speeds + what each standard jump would do (jumpview.py)
        motion = [("x_speed", "scalar"), ("y_speed", "scalar"), ("on_ground", "scalar")]
        return ([("solid", "grid"), ("enemy", "grid"), ("item", "grid"), ("coin", "grid")] + scalars + motion
                + [(name, "scalar") for name in jumpview.names()])
    if cfg.inputs == "collect":
        return [("solid", "grid"), ("enemy", "grid"), ("item", "grid"), ("coin", "grid")] + scalars
    if cfg.inputs == "channels":
        return [("solid", "grid"), ("enemy", "grid"), ("item", "grid")] + scalars
    return [("tiles", "grid")]


def n_inputs(cfg):
    rows, cols = R.grid_shape(view(cfg))
    cells = rows * cols
    return sum(cells if kind == "grid" else 1 for _, kind in input_layout(cfg))


def build_observation(ram, state, cfg):
    """Flat int8 vector in input_layout order."""
    if cfg.inputs in ("channels", "collect", "physics"):
        collect = cfg.inputs in ("collect", "physics")
        enemy, item = R.sprite_grids(state, view(cfg), legacy=not collect)
        parts = [R.solid_grid(ram, state, view(cfg)).ravel(), enemy.ravel(), item.ravel()]
        if collect:
            parts.append(R.coin_grid(ram, state, view(cfg)).ravel())
        parts.append(np.array([state.in_water, state.on_yoshi, min(state.powerup, 3)], dtype=np.int8))
        if cfg.inputs == "physics":
            parts.append(np.array(jumpview.motion_scalars(state), dtype=np.int8))
            parts.append(jumpview.features(ram, state))
        return np.concatenate(parts).astype(np.int8)
    return R.tile_grid(ram, state, view(cfg)).ravel()


class SmwEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config=None, render_mode=None):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.layout = input_layout(self.cfg)
        self.observation_space = spaces.Box(-16, 16, (n_inputs(self.cfg),), dtype=np.int8)
        self.action_space = spaces.MultiBinary(len(BUTTONS))
        self._env = None
        self._level = None
        self._pad = None
        self.frame = None
        self.capture = None            # set to a list to collect every emulator frame (video)
        self.events = []

    def _open(self, level):
        import stable_retro as retro  # Linux/WSL only; imported lazily so config/tests work anywhere

        if self._env is not None and self._level == level:
            return
        if self._env is not None:
            self._env.close()
        self._env = retro.make(GAME, state=level, use_restricted_actions=retro.Actions.ALL,
                               render_mode=None)
        self._level = level
        names = list(self._env.buttons)
        self._pad = [names.index(b) for b in BUTTONS]

    def _read(self):
        ram = R.wram(self._env.unwrapped.get_ram())
        st = R.read_state(ram)
        return st, build_observation(ram, st, self.cfg)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get("level_index", 0)
        self._open(self.cfg.levels[index])
        self.frame = self._env.reset()[0]
        for _ in range(2):  # let the first frames settle so RAM is valid
            self.frame = self._env.step(np.zeros(12, dtype=np.int8))[0]
        self.state, grid = self._read()
        self.max_x = self.state.x
        self.frames = 0
        self.stuck = 0
        self.events = []
        if self.capture is not None:
            self.capture.append(self.frame.copy())
        return grid, self._info()

    def _info(self):
        return {"x": self.state.x, "y": self.state.y, "max_x": self.max_x,
                "powerup": self.state.powerup, "frames": self.frames}

    def step(self, action):
        pad = np.zeros(12, dtype=np.int8)
        for i, on in enumerate(action):
            if on:
                pad[self._pad[i]] = 1
        prev = self.state
        for _ in range(self.cfg.frame_skip):
            self.frame = self._env.step(pad)[0]
            self.frames += 1
            if self.capture is not None:          # video recording: keep every emulator frame
                self.capture.append(self.frame.copy())
        self.state, grid = self._read()
        rc = self.cfg.reward
        reward = -rc.time_penalty * self.cfg.frame_skip
        if self.state.x > self.max_x:
            reward += rc.progress * (self.state.x - self.max_x)
            self.max_x = self.state.x
            self.stuck = 0
        else:
            self.stuck += self.cfg.frame_skip
        reward += rc.speed_bonus * max(0.0, self.state.x_speed / 16.0) * self.cfg.frame_skip
        gained = {"powerup": max(0, self.state.powerup - prev.powerup),
                  "coin": max(0, self.state.coins - prev.coins),
                  "yoshi": max(0, self.state.yoshi_coins - prev.yoshi_coins),
                  "block": max(0, self.state.used_blocks - prev.used_blocks)}
        weights = {"powerup": rc.powerup_bonus, "coin": rc.coin_bonus,
                   "yoshi": rc.yoshi_coin_bonus, "block": rc.block_bonus}
        for kind, n in gained.items():
            if n:
                reward += weights[kind] * n
                self.events.append([self.frames, kind, self.state.x, self.state.y])
                if weights[kind] > 0:
                    self.stuck = 0          # collecting something worth reward counts as progress
        died = self.state.dying or self.state.lives < prev.lives
        cleared = self.state.level_cleared
        if died:
            reward -= rc.death_penalty
        if cleared:
            reward += rc.clear_bonus
            reward += rc.goal_bar_bonus * self.state.bonus_stars
            reward += rc.clear_time_bonus * max(0.0, 1.0 - self.frames / rc.clear_time_ref)
        terminated = died or cleared
        truncated = self.stuck >= self.cfg.stuck_frames or self.frames >= self.cfg.max_frames
        info = self._info()
        info.update(died=died, cleared=cleared)
        if terminated or truncated:
            info["events"] = list(self.events)      # [frame, kind, x, y] for the path map
        return grid, float(reward), terminated, truncated, info

    def render(self):
        return self.frame

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
