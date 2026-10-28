"""Gymnasium environment for Super Mario World on stable-retro.

Observation: (2r+1, 2r+1) int8 grid around Mario (1 solid, -1 enemy, 2 power-up).
Action: MultiBinary over BUTTONS (each output of a NEAT network maps to one button).
Reward: progress to the right (new max x) plus optional bonuses, all configurable.
"""
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import stable_retro as retro
from gymnasium import spaces

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


@dataclass
class EnvConfig:
    levels: list = field(default_factory=lambda: ["YoshiIsland2"])
    frame_skip: int = 4
    radius: int = 6
    stuck_frames: int = 120        # no new max x for this many frames -> truncate
    max_frames: int = 20000
    reward: RewardConfig = field(default_factory=RewardConfig)


class SmwEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config=None, render_mode=None):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        size = self.cfg.radius * 2 + 1
        self.observation_space = spaces.Box(-1, 2, (size, size), dtype=np.int8)
        self.action_space = spaces.MultiBinary(len(BUTTONS))
        self._env = None
        self._level = None
        self._pad = None
        self.frame = None

    def _open(self, level):
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
        return st, R.tile_grid(ram, st, self.cfg.radius)

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
        self.state, grid = self._read()
        rc = self.cfg.reward
        reward = -rc.time_penalty * self.cfg.frame_skip
        if self.state.x > self.max_x:
            reward += rc.progress * (self.state.x - self.max_x)
            self.max_x = self.state.x
            self.stuck = 0
        else:
            self.stuck += self.cfg.frame_skip
        reward += rc.powerup_bonus * max(0, self.state.powerup - prev.powerup)
        reward += rc.coin_bonus * max(0, self.state.coins - prev.coins)
        died = self.state.dying or self.state.lives < prev.lives
        cleared = self.state.level_cleared
        if died:
            reward -= rc.death_penalty
        if cleared:
            reward += rc.clear_bonus
        terminated = died or cleared
        truncated = self.stuck >= self.cfg.stuck_frames or self.frames >= self.cfg.max_frames
        info = self._info()
        info.update(died=died, cleared=cleared)
        return grid, float(reward), terminated, truncated, info

    def render(self):
        return self.frame

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
