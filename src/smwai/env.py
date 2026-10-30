"""Gymnasium environment for Super Mario World on stable-retro.

Observation: (2r+1, 2r+1) int8 grid around Mario (1 solid, -1 enemy, 2 power-up).
Action: MultiBinary over BUTTONS (each output of a NEAT network maps to one button).
Reward: progress to the right (new max x) plus optional bonuses, all configurable.
"""
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import deathmap, jumpview
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
    water_penalty: float = 0.0     # per emulator frame spent in water (off: it is not clear yet that avoiding water is right)
    height_bonus: float = 0.0      # per pixel of new standing height: landing on something higher than ever before
    pipe_bonus: float = 0.0         # per pipe entered, down or sideways (each pipe pays once per episode)


@dataclass
class EnvConfig:
    levels: list = field(default_factory=lambda: ["YoshiIsland2"])
    frame_skip: int = 4
    radius: int = 6                # view: this many tiles every way around Mario ...
    window: list = None            # ... or [left, right, up, down] in tiles (overrides radius), e.g. [4, 8, 4, 8]
    inputs: str = "mixed"          # mixed: one tile grid | channels: solid/enemy/item + scalars | collect: + coins | physics: + jump options | explore: + pipes
    death_inputs: bool = False     # + four inputs from the death map: where earlier attempts died or got stuck (deathmap.py)
    takeoff_inputs: bool = False   # + four inputs from where the best runs took off for their fatal fall, and how (deathmap.py)
    jump_fixes: bool = False       # jump inputs: walls stop the arc, the message box is no enemy (also not in the enemy grid), and the
                                   # values stay as they were at takeoff while Mario is in the air
    run_options: bool = False      # + 'keep running' and the jump options as from full run speed (jumpview.run_names): 35 inputs
    predict_platforms: bool = False  # jump options use where moving platforms will be during the arc; + 4 inputs about the nearest platform
    cancel_opposites: bool = False  # LEFT+RIGHT cancel each other, and so do UP+DOWN (networks otherwise hold both most of the time)
    room_progress: bool = False    # measure progress per room (pipes/doors): a bonus room starts its own 'furthest x'
    stuck_frames: int = 120       # no new max x for this many frames -> truncate
    max_frames: int = 20000
    reward: RewardConfig = field(default_factory=RewardConfig)


def cancel_opposites(action):
    """Button vector with LEFT+RIGHT and UP+DOWN pairs switched off (neither is pressed when both are)."""
    action = np.array(action, dtype=bool)
    for a, b in ((BUTTONS.index("LEFT"), BUTTONS.index("RIGHT")), (BUTTONS.index("UP"), BUTTONS.index("DOWN"))):
        if action[a] and action[b]:
            action[a] = action[b] = False
    return action


def view(cfg):
    """The (left, right, up, down) window around Mario that the network sees, in tiles."""
    return R.window(cfg.window if cfg.window else cfg.radius)


PLATFORM_INPUTS = ("plat_dx", "plat_dy", "plat_vx", "plat_vy")


def platform_inputs(state):
    """Nearest moving platform: position relative to Mario in tiles, speed in 1/8 pixel per frame (all 0 when there is none)."""
    p = R.nearest_platform(state)
    if p is None:
        return np.zeros(4, dtype=np.int8)
    dx, dy, vx, vy = p
    clip = lambda v: int(max(-16, min(16, round(v))))
    return np.array([clip(dx / 16), clip(dy / 16), clip(vx * 8), clip(vy * 8)], dtype=np.int8)


def input_layout(cfg):
    """Ordered input groups as (name, kind); kind 'grid' = (2r+1)^2 cells, 'scalar' = one value."""
    return (_base_layout(cfg) + ([(name, "scalar") for name in jumpview.run_names()] if cfg.run_options else [])
            + ([(name, "scalar") for name in PLATFORM_INPUTS] if cfg.predict_platforms else [])
            + ([(name, "scalar") for name in deathmap.NAMES] if cfg.death_inputs else [])
            + ([(name, "scalar") for name in deathmap.TAKEOFF_NAMES] if cfg.takeoff_inputs else []))


def _base_layout(cfg):
    scalars = [("in_water", "scalar"), ("on_yoshi", "scalar"), ("powerup", "scalar")]
    if cfg.inputs in ("ride", "ride2"):        # explore + a platform channel: moving platforms are ground (and not enemies) for the network and the jump model
        motion = [("x_speed", "scalar"), ("y_speed", "scalar"), ("on_ground", "scalar")]
        return ([("solid", "grid"), ("enemy", "grid"), ("item", "grid"), ("coin", "grid"), ("pipe", "grid"), ("platform", "grid")]
                + scalars + motion + [(name, "scalar") for name in jumpview.names()])
    if cfg.inputs == "explore":     # physics + a pipe channel: pipes can be told apart from ground, and entering one can be rewarded
        motion = [("x_speed", "scalar"), ("y_speed", "scalar"), ("on_ground", "scalar")]
        return ([("solid", "grid"), ("enemy", "grid"), ("item", "grid"), ("coin", "grid"), ("pipe", "grid")] + scalars
                + motion + [(name, "scalar") for name in jumpview.names()])
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


def build_observation(ram, state, cfg, deaths=None, memory=None):
    """Flat int8 vector in input_layout order; `deaths` is the death map (DeathMap.export()) when cfg.death_inputs is on."""
    obs = _base_observation(ram, state, cfg, memory)
    if cfg.run_options:
        if cfg.jump_fixes and state.in_air and memory is not None and "run" in memory:
            extra = memory["run"]
        else:
            extra = jumpview.run_features(ram, state, ride=cfg.inputs in ("ride", "ride2"), ceilings=cfg.inputs == "ride2",
                                          fixed=cfg.jump_fixes, predict=cfg.predict_platforms)
            if memory is not None:
                memory["run"] = extra
        obs = np.concatenate([obs, extra])
    if cfg.predict_platforms:
        obs = np.concatenate([obs, platform_inputs(state)])
    if cfg.death_inputs:
        obs = np.concatenate([obs, np.array(deathmap.features(deaths, state.x), dtype=np.int8)])
    if cfg.takeoff_inputs:
        obs = np.concatenate([obs, np.array(deathmap.takeoff_features(deaths, state.x), dtype=np.int8)])
    return obs


def _base_observation(ram, state, cfg, memory=None):
    if cfg.inputs in ("channels", "collect", "physics", "explore", "ride", "ride2"):
        collect = cfg.inputs in ("collect", "physics", "explore", "ride", "ride2")
        enemy, item = R.sprite_grids(state, view(cfg), legacy=not collect, skip_platforms=cfg.inputs in ("ride", "ride2"),
                                   skip_harmless=cfg.jump_fixes)
        parts = [R.solid_grid(ram, state, view(cfg)).ravel(), enemy.ravel(), item.ravel()]
        if collect:
            parts.append(R.coin_grid(ram, state, view(cfg)).ravel())
        if cfg.inputs in ("explore", "ride", "ride2"):
            parts.append(R.pipe_grid(ram, state, view(cfg)).ravel())
        if cfg.inputs in ("ride", "ride2"):
            parts.append(R.platform_grid(state, view(cfg)).ravel())
        parts.append(np.array([state.in_water, state.on_yoshi, min(state.powerup, 3)], dtype=np.int8))
        if cfg.inputs in ("physics", "explore", "ride", "ride2"):
            parts.append(np.array(jumpview.motion_scalars(state), dtype=np.int8))
            if cfg.jump_fixes and state.in_air and memory is not None and "jump" in memory:
                parts.append(memory["jump"])                    # in the air the jump is under way: keep what the takeoff predicted
            else:
                jumps = jumpview.features(ram, state, ride=cfg.inputs in ("ride", "ride2"), ceilings=cfg.inputs == "ride2",
                                          fixed=cfg.jump_fixes, predict=cfg.predict_platforms)
                if memory is not None:
                    memory["jump"] = jumps
                parts.append(jumps)
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
        self._memory = {}
        self._takeoff = None
        self.death_map = None         # DeathMap.export(): set by the trainer (and by replays) when cfg.death_inputs is on
        self.best_ground_y = 0
        self._pipes = set()
        self.room_max, self.room_sig, self._room_pending = {}, None, False

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

    def _read(self, prev=None):
        ram = R.wram(self._env.unwrapped.get_ram())
        st = R.read_state(ram)
        if prev is not None:
            R.annotate_motion(prev.sprites, st.sprites, self.cfg.frame_skip)      # platform speeds from the last step
        return st, build_observation(ram, st, self.cfg, self.death_map, self._memory)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get("level_index", 0)
        self._open(self.cfg.levels[index])
        self.frame = self._env.reset()[0]
        for _ in range(2):  # let the first frames settle so RAM is valid
            self.frame = self._env.step(np.zeros(12, dtype=np.int8))[0]
        self._memory = {}
        self._takeoff = None
        self.state, grid = self._read()
        self.max_x = self.state.x
        self.frames = 0
        self.stuck = 0
        self.events = []
        self.best_ground_y = self.state.y        # highest standing position so far (smallest y)
        self._pipes = set()
        self.room_max, self.room_sig, self._room_pending = {}, None, False
        if self.capture is not None:
            self.capture.append(self.frame.copy())
        return grid, self._info()

    def _info(self):
        return {"x": self.state.x, "y": self.state.y, "max_x": max([self.max_x, *self.room_max.values()]),
                "powerup": self.state.powerup, "frames": self.frames}

    def step(self, action):
        if self.cfg.cancel_opposites:
            action = cancel_opposites(action)
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
        self.state, grid = self._read(prev)
        self._track_takeoff(prev, action)
        self._track_room(prev)
        rc = self.cfg.reward
        reward = -rc.time_penalty * self.cfg.frame_skip
        if self.state.in_water:
            reward -= rc.water_penalty * self.cfg.frame_skip
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
        st = self.state
        if prev.pipe_timer == 0 and st.pipe_timer > 0 and st.pipe_action in R.PIPE_ENTER:
            key = (st.rooms, st.x // 16, st.y // 16)          # one payment per pipe, so going back and forth is no farm
            if key not in self._pipes:
                self._pipes.add(key)
                reward += rc.pipe_bonus
                self.events.append([self.frames, "pipe", st.x, st.y])
                if rc.pipe_bonus > 0:
                    self.stuck = 0
        if st.rooms != prev.rooms:
            self.best_ground_y = st.y                         # new room: new height reference
        elif not st.in_air and st.y < self.best_ground_y - 12:
            reward += rc.height_bonus * (self.best_ground_y - st.y)
            self.events.append([self.frames, "climb", st.x, st.y])
            self.best_ground_y = st.y
            if rc.height_bonus > 0:
                self.stuck = 0
        if st.pipe_timer > 0 or not st.in_level:
            self.stuck = 0                                    # no control while in a pipe or while a room loads: not 'stuck'
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
        if died and self._takeoff is not None:
            t = self._takeoff
            info["takeoff"] = [t["x"], t["y"], t["mask"], t["hold"], t["run"]]      # the jump (or walk off a ledge) that ended in death
        if terminated or truncated:
            info["events"] = list(self.events)      # [frame, kind, x, y] for the path map
        return grid, float(reward), terminated, truncated, info

    def _track_takeoff(self, prev, action):
        """Remember the last takeoff: where Mario left the ground, what was pressed, and for how many steps jump (A or B) stayed held."""
        st = self.state
        jump = bool(action[BUTTONS.index("A")] or action[BUTTONS.index("B")])
        if not st.in_air:
            self._takeoff = None                          # landed (or never left): nothing to blame
        elif not prev.in_air:
            mask = int(sum(1 << i for i, on in enumerate(action) if on))
            self._takeoff = {"x": prev.x, "y": prev.y, "mask": mask, "hold": int(jump), "open": jump,
                             "run": int(bool(action[BUTTONS.index("Y")]))}
        elif self._takeoff is not None and self._takeoff["open"]:
            if jump:
                self._takeoff["hold"] += 1
            else:
                self._takeoff["open"] = False

    def _track_room(self, prev):
        """With room_progress on, every room (main level, bonus room) keeps its own furthest x."""
        if not self.cfg.room_progress:
            return
        st = self.state
        if st.rooms != prev.rooms:
            self._room_pending = True                     # the new room is still loading: fingerprint it once it runs
        if self._room_pending and st.in_level and st.pipe_timer == 0:
            wram = R.wram(self._env.unwrapped.get_ram())
            sig = R.room_signature(wram)
            if self.room_sig is not None and sig != self.room_sig:
                self.room_max[self.room_sig] = self.max_x
                self.max_x = self.room_max.get(sig, st.x)
                self.stuck = 0
            self.room_sig, self._room_pending = sig, False
        elif self.room_sig is None and st.in_level:
            self.room_sig = R.room_signature(R.wram(self._env.unwrapped.get_ram()))

    def render(self):
        return self.frame

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
