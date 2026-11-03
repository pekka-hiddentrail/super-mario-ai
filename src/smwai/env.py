"""Gymnasium environment for Super Mario World on stable-retro.

Observation: (2r+1, 2r+1) int8 grid around Mario (1 solid, -1 enemy, 2 power-up).
Action: MultiBinary over BUTTONS (each output of a NEAT network maps to one button).
Reward: progress to the right (new max x) plus optional bonuses, all configurable.
"""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import deathmap, jumpview
from . import ram as R

GAME = "SuperMarioWorld-Snes-v0"
BUTTONS = ["A", "B", "X", "Y", "UP", "DOWN", "LEFT", "RIGHT"]
MIN_RISE = 12               # pixels higher than the takeoff spot that count as 'a higher platform' (opportunity_bonus)
MESSAGE_MAX_FRAMES = 600   # safety limit for closing a message box (it takes about 20 frames)


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
    ledge_bonus: float = 0.0       # flat, the first time Mario stands on each pass-through ledge (a run of ledge tiles is one, the rotating platform is one): like a coin
    opportunity_bonus: float = 0.0  # per pixel climbed by a jump that took an opportunity: a platform higher up was within reach (jumpview.higher_landings) when Mario left
                                    # the ground, and he lands on a higher spot (MIN_RISE+ pixels). Each landing tile pays once per episode.
    pipe_bonus: float = 0.0         # per pipe entered, down or sideways (each pipe pays once per episode)


@dataclass
class EnvConfig:
    levels: list = field(default_factory=lambda: ["YoshiIsland2"])
    frame_skip: int = 4
    radius: int = 6                # view: this many tiles every way around Mario ...
    window: list = None            # ... or [left, right, up, down] in tiles (overrides radius), e.g. [4, 8, 4, 8]
    inputs: str = "mixed"          # mixed: one tile grid | channels: solid/enemy/item + scalars | collect: + coins | physics: + jump options | explore: + pipes
    death_inputs: bool = False     # + four inputs from the death map: where earlier attempts died or got stuck (deathmap.py)
    death_nodes: bool = False      # the best fatal takeoff of every generation leaves a node: a random nudge on the buttons near that spot (deathmap.py)
    swing_model: bool = False      # the rotating platform (sprite 0x5F) as the model has measured it: where it will be, standing on it is not 'stuck', a wait-and-jump reach grid
    reach_map: bool = False        # + a grid of the cells some jump would land on (jumpview.reach_grid): where Mario can get to from here
    sprite_platforms: bool = False  # the checkerboard and floating platforms (ram.PLATFORM_EXTENT) are ground, not enemies: a platform grid, jump arcs that land on them, riding one is not 'stuck'
    block_kinds: bool = False      # the solid grid tells the turn block (the yellow one with the eyes, 2) from other solid tiles (1)
    checkpoints: list = field(default_factory=list)   # x coordinates: the first run to stand on the ground there (within checkpoint_tol, y < checkpoint_max_y) saves the game; later generations start from that save
    checkpoint_every: int = 0     # >0: a checkpoint every this many pixels of x, for every level: the first standing position after Mario crosses n * checkpoint_every (any y); no list needed
    checkpoint_patience: int = 0  # >0: this many generations in a row without a new furthest x from the current checkpoint and it is given up: the start goes back to the checkpoint before it
    checkpoint_tol: int = 20
    checkpoint_max_y: int = 200
    checkpoint_dir: str = ""      # set by train() to <run>/checkpoints: candidates in cand/, the chosen saves checkpoint_NN_xCX.state, current.txt names the start in use
    no_spin: bool = False          # the A button (spin jump) is never pressed: a spin jump from above breaks turn blocks Mario may need as a platform
    big_coins: bool = False        # the big (Yoshi) coin shows in the coin grid (as 2; only the ordinary coin did)
    ride_motion: bool = False      # + two inputs: how Mario moves while standing, beyond his own speed (a moving platform carries him; no sprite model needed)
    takeoff_inputs: bool = False   # + four inputs from where the best runs took off for their fatal fall, and how (deathmap.py)
    jump_fixes: bool = False       # jump inputs: walls stop the arc, the message box is no enemy (also not in the enemy grid), and the
                                   # values stay as they were at takeoff while Mario is in the air
    run_options: bool = False      # + 'keep running' and the jump options as from full run speed (jumpview.run_names): 35 inputs
    predict_platforms: bool = False  # jump options use where moving platforms will be during the arc; + 4 inputs about the nearest platform
    cancel_opposites: bool = False  # LEFT+RIGHT cancel each other, and so do UP+DOWN (networks otherwise hold both most of the time)
    room_progress: bool = False    # measure progress per room (pipes/doors): a bonus room starts its own 'furthest x'
    skip_messages: bool = True    # a message box freezes the game until A or B is pressed: close it by itself, not counted in the frame budget or the stuck rule
    stuck_frames: int = 120       # no new max x for this many frames -> truncate
    last_chance_frames: int = 0   # >0: in the last N frames before that, with a wall right in front of Mario, the env presses a fresh jump (B) + RIGHT + run for the policy
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


def platform_inputs(state, swing=False):
    """Nearest moving platform: position relative to Mario in tiles, speed in 1/8 pixel per frame (all 0 when there is none)."""
    p = R.nearest_platform(state, swing)
    if p is None:
        return np.zeros(4, dtype=np.int8)
    dx, dy, vx, vy = p
    clip = lambda v: int(max(-16, min(16, round(v))))
    return np.array([clip(dx / 16), clip(dy / 16), clip(vx * 8), clip(vy * 8)], dtype=np.int8)


def input_layout(cfg):
    """Ordered input groups as (name, kind); kind 'grid' = (2r+1)^2 cells, 'scalar' = one value."""
    return (_base_layout(cfg) + ([("platform", "grid")] if _extra_platform_grid(cfg) else [])
            + ([(name, "scalar") for name in jumpview.run_names()] if cfg.run_options else [])
            + ([(name, "scalar") for name in PLATFORM_INPUTS] if cfg.predict_platforms else [])
            + ([(name, "scalar") for name in deathmap.NAMES] if cfg.death_inputs else [])
            + ([(name, "scalar") for name in deathmap.TAKEOFF_NAMES] if cfg.takeoff_inputs else [])
            + ([(name, "scalar") for name in RIDE_INPUTS] if cfg.ride_motion else [])
            + ([("reach", "grid")] if cfg.reach_map else [])
            + ([("reach_wait", "grid")] if cfg.reach_map and cfg.swing_model else []))


def _extra_platform_grid(cfg):
    """sprite_platforms adds its own platform grid unless the 'ride' inputs have one already."""
    return cfg.sprite_platforms and cfg.inputs not in ("ride", "ride2")


def _ride(cfg):
    return cfg.inputs in ("ride", "ride2") or cfg.sprite_platforms


def _base_layout(cfg):
    scalars = [("in_water", "scalar"), ("on_yoshi", "scalar"), ("powerup", "scalar")]
    if cfg.inputs in ("ride", "ride2"):        # explore + a platform channel: moving platforms are ground (and not enemies) for the network and the jump model
        motion = [("x_speed", "scalar"), ("y_speed", "scalar"), ("on_ground", "scalar")]
        return ([("solid", "grid"), ("enemy", "grid"), ("item", "grid"), ("coin", "grid"), ("pipe", "grid"), ("platform", "grid")]
                + scalars + motion + [(name, "scalar") for name in jumpview.names()])
    if cfg.inputs == "ledges":      # physics + a ledge channel: the pass-through ledges and moving platforms are not walls (not in the solid grid), the jump options know it
        motion = [("x_speed", "scalar"), ("y_speed", "scalar"), ("on_ground", "scalar")]
        return ([("solid", "grid"), ("enemy", "grid"), ("item", "grid"), ("coin", "grid"), ("ledge", "grid")] + scalars
                + motion + [(name, "scalar") for name in jumpview.names()])
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
    if _extra_platform_grid(cfg):
        obs = np.concatenate([obs, R.platform_grid(state, view(cfg), cfg.swing_model).ravel().astype(np.int8)])
    if cfg.run_options:
        if cfg.jump_fixes and state.in_air and memory is not None and "run" in memory:
            extra = memory["run"]
        else:
            extra = jumpview.run_features(ram, state, ride=_ride(cfg), ceilings=cfg.inputs == "ride2",
                                          fixed=cfg.jump_fixes, predict=cfg.predict_platforms, swing=cfg.swing_model)
            if memory is not None:
                memory["run"] = extra
        obs = np.concatenate([obs, extra])
    if cfg.predict_platforms:
        obs = np.concatenate([obs, platform_inputs(state, cfg.swing_model)])
    if cfg.death_inputs:
        obs = np.concatenate([obs, np.array(deathmap.features(deaths, state.x), dtype=np.int8)])
    if cfg.takeoff_inputs:
        obs = np.concatenate([obs, np.array(deathmap.takeoff_features(deaths, state.x), dtype=np.int8)])
    if cfg.ride_motion:
        obs = np.concatenate([obs, ride_motion(state, (memory or {}).get("prev"))])
    if cfg.reach_map:
        ride = _ride(cfg)
        obs = np.concatenate([obs, jumpview.reach_grid(ram, state, view(cfg), ride=ride, ceilings=cfg.inputs == "ride2",
                                                       fixed=cfg.jump_fixes, predict=cfg.predict_platforms, swing=cfg.swing_model).ravel()])
        if cfg.swing_model:
            obs = np.concatenate([obs, jumpview.reach_wait_grid(ram, state, view(cfg), ride=ride, ceilings=cfg.inputs == "ride2",
                                                                fixed=cfg.jump_fixes, predict=cfg.predict_platforms, swing=True).ravel()])
    return obs


RIDE_INPUTS = ("ride_dy", "ride_dx")


def ride_motion(state, prev):
    """Mario's own motion while he stands (on the ground in this step and the one before): the game reports 'on the ground' and a y speed
    of 0 on a platform too, so the platform shows only as Mario moving although he does not walk or fall. ride_dy = pixels moved up/down
    over the step (positive = down; a rising platform is negative), ride_dx = pixels moved sideways beyond what his x speed explains.
    Slopes also move Mario up and down, so this is a hint, not proof of a platform."""
    if prev is None or state.in_air or prev.in_air:
        return np.zeros(2, dtype=np.int8)
    clip = lambda v: int(max(-16, min(16, v)))
    return np.array([clip(state.y - prev.y), clip(state.x - prev.x - round(state.x_speed / 4))], dtype=np.int8)


def _base_observation(ram, state, cfg, memory=None):
    if cfg.inputs in ("channels", "collect", "physics", "explore", "ride", "ride2", "ledges"):
        collect = cfg.inputs in ("collect", "physics", "explore", "ride", "ride2", "ledges")
        enemy, item = R.sprite_grids(state, view(cfg), legacy=not collect, skip_platforms=cfg.inputs in ("ride", "ride2"),
                                   skip_harmless=cfg.jump_fixes)
        parts = [R.solid_grid(ram, state, view(cfg), cfg.block_kinds, skip_ledges=cfg.inputs == "ledges").ravel(), enemy.ravel(), item.ravel()]
        if collect:
            parts.append(R.coin_grid(ram, state, view(cfg), cfg.big_coins).ravel())
        if cfg.inputs == "ledges":
            parts.append(R.ledge_grid(ram, state, view(cfg), cfg.swing_model).ravel())
        if cfg.inputs in ("explore", "ride", "ride2"):
            parts.append(R.pipe_grid(ram, state, view(cfg)).ravel())
        if cfg.inputs in ("ride", "ride2"):
            parts.append(R.platform_grid(state, view(cfg), cfg.swing_model).ravel())
        parts.append(np.array([state.in_water, state.on_yoshi, min(state.powerup, 3)], dtype=np.int8))
        if cfg.inputs in ("physics", "explore", "ride", "ride2", "ledges"):
            parts.append(np.array(jumpview.motion_scalars(state), dtype=np.int8))
            if cfg.jump_fixes and state.in_air and memory is not None and "jump" in memory:
                parts.append(memory["jump"])                    # in the air the jump is under way: keep what the takeoff predicted
            else:
                jumps = jumpview.features(ram, state, ride=_ride(cfg), ceilings=cfg.inputs == "ride2",
                                          fixed=cfg.jump_fixes, predict=cfg.predict_platforms, swing=cfg.swing_model, ledges=cfg.inputs == "ledges")
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
        self._level, self._raw = None, None
        self._pad = None
        self.frame = None
        self.capture = None            # set to a list to collect every emulator frame (video)
        self.events = []
        self._memory = {}
        self._takeoff = None
        self.node_flips = 0
        self.death_map = None         # DeathMap.export(): set by the trainer (and by replays) when cfg.death_inputs is on
        self.best_ground_y = 0
        self._pipes = set()
        self._chance, self._climbed, self._ledges = None, set(), set()
        self.room_max, self.room_sig, self._room_pending = {}, None, False
        self.start_path, self._cp_next, self._cp_n = None, 0, 0
        self.base, self._total = 0.0, 0.0        # the reward the chain had collected when this episode's start was saved, and what this episode has collected since
        self.new_cands = []           # stems of the checkpoint candidates saved since the trainer last looked: it stores the genome that made each next to its save
        self._trace = []              # with checkpoints: the emulator input of every frame since the reset (a 12-bit mask), to replay a chain of checkpoint runs from the real start

    def _open(self, level):
        import stable_retro as retro  # Linux/WSL only; imported lazily so config/tests work anywhere

        if self._env is not None and self._level == level:
            return
        if self._env is not None:
            self._env.close()
        raw = str(level).endswith(".state")          # a state saved with F5 in play_manual: raw emulator memory, loaded after the reset
        self._raw = Path(level).read_bytes() if raw else None
        self._env = retro.make(GAME, state="YoshiIsland2" if raw else level, use_restricted_actions=retro.Actions.ALL,
                               render_mode=None)
        self._level = level
        names = list(self._env.buttons)
        self._pad = [names.index(b) for b in BUTTONS]

    def _read(self, prev=None):
        ram = R.wram(self._env.unwrapped.get_ram())
        st = R.read_state(ram)
        if self.cfg.sprite_platforms:
            R.mark_platforms(st.sprites)
        if prev is not None:
            R.annotate_motion(prev.sprites, st.sprites, self.cfg.frame_skip)      # platform speeds from the last step
        self._memory["prev"] = prev
        return st, build_observation(ram, st, self.cfg, self.death_map, self._memory)

    def _close_message(self):
        """While a message box is open nothing moves and the network cannot do anything about it: press A until it closes.
        A is pressed on every other frame: the game reacts to a new press, not to a held button (a box that opens while the
        network holds A for a jump ignores a held A for good, measured on Yoshi Island 1 at x 2637).
        These frames are not counted (self.frames, the stuck rule), but a video recording keeps them."""
        press = np.zeros(12, dtype=np.int8)
        press[self._pad[BUTTONS.index("A")]] = 1
        release = np.zeros(12, dtype=np.int8)
        for i in range(MESSAGE_MAX_FRAMES):
            if R.u8(R.wram(self._env.unwrapped.get_ram()), R.MESSAGE) == 0:
                return
            self.frame = self._emu(press if i % 2 == 0 else release)
            if self.capture is not None:
                self.capture.append(self.frame.copy())

    def _emu(self, pad):
        """One emulator frame; with checkpoints on, its input goes into the trace."""
        if self.cfg.checkpoint_dir:
            self._trace.append(int(sum(1 << i for i, v in enumerate(pad) if v)))
        return self._env.step(pad)[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get("level_index", 0)
        start = (options["start"] or None) if options and "start" in options else self._checkpoint_start()    # an explicit start ("" = the level's own) wins over the current checkpoint
        self.start_path = start
        m = re.search(r"checkpoint_(\d+)_x", Path(start).name) if start else None
        self._cp_next = int(m.group(1)) if m else 0                      # checkpoint_NN: the NN checkpoints before it are behind Mario
        self.base = self._start_score(start)
        self._total = 0.0
        self._open(start or self.cfg.levels[index])
        self.frame = self._env.reset()[0]
        if self._raw:
            self._env.unwrapped.em.set_state(self._raw)
        self._trace = []
        for _ in range(2):  # let the first frames settle so RAM is valid
            self.frame = self._emu(np.zeros(12, dtype=np.int8))
        return self._begin()

    def attach(self, retro_env, frame):
        """Play on an emulator someone else runs (the whole game: a level entered from the map). No state is loaded: the episode begins where the game is."""
        self._env, self._level, self._raw = retro_env, "attached", None
        names = list(retro_env.buttons)
        self._pad = [names.index(b) for b in BUTTONS]
        self.start_path, self._cp_next, self.base, self._total, self._trace = None, 0, 0.0, 0.0, []
        self.frame = frame
        return self._begin()

    def _begin(self):
        self._memory = {}
        self._takeoff = None
        self.node_flips = 0
        self.state, grid = self._read()
        self.max_x = self.state.x
        self.frames = 0
        self.stuck = 0
        self._chance_on = False
        self.events = []
        self.best_ground_y = self.state.y        # highest standing position so far (smallest y)
        self._pipes = set()
        self._chance, self._climbed, self._ledges = None, set(), set()   # the last takeoff with a higher platform in reach (feet y, landings), the landing tiles already paid for
        self.room_max, self.room_sig, self._room_pending = {}, None, False
        if self.capture is not None:
            self.capture.append(self.frame.copy())
        return grid, self._info()

    def _info(self):
        info = {"x": self.state.x, "y": self.state.y, "max_x": max([self.max_x, *self.room_max.values()]),
                "powerup": self.state.powerup, "frames": self.frames}
        if self.start_path:
            info["start"] = self.start_path                              # the checkpoint save this episode started from (absent: the level's own start)
        return info

    @staticmethod
    def _start_score(start):
        """The reward the chain had collected at a checkpoint save (its json 'score'): runs from that save add it, so scores before and after a checkpoint compare."""
        if not start or not Path(start).with_suffix(".json").exists():
            return 0.0
        return float(json.loads(Path(start).with_suffix(".json").read_text()).get("score", 0.0))

    def _checkpoint_start(self):
        """The save the current checkpoint names (current.txt in checkpoint_dir), None before the first checkpoint."""
        if not self.cfg.checkpoint_dir:
            return None
        pointer = Path(self.cfg.checkpoint_dir) / "current.txt"
        if not pointer.exists():
            return None
        path = pointer.read_text().strip()
        return path if path and Path(path).exists() else None

    def _checkpoint(self):
        """checkpoints: standing on the ground at one of the listed x (and high enough) saves the game as a candidate; train() picks the first one of the generation."""
        cfg, st = self.cfg, self.state
        if not cfg.checkpoint_dir or not (cfg.checkpoints or cfg.checkpoint_every):
            return
        if st.in_air or st.dying or st.pipe_timer > 0 or not st.in_level:
            return
        if cfg.checkpoint_every > 0:                                     # every n * checkpoint_every: standing at or past it, any y
            n = int(st.x // cfg.checkpoint_every)
            if n >= 1 and n > self._cp_next:
                self._save_candidate(n, n * cfg.checkpoint_every)
            return
        if self._cp_next >= len(cfg.checkpoints) or st.y >= cfg.checkpoint_max_y:
            return
        for j in range(self._cp_next, len(cfg.checkpoints)):
            cx = cfg.checkpoints[j]
            if abs(st.x - cx) <= cfg.checkpoint_tol:
                self._save_candidate(j + 1, cx)
                return

    def _save_candidate(self, number, cx):
        """The game as it is now (and the input that led here from this episode's start) as a candidate for checkpoint `number`."""
        cand = Path(self.cfg.checkpoint_dir) / "cand"
        cand.mkdir(parents=True, exist_ok=True)
        self._cp_n += 1
        stem = cand / f"cand_{number:02d}_x{cx}_f{self.frames:07d}_{os.getpid()}_{self._cp_n}"
        stem.with_suffix(".state").write_bytes(self._env.unwrapped.em.get_state())
        stem.with_suffix(".json").write_text(json.dumps({"start": self.start_path, "trace": self._trace, "score": self.base + self._total}))
        self._cp_next = number
        self.new_cands.append(stem)
        self.events.append([self.frames, "checkpoint", self.state.x, self.state.y])

    def _wall_ahead(self):
        """A hard (not pass-through) tile just past Mario's right edge, at his lower body: he is standing against a wall."""
        ram = R.wram(self._env.unwrapped.get_ram())
        px, feet = self.state.x + 20, self.state.y + 32
        return any(R.solid_at(ram, px, feet - d) and not R.ledge_at(ram, px, feet - d) for d in (4, 20))

    def _last_chance(self, action):
        """last_chance_frames: when the stuck clock is within that many frames of running out and a wall stands right in front of Mario, the policy's buttons are
        replaced by run + RIGHT + a jump: the first step releases B (so the press is a new one), the rest hold it. Once started it runs on (in the air the wall is
        below him). Progress resets the clock and ends it; the clock itself is not extended."""
        n = self.cfg.last_chance_frames
        if n <= 0 or self.stuck < self.cfg.stuck_frames - n:
            self._chance_on = False
            return action
        if not self._chance_on and not self._wall_ahead():
            return action
        action = np.array(action, dtype=bool)
        action[BUTTONS.index("LEFT")] = False
        action[BUTTONS.index("RIGHT")] = action[BUTTONS.index("Y")] = True
        action[BUTTONS.index("B")] = self._chance_on
        self._chance_on = True
        return action

    def step(self, action):
        if self.cfg.cancel_opposites:
            action = cancel_opposites(action)
        if self.cfg.no_spin:
            action = np.array(action, dtype=bool)
            action[BUTTONS.index("A")] = False
        action = self._last_chance(action)
        pad = np.zeros(12, dtype=np.int8)
        for i, on in enumerate(action):
            if on:
                pad[self._pad[i]] = 1
        prev = self.state
        for _ in range(self.cfg.frame_skip):
            self.frame = self._emu(pad)
            self.frames += 1
            if self.capture is not None:          # video recording: keep every emulator frame
                self.capture.append(self.frame.copy())
        if self.cfg.skip_messages:
            self._close_message()
        self.state, grid = self._read(prev)
        self._track_takeoff(prev, action)
        self._track_room(prev)
        self._checkpoint()
        rc = self.cfg.reward
        reward = -rc.time_penalty * self.cfg.frame_skip
        reward += self._opportunity(prev)
        reward += self._ledge(prev)
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
        if self.cfg.sprite_platforms and (ride := R.riding_platform(st)) is not None and abs(ride.vx) + abs(ride.vy) > 0.2:
            self.stuck = 0                                    # carried by a moving platform: waiting is not 'stuck'
        if self.cfg.swing_model and R.standing_on_swing(st) is not None:
            self.stuck = 0                                    # carried by the rotating platform (a lap takes 1012 frames): waiting is not 'stuck'
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
        info["node_flips"] = self.node_flips
        if died and self._takeoff is not None:
            t = self._takeoff
            info["takeoff"] = [t["x"], t["y"], t["mask"], t["hold"], t["run"]]      # the jump (or walk off a ledge) that ended in death
        if terminated or truncated:
            info["events"] = list(self.events)      # [frame, kind, x, y] for the path map
        self._total += float(reward)
        return grid, float(reward), terminated, truncated, info

    def decide(self, out):
        """Network outputs (0..1) -> pressed buttons (above 0.5); with death_nodes, nodes near Mario's spot shift the outputs' logits while he is on the ground."""
        out = np.asarray(out, dtype=np.float64)
        nodes = (self.death_map or {}).get(deathmap.NODES) if self.cfg.death_nodes else None
        pressed = out > 0.5
        if nodes and not self.state.in_air:
            nudge = np.array(deathmap.node_nudge(nodes, self.state.x, self.state.y))
            if nudge.any():
                p = np.clip(out, 1e-6, 1 - 1e-6)
                nudged = np.log(p / (1 - p)) + deathmap.NODE_LOGIT * nudge > 0
                self.node_flips += int((nudged != pressed).sum())          # buttons the nodes turned on or off in this episode
                return nudged
        return pressed

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

    def _ledge(self, prev):
        """ledge_bonus: a flat payment the first time Mario stands on each ledge (R.ledge_key); it counts as progress for the stuck rule."""
        rc, st = self.cfg.reward, self.state
        if rc.ledge_bonus <= 0 or st.in_air:
            return 0.0
        key = R.ledge_key(R.wram(self._env.unwrapped.get_ram()), st, self.cfg.swing_model)
        if key is None or key in self._ledges:
            return 0.0
        self._ledges.add(key)
        self.events.append([self.frames, "climb", st.x, st.y])
        self.stuck = 0
        return rc.ledge_bonus

    def _opportunity(self, prev):
        """opportunity_bonus: at takeoff, note whether any jump from here would land on something higher up; on landing, pay per pixel climbed
        when Mario really came down at least MIN_RISE pixels higher (a landing tile pays once). Climbing by luck, with nothing in reach, pays nothing."""
        rc, st = self.cfg.reward, self.state
        if rc.opportunity_bonus <= 0:
            return 0.0
        if st.in_air and not prev.in_air:
            hits = jumpview.higher_landings(R.wram(self._env.unwrapped.get_ram()), prev, MIN_RISE)
            self._chance = (prev.y + jumpview.FEET, hits) if hits else None
        elif prev.in_air and not st.in_air and self._chance is not None:
            feet0, _ = self._chance
            self._chance = None
            rise = feet0 - (st.y + jumpview.FEET)
            key = ((st.x + 8) // 16, (st.y + jumpview.FEET) // 16)
            if rise >= MIN_RISE and key not in self._climbed and st.rooms == prev.rooms:
                self._climbed.add(key)
                self.events.append([self.frames, "climb", st.x, st.y])
                self.stuck = 0
                return rc.opportunity_bonus * rise
        return 0.0

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
