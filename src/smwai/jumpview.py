"""What would a jump do from here? Inputs for the network, computed from the measured jump formula (physics.py).

For a few standard jumps (button A or B, held 1 / 10 / 25 frames) the formula gives Mario's whole arc from his
current position and speeds. Walking that arc over the tiles and sprites on screen gives, per option:
    land   +1 = comes down on solid ground, -1 = falls into a pit (never lands)
    enemy  1 = the arc passes through an enemy
    coins  coins touched on the way (0-3)
    dx, dy where it lands, in tiles from now (dx forward, dy positive = lower)
The agent still chooses when to jump and how long to hold; the model only tells it what each choice would do, the
way a player knows how far Mario flies. No emulator rollouts, no scripts. Checked against the emulator (scripts/validate_jumps.py, 216 jumps on Yoshi Island 2): land-or-pit is right 97% of
the time, the landing is within one tile 65% of the time. Limits: walls, ceilings and bounces are ignored,
only solid tiles (not ledges that are solid from above only) count as ground, enemies are taken as they are now.
"""
import dataclasses

import numpy as np

from . import physics
from . import ram as R

HOLDS = (1, 10, 25)
BUTTONS_MODELLED = ("A", "B")
FIELDS = ("land", "enemy", "coins", "dx", "dy")
FEET = 32            # Mario's y is the top of his 32-pixel box; his feet are at y + 32 (measured: ground top = y + 32)
LEVEL_BOTTOM = 27 * 16 + 48


def names():
    return [f"{b}{h}:{f}" for b in BUTTONS_MODELLED for h in HOLDS for f in FIELDS]


RUN_SPEED = 48           # full run speed (1/16 px per frame): what a run-up would give
WALK_FRAMES = 60


def run_names():
    """'keep running' (no jump), then the six jumps as they would be from full run speed."""
    return [f"run:{f}" for f in FIELDS] + [f"fast {b}{h}:{f}" for b in BUTTONS_MODELLED for h in HOLDS for f in FIELDS]


def _clip(v, lo, hi):
    return int(max(lo, min(hi, round(v))))


def _option(ram, state, vy0, hold, enemies, blocks=(), ceilings=False, walls=False, moving=None):
    """One jump option. ceilings=True: rising into a solid tile or platform block bumps the head (the rise stops), so a jump from
    below a platform no longer 'lands on top of it'."""
    cx0, feet0 = state.x + 8, state.y + FEET
    coins, hit = set(), 0
    vy, dy = vy0, 0.0
    cx = cx0
    for t in range(100):
        if moving:
            blocks = moving[min(t + 1, len(moving) - 1)]           # where the platforms will be during this frame
        if walls:                                              # running into a solid tile at body height stops the sideways motion
            nxt = cx + state.x_speed / 16
            if state.x_speed and R.solid_at(ram, nxt + (7 if state.x_speed > 0 else -7), feet0 + dy - 16):
                nxt = cx
            cx = nxt
        else:
            cx = cx0 + (t + 1) * state.x_speed / 16
        step = vy / 16
        if ceilings and vy < 0:
            head = feet0 + dy + step - FEET
            if R.solid_at(ram, cx, head) or R.on_platform(blocks, cx, head):
                vy, step = 0, 0.0                              # head bump: no more rise, the arc turns down
        dy += step
        feet = feet0 + dy
        if feet > LEVEL_BOTTOM:
            break                                              # below the level: a pit
        for ex, ey in enemies:
            if abs(ex + 8 - cx) < 14 and abs(ey + 8 - (feet - 16)) < 20:
                hit = 1
        if R.coin_at(ram, cx, feet - 16):
            coins.add((int(cx) // 16, int(feet - 16) // 16))
        if vy > 0 and (R.solid_at(ram, cx, feet) or R.on_platform(blocks, cx, feet)):
            return [1, hit, min(len(coins), 3), _clip((cx - cx0) / 16, -15, 15), _clip((feet - feet0) / 16, -8, 8)]
        gravity = physics.GRAVITY_HELD if (t + 1 < hold and vy < physics.HELD_LIMIT) else physics.GRAVITY_FREE
        vy = min(vy + gravity, physics.TERMINAL)
    return [-1, hit, min(len(coins), 3), 0, 0]


REACH_HOLDS = (1, 5, 10, 15, 20, 25)         # button hold times of the arcs of the reach map (frames)


def _land(ram, state, x_speed, vy0, hold, blocks=(), ceilings=False, walls=False, moving=None):
    """Where does this arc come down? (x, feet y) of the point that enters the solid tile or platform, or None when it falls into a pit.
    The same arc as _option, with the sideways speed given: no enemies, no coins."""
    cx0, feet0 = state.x + 8, state.y + FEET
    vy, dy, cx = vy0, 0.0, cx0
    for t in range(100):
        if moving:
            blocks = moving[min(t + 1, len(moving) - 1)]
        if walls:
            nxt = cx + x_speed / 16
            if x_speed and R.solid_at(ram, nxt + (7 if x_speed > 0 else -7), feet0 + dy - 16):
                nxt = cx
            cx = nxt
        else:
            cx = cx0 + (t + 1) * x_speed / 16
        step = vy / 16
        if ceilings and vy < 0:
            head = feet0 + dy + step - FEET
            if R.solid_at(ram, cx, head) or R.on_platform(blocks, cx, head):
                vy, step = 0, 0.0
        dy += step
        feet = feet0 + dy
        if feet > LEVEL_BOTTOM:
            return None
        if vy > 0 and (R.solid_at(ram, cx, feet) or R.on_platform(blocks, cx, feet)):
            return cx, feet
        gravity = physics.GRAVITY_HELD if (t + 1 < hold and vy < physics.HELD_LIMIT) else physics.GRAVITY_FREE
        vy = min(vy + gravity, physics.TERMINAL)
    return None


def reach_grid(ram, state, view, ride=False, ceilings=False, fixed=False, predict=False):
    """Where can Mario land? A window grid like the solid grid (rows up..down, columns left..right around Mario's tile): each cell
    that is the tile some jump would come down on gets a count of the arcs that land there (1-3). On the ground the arcs are
    button A and B with holds REACH_HOLDS from the speed he has now, from full run speed to the right and to the left; in the air
    the arc from the current speeds, with the button let go now or held on. Only what the physics model says is possible: the
    network decides whether and when."""
    left, right, up, down = R.window(view)
    grid = np.zeros((up + down + 1, left + right + 1), dtype=np.int8)
    _, blocks = _scene(state, ride, fixed)
    moving = _moving(state, ride, predict)
    here_x, here_y = (state.x + 8) // 16, state.y // 16
    if state.in_air:
        arcs = [(state.x_speed, state.y_speed, hold) for hold in (1, 25)]
    else:
        arcs = [(speed, physics.takeoff_for(button, speed), hold) for speed in dict.fromkeys((state.x_speed, RUN_SPEED, -RUN_SPEED))
                for button in BUTTONS_MODELLED for hold in REACH_HOLDS]
    for speed, vy0, hold in arcs:
        hit = _land(ram, state, speed, vy0, hold, blocks, ceilings, fixed, moving)
        if hit is None:
            continue
        row, col = up + int(hit[1]) // 16 - here_y, left + int(hit[0]) // 16 - here_x
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            grid[row, col] = min(3, grid[row, col] + 1)
    return grid


def _walk(ram, state, enemies, blocks, walls, moving=None):
    """Keep running right without jumping: where does Mario end up after WALK_FRAMES? Same five values as a jump option."""
    speed = max(abs(state.x_speed), 24)
    cx0, feet0 = state.x + 8, state.y + FEET
    cx, feet, vy = cx0, float(feet0), 0.0
    coins, hit = set(), 0
    for t in range(WALK_FRAMES):
        if moving:
            blocks = moving[min(t + 1, len(moving) - 1)]
        nxt = cx + speed / 16
        if walls and R.solid_at(ram, nxt + 7, feet - 16):
            nxt = cx                                           # a wall: Mario stands still against it
        cx = nxt
        if R.solid_at(ram, cx, feet) or R.on_platform(blocks, cx, feet):
            vy = 0.0
        else:
            vy = min(vy + physics.GRAVITY_FREE, physics.TERMINAL)
            feet += vy / 16
            if R.solid_at(ram, cx, feet) or R.on_platform(blocks, cx, feet):
                feet, vy = (int(feet) // 16) * 16, 0.0         # landed on the tile top
        if feet > LEVEL_BOTTOM:
            return [-1, hit, min(len(coins), 3), 0, 0]
        for ex, ey in enemies:
            if abs(ex + 8 - cx) < 14 and abs(ey + 8 - (feet - 16)) < 20:
                hit = 1
        if R.coin_at(ram, cx, feet - 16):
            coins.add((int(cx) // 16, int(feet - 16) // 16))
    return [1, hit, min(len(coins), 3), _clip((cx - cx0) / 16, -15, 15), _clip((feet - feet0) / 16, -8, 8)]


def _moving(state, ride, predict):
    """Platform blocks for each of the next 100 frames (index 0 = now), or None when platforms are taken as they are now."""
    return [R.platform_blocks(state, t) for t in range(101)] if ride and predict else None


def run_features(ram, state, ride=False, ceilings=False, fixed=False, predict=False):
    """35 values in the order of run_names(): the 'keep running' option, then the jump options as if Mario ran at full speed
    (what a run-up would make possible)."""
    enemies, blocks = _scene(state, ride, fixed)
    moving = _moving(state, ride, predict)
    out = _walk(ram, state, enemies, blocks, fixed, moving)
    fast = dataclasses.replace(state, x_speed=RUN_SPEED)
    for button in BUTTONS_MODELLED:
        for hold in HOLDS:
            vy0 = state.y_speed if state.in_air else physics.takeoff_for(button, RUN_SPEED)
            out += _option(ram, fast, vy0, hold, enemies, blocks, ceilings, walls=fixed, moving=moving)
    return np.array(out, dtype=np.int8)


def _scene(state, ride, fixed):
    enemies = [(s.x, s.y) for s in state.sprites if s.kind == "enemy" and not (ride and s.number in R.PLATFORM_SPRITES)
               and not (fixed and s.number in R.HARMLESS_SPRITES)]
    return enemies, (R.platform_blocks(state) if ride else ())


def features(ram, state, ride=False, ceilings=False, fixed=False, predict=False):
    """The 30 jump-option values, in the order of names(). ride=True: moving platforms are ground, not enemies."""
    enemies, blocks = _scene(state, ride, fixed)
    moving = _moving(state, ride, predict)
    out = []
    for button in BUTTONS_MODELLED:
        for hold in HOLDS:
            vy0 = state.y_speed if state.in_air else physics.takeoff_for(button, state.x_speed)
            out += _option(ram, state, vy0, hold, enemies, blocks, ceilings, walls=fixed, moving=moving)
    return np.array(out, dtype=np.int8)


def motion_scalars(state):
    """x speed (units of 8), y speed (units of 16), on-ground flag: small ints."""
    return [int(state.x_speed) // 8, int(state.y_speed) // 16, 0 if state.in_air else 1]
