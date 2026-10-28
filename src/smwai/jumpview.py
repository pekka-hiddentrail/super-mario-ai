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


def _clip(v, lo, hi):
    return int(max(lo, min(hi, round(v))))


def _option(ram, state, vy0, hold, enemies):
    cx0, feet0 = state.x + 8, state.y + FEET
    coins, hit = set(), 0
    for p in physics.trajectory_from(state.x_speed, vy0, hold, 100):
        cx, feet = cx0 + p.dx, feet0 + p.dy
        if feet > LEVEL_BOTTOM:
            break                                        # below the level: a pit
        for ex, ey in enemies:
            if abs(ex + 8 - cx) < 14 and abs(ey + 8 - (feet - 16)) < 20:
                hit = 1
        if R.coin_at(ram, cx, feet - 16):
            coins.add((int(cx) // 16, int(feet - 16) // 16))
        if p.vy > 0 and R.solid_at(ram, cx, feet):
            return [1, hit, min(len(coins), 3), _clip((cx - cx0) / 16, -15, 15), _clip((feet - feet0) / 16, -8, 8)]
    return [-1, hit, min(len(coins), 3), 0, 0]


def features(ram, state):
    """The 30 jump-option values, in the order of names()."""
    enemies = [(s.x, s.y) for s in state.sprites if s.kind == "enemy"]
    out = []
    for button in BUTTONS_MODELLED:
        for hold in HOLDS:
            vy0 = state.y_speed if state.in_air else physics.takeoff_for(button, state.x_speed)
            out += _option(ram, state, vy0, hold, enemies)
    return np.array(out, dtype=np.int8)


def motion_scalars(state):
    """x speed (units of 8), y speed (units of 16), on-ground flag: small ints."""
    return [int(state.x_speed) // 8, int(state.y_speed) // 16, 0 if state.in_air else 1]
