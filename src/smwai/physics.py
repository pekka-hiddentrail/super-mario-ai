"""Mario's jump as a formula, measured from game memory with scripts/probe_jump.py (no emulator tricks).

Speeds are in 1/16 pixel per frame, positive = down / right. Per frame:
    y_speed  = y_speed + gravity           gravity = 3 while the jump button is held and y_speed < 36, else 6
    y_speed  = min(y_speed, 70)            terminal fall speed
    y       += y_speed / 16                pixels
    x       += x_speed / 16                x_speed stays (almost) constant in the air
The takeoff speed grows with run speed, in bins of 8:  -71, -73, -75, -77, -79, -82 for |x_speed| // 8 = 0..5.
Holding A for H frames therefore gives constant-acceleration motion in two pieces, the 'x^2 formula':
rising with gravity 3 for H-1 frames, then gravity 6:
    y(t) = v0*t + 3/2*t^2            (t <= H-1, in speed units; divide by 16 for pixels)
    y(t) = y(H-1) + v(H-1)*s + 3*s^2 (s = t-(H-1) frames after release)
"""
from dataclasses import dataclass

TAKEOFF = (71, 73, 75, 77, 79, 82)    # |takeoff y_speed| by |x_speed| // 8
GRAVITY_HELD, GRAVITY_FREE = 3, 6
HELD_LIMIT = 36                        # from this fall speed on the held-button gravity no longer applies
TERMINAL = 70


def takeoff_speed(x_speed):
    return -TAKEOFF[min(abs(int(x_speed)) // 8, len(TAKEOFF) - 1)]


@dataclass
class Point:
    t: int
    dx: float       # pixels right of the takeoff position (x_speed assumed constant)
    dy: float       # pixels below the takeoff position (negative = above)
    vy: int


def trajectory(x_speed, hold, frames=80):
    """Predicted path of a jump started now with `hold` frames of the jump button, one Point per frame."""
    vy, y, out = takeoff_speed(x_speed), 0.0, []
    for t in range(frames):
        y += vy / 16
        out.append(Point(t, (t + 1) * x_speed / 16, y, vy))
        gravity = GRAVITY_HELD if (t + 1 < hold and vy < HELD_LIMIT) else GRAVITY_FREE
        vy = min(vy + gravity, TERMINAL)
    return out


def landing(x_speed, hold, is_solid, x0, y0, frames=120):
    """First frame where the feet (x0, y0 in level pixels) enter a solid tile on the way down.

    `is_solid(px, py)` tells whether a level pixel is solid. Returns (frame, x, y) or None if it never lands
    (falls into a pit). The path is sampled at one point per frame, so thin floors at terminal speed (4 px)
    are still caught because the floor is a whole 16-px tile."""
    for p in trajectory(x_speed, hold, frames):
        if p.vy > 0 and is_solid(x0 + p.dx, y0 + p.dy):
            return p.t, x0 + p.dx, y0 + p.dy
    return None


def hold_for_apex(x_speed, height):
    """Shortest button hold (frames) whose jump rises at least `height` pixels, or None if it cannot."""
    for hold in range(1, 60):
        if -min(p.dy for p in trajectory(x_speed, hold)) >= height:
            return hold
    return None


# --- jump model extensions (button B and arcs that start in mid-air) -------------------------------------------
# The B button jumps harder than A at the same run speed. Measured takeoff speeds (probe_takeoff.py): bins 0, 1, 2
# and 4 are measured (77, 79, 82, 87); bins 3 and 5 are extrapolated and unverified.
TAKEOFF_B = (77, 79, 82, 85, 87, 90)


def takeoff_for(button, x_speed):
    table = TAKEOFF_B if button == "B" else TAKEOFF
    return -table[min(abs(int(x_speed)) // 8, len(table) - 1)]


def trajectory_from(x_speed, vy0, hold, frames=80):
    """Arc that starts with vertical speed vy0 (negative = rising): a takeoff speed, or the current y speed in
    mid-air. `hold` = frames the jump button stays down from now (1 = release immediately)."""
    vy, y, out = vy0, 0.0, []
    for t in range(frames):
        y += vy / 16
        out.append(Point(t, (t + 1) * x_speed / 16, y, vy))
        gravity = GRAVITY_HELD if (t + 1 < hold and vy < HELD_LIMIT) else GRAVITY_FREE
        vy = min(vy + gravity, TERMINAL)
    return out
