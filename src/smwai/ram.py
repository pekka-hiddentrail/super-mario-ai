"""Read Super Mario World game state from a WRAM array (index == offset from $7E0000).

Addresses follow docs/smw_ram_map.json. The tile map is the Map16 *high byte* table at
$7FC800 (WRAM offset 0x1C800), the same one MarI/O used: value 1 means a solid block.
"""
from dataclasses import dataclass, field

import numpy as np

PLAYER_X = 0x94            # 16-bit, level pixels
PLAYER_Y = 0x96
LAYER1_X = 0x1A            # 16-bit camera position
LAYER1_Y = 0x1C
POWERUP = 0x19             # 0 small, 1 big, 2 cape, 3 fire
PLAYER_STATE = 0x71        # 9 = dying, 0 = normal
GAME_MODE = 0x100          # 0x14 = in level
LIVES = 0x0DBE
COINS = 0x0DBF
PLAYER_X_SPEED = 0x7B      # signed 8-bit, 1/16 pixel per frame
PLAYER_Y_SPEED = 0x7D      # signed 8-bit, 1/16 pixel per frame (positive = down)
IN_AIR = 0x72              # non-zero while Mario is in the air
PIPE_TIMER = 0x88          # counts down while Mario goes into (or comes out of) a pipe
PIPE_ACTION = 0x89         # entering: 1 sideways right, 3 downward; coming out: 5, 7 (measured on Yoshi Island 2 and 4)
PIPE_DOWN = 3
PIPE_ENTER = (1, 2, 3, 4)   # entering a pipe. Measured: 3 = downward, 1 = sideways to the right (Yoshi Island 4, x 2723); 2 and 4 assumed
                            # (left, up). Coming out: 5 (sideways) and 7 (downward) were measured and are never paid.
ROOMS = 0x141A             # increments every time a new room is entered through a pipe or door
YOSHI_COINS = 0x1420       # big (Yoshi/dragon) coins collected in this level
BONUS_STARS = 0x1900       # bonus stars won at the goal bar (read on the clear step, before the tally counts it down)
USED_BLOCK = 0x32          # Map16 0x132: a ? block that has been hit (count rises when one is hit)
IN_WATER = 0x75            # player is in water
ON_YOSHI = 0x187A          # riding Yoshi
END_LEVEL_TIMER = 0x1493   # non-zero once the goal is hit
SPRITE_STATUS = 0x14C8     # 12 slots, 0 = empty
SPRITE_NUMBER = 0x9E
SPRITE_X_LO, SPRITE_X_HI = 0xE4, 0x14E0
SPRITE_Y_LO, SPRITE_Y_HI = 0xD8, 0x14D4
EXT_NUMBER, EXT_X_LO, EXT_X_HI, EXT_Y_LO, EXT_Y_HI = 0x170B, 0x171F, 0x1733, 0x1715, 0x1729
MAP16_HIGH = 0x1C800
TILES_PER_PAGE = 0x1B0     # 16 columns x 27 rows per horizontal-level page
POWERUP_SPRITES = {0x74, 0x75, 0x76, 0x77, 0x78}   # mushroom, flower, star, feather, 1-up
NEUTRAL_SPRITES = {0x35}   # Yoshi
BRIDGE_SPRITE = 0x5A       # turn-block bridge: 5 solid blocks that pull in to one and push out again (value at $151C: 0..32)
SWING_SPRITE = 0x5F        # chain-hung block that carries Mario (sprite position = the block; geometry only roughly checked)
PLATFORM_SPRITES = {BRIDGE_SPRITE, SWING_SPRITE}
COIN_TILE = 0x2B           # Map16 low byte of a coin block (high byte 0)
MAP16_LOW = 0xC800
PIPE_TILES = range(0x34, 0x40)   # Map16 0x134-0x13F: the green pipe pieces (seen: 0x135-0x138 down pipe, 0x13B-0x13F sideways pipe)
N_SPRITES = 12
WRAM_OFFSET = 0x2800   # stable-retro get_ram(): 8K low mirror + 2K SRAM, then 128K WRAM


def wram(ram):
    """Slice the 128K WRAM ($7E0000-$7FFFFF) out of stable-retro's get_ram() array."""
    return ram[WRAM_OFFSET:]


def u8(ram, addr):
    return int(ram[addr])


def s16(ram, addr):
    v = int(ram[addr]) | int(ram[addr + 1]) << 8
    return v - 0x10000 if v & 0x8000 else v


@dataclass
class Sprite:
    x: int
    y: int
    number: int
    is_item: bool
    kind: str = "enemy"     # enemy | item | neutral (e.g. Yoshi) | other (extended sprites)
    slot: int = -1
    ext: int = 0            # turn-block bridge: how far the blocks have moved out from the middle one (pixels)


@dataclass
class MarioState:
    x: int
    y: int
    screen_x: int
    screen_y: int
    powerup: int
    lives: int
    coins: int
    dying: bool
    level_cleared: bool
    in_level: bool
    in_water: bool = False
    on_yoshi: bool = False
    x_speed: int = 0            # signed, 1/16 pixel per frame (positive = right)
    y_speed: int = 0            # signed, 1/16 pixel per frame (positive = down)
    in_air: bool = False
    ducking: bool = False       # crouching or sliding (down held on the ground)
    pipe_timer: int = 0
    pipe_action: int = 0
    rooms: int = 0
    yoshi_coins: int = 0
    bonus_stars: int = 0        # set when the goal bar is hit: 0 for a low hit, more the higher the bar was hit
    used_blocks: int = 0        # number of already-hit blocks in the level (rises on every block hit)
    sprites: list = field(default_factory=list)


def read_sprites(ram):
    sprites = []
    for slot in range(N_SPRITES):
        if u8(ram, SPRITE_STATUS + slot) == 0:
            continue
        number = u8(ram, SPRITE_NUMBER + slot)
        sprites.append(Sprite(
            x=u8(ram, SPRITE_X_LO + slot) | u8(ram, SPRITE_X_HI + slot) << 8,
            y=u8(ram, SPRITE_Y_LO + slot) | u8(ram, SPRITE_Y_HI + slot) << 8,
            number=number, is_item=number in POWERUP_SPRITES,
            kind="item" if number in POWERUP_SPRITES else "neutral" if number in NEUTRAL_SPRITES else "enemy",
            slot=slot, ext=u8(ram, 0x151C + slot) if number == BRIDGE_SPRITE else 0))
    for slot in range(N_SPRITES):
        if u8(ram, EXT_NUMBER + slot) == 0:
            continue
        sprites.append(Sprite(
            x=u8(ram, EXT_X_LO + slot) | u8(ram, EXT_X_HI + slot) << 8,
            y=u8(ram, EXT_Y_LO + slot) | u8(ram, EXT_Y_HI + slot) << 8,
            number=-1, is_item=False, kind="other"))
    return sprites


def used_block_count(ram):
    """How many Map16 'used block' tiles the loaded level currently has."""
    n = u8(ram, 0x5D) * TILES_PER_PAGE
    high, low = ram[MAP16_HIGH:MAP16_HIGH + n], ram[MAP16_LOW:MAP16_LOW + n]
    return int(np.count_nonzero((high == 1) & (low == USED_BLOCK)))


def read_state(ram):
    x, y = s16(ram, PLAYER_X), s16(ram, PLAYER_Y)
    return MarioState(
        x=x, y=y,
        screen_x=x - s16(ram, LAYER1_X), screen_y=y - s16(ram, LAYER1_Y),
        powerup=u8(ram, POWERUP), lives=u8(ram, LIVES), coins=u8(ram, COINS),
        dying=u8(ram, PLAYER_STATE) == 9,
        level_cleared=u8(ram, END_LEVEL_TIMER) != 0,
        in_level=u8(ram, GAME_MODE) == 0x14,
        in_water=u8(ram, IN_WATER) != 0,
        on_yoshi=u8(ram, ON_YOSHI) != 0,
        yoshi_coins=u8(ram, YOSHI_COINS), bonus_stars=u8(ram, BONUS_STARS), used_blocks=used_block_count(ram),
        x_speed=u8(ram, PLAYER_X_SPEED) - 256 if u8(ram, PLAYER_X_SPEED) > 127 else u8(ram, PLAYER_X_SPEED),
        y_speed=u8(ram, PLAYER_Y_SPEED) - 256 if u8(ram, PLAYER_Y_SPEED) > 127 else u8(ram, PLAYER_Y_SPEED),
        in_air=u8(ram, IN_AIR) != 0, ducking=u8(ram, 0x73) != 0,
        pipe_timer=u8(ram, PIPE_TIMER), pipe_action=u8(ram, PIPE_ACTION), rooms=u8(ram, ROOMS),
        sprites=read_sprites(ram))


# Grid cell codes
EMPTY, SOLID, ENEMY, ITEM = 0, 1, -1, 2


def window(view):
    """Normalise a view spec to (left, right, up, down) in tiles: an int r means r tiles every way."""
    if isinstance(view, int):
        return view, view, view, view
    left, right, up, down = view
    return int(left), int(right), int(up), int(down)


def grid_shape(view):
    left, right, up, down = window(view)
    return up + down + 1, left + right + 1


def _cells(state, view):
    """(row, col, world_x, world_y) for each tile of the window around Mario."""
    left, right, up, down = window(view)
    for row, dy in enumerate(range(-up * 16, down * 16 + 1, 16)):
        for col, dx in enumerate(range(-left * 16, right * 16 + 1, 16)):
            yield row, col, state.x + dx, state.y + dy


def solid_grid(ram, state, view=6):
    """Window grid centred on Mario (rows up..down, columns left..right): 1 where the Map16 tile is solid."""
    grid = np.zeros(grid_shape(view), dtype=np.int8)
    for row, col, wx, wy in _cells(state, view):
        tx, ty = (wx + 8) // 16, wy // 16
        if wx < 0 or wy < 0 or ty >= 27:
            continue
        idx = MAP16_HIGH + (tx // 16) * TILES_PER_PAGE + ty * 16 + tx % 16
        if idx < len(ram) and ram[idx] == 1:
            grid[row, col] = SOLID
    return grid


def coin_grid(ram, state, view=6):
    """Window grid centred on Mario: 1 where a coin tile is still uncollected."""
    grid = np.zeros(grid_shape(view), dtype=np.int8)
    for row, col, wx, wy in _cells(state, view):
        tx, ty = (wx + 8) // 16, wy // 16
        if wx < 0 or wy < 0 or ty >= 27:
            continue
        off = (tx // 16) * TILES_PER_PAGE + ty * 16 + tx % 16
        if MAP16_HIGH + off < len(ram) and ram[MAP16_LOW + off] == COIN_TILE and ram[MAP16_HIGH + off] == 0:
            grid[row, col] = 1
    return grid


def pipe_grid(ram, state, view=6):
    """Window grid centred on Mario: 1 on the pieces of a green pipe (so the network can tell pipes from ordinary ground)."""
    grid = np.zeros(grid_shape(view), dtype=np.int8)
    for row, col, wx, wy in _cells(state, view):
        tx, ty = (wx + 8) // 16, wy // 16
        if wx < 0 or wy < 0 or ty >= 27:
            continue
        off = (tx // 16) * TILES_PER_PAGE + ty * 16 + tx % 16
        if MAP16_HIGH + off < len(ram) and ram[MAP16_HIGH + off] == 1 and ram[MAP16_LOW + off] in PIPE_TILES:
            grid[row, col] = 1
    return grid


def sprite_grids(state, view=6, legacy=True, skip_platforms=False):
    """(enemy, item) window grids with 1 where such a sprite is.

    legacy=True keeps the original behaviour (every non-item sprite, extended ones included, is an
    enemy). legacy=False only marks real sprites: enemies, power-ups; Yoshi and extended sprites are skipped.
    """
    left, right, up, down = window(view)
    rows, cols = grid_shape(view)
    enemy = np.zeros((rows, cols), dtype=np.int8)
    item = np.zeros((rows, cols), dtype=np.int8)
    for sp in state.sprites:
        col = round((sp.x - state.x) / 16) + left
        row = round((sp.y - state.y) / 16) + up
        if not (0 <= row < rows and 0 <= col < cols):
            continue
        if skip_platforms and sp.number in PLATFORM_SPRITES:
            continue
        if legacy:
            (item if sp.is_item else enemy)[row, col] = 1
        elif sp.kind == "item":
            item[row, col] = 1
        elif sp.kind == "enemy":
            enemy[row, col] = 1
    return enemy, item


def tile_grid(ram, state, view=6):
    """Mixed grid: 1 solid, -1 enemy, 2 power-up, 0 empty (sprites overwrite tiles)."""
    grid = solid_grid(ram, state, view)
    enemy, item = sprite_grids(state, view)
    grid[enemy == 1] = ENEMY
    grid[item == 1] = ITEM
    return grid


def _tile_offset(px, py):
    """Offset of the Map16 tile at a level pixel inside the table, or None outside the level."""
    tx, ty = int(px) // 16, int(py) // 16
    if px < 0 or py < 0 or ty >= 27:
        return None
    return (tx // 16) * TILES_PER_PAGE + ty * 16 + tx % 16


def solid_at(ram, px, py):
    off = _tile_offset(px, py)
    return off is not None and MAP16_HIGH + off < len(ram) and ram[MAP16_HIGH + off] == 1


def coin_at(ram, px, py):
    off = _tile_offset(px, py)
    return (off is not None and MAP16_HIGH + off < len(ram)
            and ram[MAP16_LOW + off] == COIN_TILE and ram[MAP16_HIGH + off] == 0)


def room_signature(ram):
    """A fingerprint of the room that is loaded: screen count and number of solid tiles. Hitting blocks (still solid) and
    taking coins do not change it, so coming back to a room gives the same value."""
    n = u8(ram, 0x5D) * TILES_PER_PAGE
    return (u8(ram, 0x5D), int(np.count_nonzero(ram[MAP16_HIGH:MAP16_HIGH + n] == 1)))


def platform_blocks(state):
    """Solid 16x16 blocks (world x, y of the top-left corner) that belong to moving platforms right now.
    Bridge: the middle block plus one pair at +-ext/2 and one pair at +-ext. Swing: one block at the sprite position."""
    blocks = []
    for sp in state.sprites:
        if sp.number == BRIDGE_SPRITE:
            for off in {0, sp.ext // 2, -(sp.ext // 2), sp.ext, -sp.ext}:
                blocks.append((sp.x + off, sp.y))
        elif sp.number == SWING_SPRITE:
            blocks.append((sp.x, sp.y))
    return blocks


def platform_grid(state, view=6):
    """Window grid centred on Mario: 1 where a moving-platform block is."""
    left, right, up, down = window(view)
    rows, cols = grid_shape(view)
    grid = np.zeros((rows, cols), dtype=np.int8)
    for bx, by in platform_blocks(state):
        col = round((bx - state.x) / 16) + left
        row = round((by - state.y) / 16) + up
        if 0 <= row < rows and 0 <= col < cols:
            grid[row, col] = 1
    return grid


def on_platform(blocks, px, py):
    """Is the world point (px, py) inside one of the platform blocks?"""
    return any(bx <= px < bx + 16 and by <= py < by + 16 for bx, by in blocks)
