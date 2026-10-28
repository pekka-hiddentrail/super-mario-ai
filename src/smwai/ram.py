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
            number=number, is_item=number in POWERUP_SPRITES))
    for slot in range(N_SPRITES):
        if u8(ram, EXT_NUMBER + slot) == 0:
            continue
        sprites.append(Sprite(
            x=u8(ram, EXT_X_LO + slot) | u8(ram, EXT_X_HI + slot) << 8,
            y=u8(ram, EXT_Y_LO + slot) | u8(ram, EXT_Y_HI + slot) << 8,
            number=-1, is_item=False))
    return sprites


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
        sprites=read_sprites(ram))


# Grid cell codes
EMPTY, SOLID, ENEMY, ITEM = 0, 1, -1, 2


def solid_grid(ram, state, radius=6):
    """(2r+1, 2r+1) int8 grid centred on Mario: 1 where the Map16 tile is solid."""
    size = radius * 2 + 1
    grid = np.zeros((size, size), dtype=np.int8)
    for row, dy in enumerate(range(-radius * 16, radius * 16 + 1, 16)):
        for col, dx in enumerate(range(-radius * 16, radius * 16 + 1, 16)):
            wx, wy = state.x + dx, state.y + dy
            tx, ty = (wx + 8) // 16, wy // 16
            if wx < 0 or wy < 0 or ty >= 27:
                continue
            idx = MAP16_HIGH + (tx // 16) * TILES_PER_PAGE + ty * 16 + tx % 16
            if idx < len(ram) and ram[idx] == 1:
                grid[row, col] = SOLID
    return grid


def sprite_grids(state, radius=6):
    """(enemy, item) int8 grids centred on Mario with 1 where such a sprite is."""
    size = radius * 2 + 1
    enemy = np.zeros((size, size), dtype=np.int8)
    item = np.zeros((size, size), dtype=np.int8)
    for sp in state.sprites:
        col = round((sp.x - state.x) / 16) + radius
        row = round((sp.y - state.y) / 16) + radius
        if 0 <= row < size and 0 <= col < size:
            (item if sp.is_item else enemy)[row, col] = 1
    return enemy, item


def tile_grid(ram, state, radius=6):
    """Mixed grid: 1 solid, -1 enemy, 2 power-up, 0 empty (sprites overwrite tiles)."""
    grid = solid_grid(ram, state, radius)
    enemy, item = sprite_grids(state, radius)
    grid[enemy == 1] = ENEMY
    grid[item == 1] = ITEM
    return grid
