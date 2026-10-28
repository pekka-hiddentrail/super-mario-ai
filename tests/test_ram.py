import numpy as np

from smwai import ram as R


def make_wram():
    return np.zeros(0x20000, dtype=np.uint8)


def put16(w, addr, value):
    w[addr], w[addr + 1] = value & 0xFF, (value >> 8) & 0xFF


def test_read_state_basic():
    w = make_wram()
    put16(w, R.PLAYER_X, 155)
    put16(w, R.PLAYER_Y, 352)
    put16(w, R.LAYER1_X, 31)
    w[R.GAME_MODE] = 0x14
    w[R.LIVES] = 4
    st = R.read_state(w)
    assert (st.x, st.y, st.screen_x) == (155, 352, 124)
    assert st.in_level and not st.dying and not st.level_cleared


def test_dying_and_clear_flags():
    w = make_wram()
    w[R.PLAYER_STATE] = 9
    w[R.END_LEVEL_TIMER] = 0xFF
    st = R.read_state(w)
    assert st.dying and st.level_cleared


def test_tile_grid_solid_row_and_sprites():
    w = make_wram()
    put16(w, R.PLAYER_X, 160)
    put16(w, R.PLAYER_Y, 352)
    # solid ground two tile rows below Mario (ty = 352 // 16 + 2 = 24), across page 0 and 1
    for tx in range(0, 40):
        w[R.MAP16_HIGH + (tx // 16) * R.TILES_PER_PAGE + 24 * 16 + tx % 16] = 1
    # one enemy one tile right, one power-up one tile left, both on Mario's row
    w[R.SPRITE_STATUS] = 8
    w[R.SPRITE_NUMBER] = 0x10
    put16(w, 0xE4, 0)
    w[R.SPRITE_X_LO], w[R.SPRITE_X_HI] = (160 + 16) & 0xFF, 0
    w[R.SPRITE_Y_LO], w[R.SPRITE_Y_HI] = 352 & 0xFF, 352 >> 8
    w[R.SPRITE_STATUS + 1] = 8
    w[R.SPRITE_NUMBER + 1] = 0x74
    w[R.SPRITE_X_LO + 1], w[R.SPRITE_X_HI + 1] = (160 - 16) & 0xFF, 0
    w[R.SPRITE_Y_LO + 1], w[R.SPRITE_Y_HI + 1] = 352 & 0xFF, 352 >> 8
    grid = R.tile_grid(w, R.read_state(w))
    assert grid.shape == (13, 13)
    assert (grid[8] == R.SOLID).all()
    assert grid[6, 7] == R.ENEMY
    assert grid[6, 5] == R.ITEM
