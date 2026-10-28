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


def test_channels_split_matches_mixed_grid():
    w = make_wram()
    put16(w, R.PLAYER_X, 160)
    put16(w, R.PLAYER_Y, 352)
    for tx in range(0, 40):
        w[R.MAP16_HIGH + (tx // 16) * R.TILES_PER_PAGE + 24 * 16 + tx % 16] = 1
    w[R.SPRITE_STATUS] = 8
    w[R.SPRITE_NUMBER] = 0x10
    w[R.SPRITE_X_LO], w[R.SPRITE_X_HI] = (160 + 16) & 0xFF, 0
    w[R.SPRITE_Y_LO], w[R.SPRITE_Y_HI] = 352 & 0xFF, 352 >> 8
    st = R.read_state(w)
    solid = R.solid_grid(w, st)
    enemy, item = R.sprite_grids(st)
    assert solid[8].all() and enemy[6, 7] == 1 and item.sum() == 0
    mixed = R.tile_grid(w, st)
    assert mixed[6, 7] == R.ENEMY and (mixed[8] == R.SOLID).all()


def test_water_and_yoshi_flags():
    w = make_wram()
    w[R.IN_WATER] = 1
    w[R.ON_YOSHI] = 2
    st = R.read_state(w)
    assert st.in_water and st.on_yoshi


def test_coin_grid_and_sprite_kinds():
    w = make_wram()
    put16(w, R.PLAYER_X, 160)
    put16(w, R.PLAYER_Y, 352)
    # a coin tile one row above Mario's row, same column: tile (x=(160+8)//16=10, y=21)
    off = (10 // 16) * R.TILES_PER_PAGE + 21 * 16 + 10
    w[R.MAP16_LOW + off] = R.COIN_TILE
    w[R.MAP16_HIGH + off] = 0
    # Yoshi (neutral) and an enemy one tile right
    w[R.SPRITE_STATUS] = 8
    w[R.SPRITE_NUMBER] = 0x35
    w[R.SPRITE_X_LO] = 176
    w[R.SPRITE_Y_LO], w[R.SPRITE_Y_HI] = 352 & 0xFF, 352 >> 8
    w[R.SPRITE_STATUS + 1] = 8
    w[R.SPRITE_NUMBER + 1] = 0x10
    w[R.SPRITE_X_LO + 1] = 192
    w[R.SPRITE_Y_LO + 1], w[R.SPRITE_Y_HI + 1] = 352 & 0xFF, 352 >> 8
    # one extended sprite (e.g. a fireball)
    w[R.EXT_NUMBER] = 5
    w[R.EXT_X_LO], w[R.EXT_Y_LO] = 144, 96
    w[R.EXT_Y_HI] = 1
    st = R.read_state(w)
    assert R.coin_grid(w, st)[5, 6] == 1 and R.coin_grid(w, st).sum() == 1
    enemy, item = R.sprite_grids(st, legacy=False)
    assert enemy.sum() == 1 and enemy[6, 8] == 1 and item.sum() == 0   # Yoshi + extended sprite skipped
    legacy_enemy, _ = R.sprite_grids(st, legacy=True)
    assert legacy_enemy.sum() == 3                                       # old behaviour kept


def test_x_speed_is_signed():
    w = make_wram()
    w[R.PLAYER_X_SPEED] = 0x30
    assert R.read_state(w).x_speed == 0x30
    w[R.PLAYER_X_SPEED] = 0xF0
    assert R.read_state(w).x_speed == -16


def test_used_blocks_and_yoshi_coins():
    w = make_wram()
    w[0x5D] = 2                                   # two screens loaded
    for i in (3, 500):                            # two used-block tiles (Map16 0x132)
        w[R.MAP16_HIGH + i] = 1
        w[R.MAP16_LOW + i] = R.USED_BLOCK
    w[R.MAP16_HIGH + 7] = 1                       # a solid tile that is not a used block
    w[R.MAP16_LOW + 7] = 0x30
    w[R.YOSHI_COINS] = 2
    st = R.read_state(w)
    assert st.used_blocks == 2 and st.yoshi_coins == 2


def test_bonus_stars_read_from_goal_bar_counter():
    w = make_wram()
    w[R.BONUS_STARS] = 25
    assert R.read_state(w).bonus_stars == 25


def test_window_shapes_and_shifted_view_sees_more_ground():
    w = make_wram()
    put16(w, R.PLAYER_X, 160)
    put16(w, R.PLAYER_Y, 352)
    for tx in range(0, 40):                               # ground 7 tiles below Mario's row (ty = 22 + 7 = 29 > 26): use ty 26
        w[R.MAP16_HIGH + (tx // 16) * R.TILES_PER_PAGE + 26 * 16 + tx % 16] = 1
    st = R.read_state(w)
    assert R.grid_shape(6) == (13, 13) and R.grid_shape((4, 8, 4, 8)) == (13, 13)
    assert R.grid_shape((3, 9, 2, 10)) == (13, 13) and R.grid_shape((2, 5, 3, 8)) == (12, 8)
    symmetric = R.solid_grid(w, st, 6)                    # rows up to +6: ground (4 rows below Mario's row 22) visible
    assert R.window(6) == (6, 6, 6, 6) and symmetric.shape == (13, 13)
    tall = R.solid_grid(w, st, (4, 8, 4, 8))
    assert tall.shape == (13, 13) and tall[4 + 4].all()   # row 4 (Mario) + 4 below = ground row, whole width
    enemy, item = R.sprite_grids(st, (4, 8, 4, 8), legacy=False)
    assert enemy.shape == (13, 13) and item.sum() == 0
