import numpy as np

from smwai import jumpview, physics
from smwai import ram as R


def wram():
    return np.zeros(0x20000, dtype=np.uint8)


def put16(w, addr, value):
    w[addr], w[addr + 1] = value & 0xFF, (value >> 8) & 0xFF


def flat_level(w, ground_row=24, tiles=60):
    w[0x5D] = 4
    for tx in range(tiles):
        w[R.MAP16_HIGH + (tx // 16) * R.TILES_PER_PAGE + ground_row * 16 + tx % 16] = 1


def state_at(w, x, y, x_speed=0, y_speed=0, in_air=False):
    put16(w, R.PLAYER_X, x)
    put16(w, R.PLAYER_Y, y)
    w[R.PLAYER_X_SPEED] = x_speed & 0xFF
    w[R.PLAYER_Y_SPEED] = y_speed & 0xFF
    w[R.IN_AIR] = 1 if in_air else 0
    return R.read_state(w)


def test_names_match_feature_count_and_range():
    w = wram()
    flat_level(w)
    st = state_at(w, 100, 24 * 16 - 32, x_speed=36)
    f = jumpview.features(w, st)
    assert len(f) == len(jumpview.names()) == 30 and f.dtype == np.int8
    assert abs(f).max() <= 16


def test_running_jump_over_flat_ground_lands_ahead_at_same_height():
    w = wram()
    flat_level(w)
    st = state_at(w, 100, 24 * 16 - 32, x_speed=36)           # running right on the ground
    f = jumpview.features(w, st).reshape(6, 5)
    land, enemy, coins, dx, dy = f[1]                          # button A held 10 frames
    assert (land, enemy, coins, dy) == (1, 0, 0, 0)
    assert 3 <= dx <= 8
    assert f[2][3] > f[0][3]                                   # a longer hold flies farther than a tap


def test_pit_is_reported_and_enemy_on_the_arc_is_seen():
    w = wram()
    flat_level(w, tiles=10)                                    # ground ends at tile 10 (x = 160 px)
    st = state_at(w, 120, 24 * 16 - 32, x_speed=36)
    assert (jumpview.features(w, st).reshape(6, 5)[0][0]) == -1   # a tap jump drops into the pit

    w2 = wram()
    flat_level(w2)
    w2[R.SPRITE_STATUS] = 8
    w2[R.SPRITE_NUMBER] = 0x10
    w2[R.SPRITE_X_LO] = 130
    flying = 24 * 16 - 72                                       # an enemy hovering at jump height
    w2[R.SPRITE_Y_LO], w2[R.SPRITE_Y_HI] = flying & 0xFF, flying >> 8
    st2 = state_at(w2, 100, 24 * 16 - 32, x_speed=36)
    assert jumpview.features(w2, st2).reshape(6, 5)[1][1] == 1    # an enemy in the path of the jump


def test_in_air_arc_uses_current_vertical_speed():
    w = wram()
    flat_level(w)
    st = state_at(w, 100, 24 * 16 - 80, x_speed=0, y_speed=40, in_air=True)    # falling, feet 3 tiles above the ground
    land, enemy, coins, dx, dy = jumpview.features(w, st).reshape(6, 5)[0]
    assert land == 1 and dx == 0 and dy == 3


def test_b_button_takes_off_harder_than_a():
    assert physics.takeoff_for("B", 0) < physics.takeoff_for("A", 0) < 0
    assert physics.takeoff_for("A", 36) == -79 and physics.takeoff_for("B", 36) == -87


def test_physics_mode_layout_and_size():
    from smwai.env import EnvConfig, input_layout, n_inputs
    cfg = EnvConfig(inputs="physics")
    names = [n for n, _ in input_layout(cfg)]
    assert names[:4] == ["solid", "enemy", "item", "coin"] and "x_speed" in names and "A10:land" in names
    assert n_inputs(cfg) == 169 * 4 + 3 + 3 + 30


def test_explore_mode_has_a_pipe_channel_and_sees_pipe_tiles():
    from smwai.env import EnvConfig, input_layout, n_inputs
    cfg = EnvConfig(inputs="explore")
    names = [n for n, _ in input_layout(cfg)]
    assert names[:5] == ["solid", "enemy", "item", "coin", "pipe"] and "A10:land" in names
    assert n_inputs(cfg) == 169 * 5 + 3 + 3 + 30
    w = wram()
    flat_level(w)
    tx, ty = 108 // 16, 23                                            # the tile under Mario's centre, one row below his row
    off = (tx // 16) * R.TILES_PER_PAGE + ty * 16 + tx % 16
    w[R.MAP16_HIGH + off], w[R.MAP16_LOW + off] = 1, 0x37             # a pipe top piece (0x137)
    st = state_at(w, 100, 24 * 16 - 32)
    grid = R.pipe_grid(w, st, 6)
    assert grid.sum() == 1 and grid[6 + (ty - 22), 6] == 1
    assert R.solid_grid(w, st, 6)[6 + (ty - 22), 6] == 1              # pipes are still solid blocks


def _put_sprite(w, slot, number, x, y, ext=0):
    w[R.SPRITE_STATUS + slot] = 8
    w[R.SPRITE_NUMBER + slot] = number
    w[R.SPRITE_X_LO + slot], w[R.SPRITE_X_HI + slot] = x & 0xFF, x >> 8
    w[R.SPRITE_Y_LO + slot], w[R.SPRITE_Y_HI + slot] = y & 0xFF, y >> 8
    w[0x151C + slot] = ext


def test_turn_block_bridge_blocks_follow_its_extension():
    w = wram()
    _put_sprite(w, 0, R.BRIDGE_SPRITE, 704, 208, ext=0)
    st = state_at(w, 645, 224)
    assert sorted(R.platform_blocks(st)) == [(704, 208)]                      # pulled in: one block
    _put_sprite(w, 0, R.BRIDGE_SPRITE, 704, 208, ext=32)
    st = state_at(w, 645, 224)
    assert sorted(R.platform_blocks(st)) == [(672, 208), (688, 208), (704, 208), (720, 208), (736, 208)]   # five blocks, 80 px


def test_ride_mode_hides_platforms_from_enemies_and_lands_on_them():
    from smwai.env import EnvConfig, input_layout, n_inputs
    cfg = EnvConfig(inputs="ride")
    names = [n for n, _ in input_layout(cfg)]
    assert names[:6] == ["solid", "enemy", "item", "coin", "pipe", "platform"]
    assert n_inputs(cfg) == 169 * 6 + 3 + 3 + 30
    w = wram()                                                                # no ground at all: a pit, except the bridge
    w[0x5D] = 4
    _put_sprite(w, 0, R.BRIDGE_SPRITE, 160, 24 * 16 - 16, ext=32)             # bridge 16 px above the (absent) floor line
    st = state_at(w, 100, 24 * 16 - 64, x_speed=36)                           # running, feet 2 tiles above the bridge top
    enemy, _ = R.sprite_grids(st, 6, legacy=False, skip_platforms=True)
    assert enemy.sum() == 0 and R.platform_grid(st, 6).sum() == 5
    plain = jumpview.features(w, st, ride=False).reshape(6, 5)
    ride = jumpview.features(w, st, ride=True).reshape(6, 5)
    assert plain[1][0] == -1                                                  # without the platform: a pit
    assert ride[1][0] == 1 and ride[1][1] == 0                                # with it: lands, and the bridge is no enemy


def test_ceiling_bump_stops_a_jump_from_below_a_platform_from_landing_on_top():
    w = wram()
    flat_level(w)
    _put_sprite(w, 0, R.BRIDGE_SPRITE, 130, 24 * 16 - 3 * 16, ext=32)       # five blocks whose bottom edge touches Mario's head
    st = state_at(w, 130 - 8, 24 * 16 - 32, x_speed=0)                        # standing directly under the middle block
    no_ceiling = jumpview.features(w, st, ride=True, ceilings=False).reshape(6, 5)
    with_ceiling = jumpview.features(w, st, ride=True, ceilings=True).reshape(6, 5)
    assert no_ceiling[2][4] < 0                                               # old model: 'lands on the platform' (dy negative = higher)
    assert with_ceiling[2][4] == 0 and with_ceiling[2][0] == 1                # new model: bumps and comes back down to the ground


def test_ceilings_off_matches_the_old_arc_exactly():
    w = wram()
    flat_level(w)
    st = state_at(w, 100, 24 * 16 - 32, x_speed=36)
    a = jumpview.features(w, st, ride=False, ceilings=False)
    b = jumpview.features(w, st)
    assert (a == b).all() and a.reshape(6, 5)[1][0] == 1


def test_ride2_has_the_ride_layout():
    from smwai.env import EnvConfig, input_layout, n_inputs
    assert input_layout(EnvConfig(inputs="ride2")) == input_layout(EnvConfig(inputs="ride"))
    assert n_inputs(EnvConfig(inputs="ride2")) == n_inputs(EnvConfig(inputs="ride"))
