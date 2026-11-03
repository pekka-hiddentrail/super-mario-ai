"""The overworld of Super Mario World: where Mario stands on the map, how he walks it, how a level is entered and how the map is reached again.

Everything here was measured on the game (scripts in docs/MAP_NAVIGATION_PLAN.md, section "What was measured"): the game mode at $0100 is 0x0E on the map and 0x14 in a level, the
loading modes 0x0F..0x13 lie between, Mario's map position is the 16-bit pair $1F17 / $1F19, a level is entered with A and its translevel number lands in $13BF.
"""
import numpy as np

from . import ram as R

GAME = "SuperMarioWorld-Snes-v0"
MODE, OW_X, OW_Y, SUBMAP, TRANSLEVEL, LIVES, GOTO_X, GOTO_Y = 0x100, 0x1F17, 0x1F19, 0x13C3, 0x13BF, 0xDBE, 0xDC7, 0xDC9
MODE_MAP, MODE_LEVEL = 0x0E, 0x14
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

# translevel number ($13BF while the level is entered) of every level the game ships a start state for; read from those states
LEVELS = {0x29: "YoshiIsland1", 0x2A: "YoshiIsland2", 0x27: "YoshiIsland3", 0x26: "YoshiIsland4",
          0x15: "DonutPlains1", 0x09: "DonutPlains2", 0x04: "DonutPlains3", 0x05: "DonutPlains4", 0x06: "DonutPlains5",
          0x3E: "VanillaDome1", 0x3C: "VanillaDome2", 0x2B: "VanillaDome3", 0x2E: "VanillaDome4", 0x3D: "VanillaDome5",
          0x0F: "Bridges1", 0x10: "Bridges2", 0x42: "Forest1", 0x44: "Forest2", 0x47: "Forest3", 0x41: "Forest4", 0x43: "Forest5",
          0x22: "ChocolateIsland1", 0x21: "ChocolateIsland2", 0x24: "ChocolateIsland3"}


class Game:
    """The whole game on one emulator: the map, the levels and what lies between. `sink(frame, caption)` sees every frame (the map video)."""

    def __init__(self, state="Start", sink=None):
        import stable_retro as retro

        self.env = retro.make(GAME, state=state, use_restricted_actions=retro.Actions.ALL, render_mode=None)
        self.obs = self.env.reset()[0]
        self.names = list(self.env.buttons)
        self.sink, self.caption, self.frames = sink, "", 0
        self.panel = None            # the network picture shown beside the game while a trained genome plays (set by the level player)
        self.level_save = None       # the game as the current level began: a death loads it again instead of costing a life
        self.save_level()

    # --- the emulator
    def frame(self, *pressed):
        pad = np.zeros(12, dtype=np.int8)
        for b in pressed:
            pad[self.names.index(b)] = 1
        self.obs = self.env.step(pad)[0]
        self.frames += 1
        if self.sink:
            self.sink(self.obs, self.caption)

    def frame_mask(self, mask):
        """One frame with the buttons of a recorded 12-bit input mask (the bit i is the emulator button i)."""
        self.obs = self.env.step(np.array([(mask >> b) & 1 for b in range(12)], dtype=np.int8))[0]
        self.frames += 1
        if self.sink:
            self.sink(self.obs, self.caption)

    def hold(self, frames, *pressed):
        for _ in range(frames):
            self.frame(*pressed)

    def wram(self):
        return R.wram(self.env.unwrapped.get_ram())

    def u8(self, address):
        return int(self.wram()[address])

    def u16(self, address):
        w = self.wram()
        return int(w[address]) | int(w[address + 1]) << 8

    def save_level(self):
        self.level_save = self.env.unwrapped.em.get_state()

    def retry(self):
        """Back to the start of the level, with the lives and everything else as they were then. Two idle frames, as every episode starts."""
        self.env.unwrapped.em.set_state(self.level_save)
        self.hold(2)

    # --- what is going on
    @property
    def mode(self):
        return self.u8(MODE)

    @property
    def position(self):
        return self.u16(OW_X), self.u16(OW_Y), self.u8(SUBMAP)

    @property
    def lives(self):
        return self.u8(LIVES)

    @property
    def level_name(self):
        return LEVELS.get(self.u8(TRANSLEVEL), f"level {self.u8(TRANSLEVEL):02x}")

    # --- the map
    def wait_map(self, limit=3000, settle=90):
        """Run idle frames until Mario stands still on the map (the end-of-level walk and fades take about 700 frames). True when he does."""
        stable, last = 0, None
        for _ in range(limit):
            self.frame()
            here = (self.mode, self.position)
            stable = stable + 1 if here == last and self.mode == MODE_MAP else 0
            last = here
            if stable >= settle:
                return True
            if self.mode != MODE_MAP and self.lives == 0 and self.mode not in (0x0B, 0x0C, 0x0D, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14):
                return False                                    # game over or the title screen
        return False

    def walk(self, direction):
        """One step on the map: press the direction until Mario starts to move, let him walk to the next stop. True when he moved at all."""
        start = self.position
        for _ in range(30):
            self.frame(direction)
            if self.position != start:
                break
        moved = self.position != start
        stable, last = 0, self.position
        for _ in range(300):
            self.frame()
            stable = stable + 1 if self.position == last else 0
            last = self.position
            if stable >= 20:
                break
        return moved

    def enter(self):
        """Press A on the map. The level's name when a level is entered (the game is then in the level), None when nothing happens."""
        self.hold(2, "A")
        for _ in range(60):
            self.frame()
            if self.mode != MODE_MAP:
                break
        else:
            return None
        for _ in range(900):
            if self.mode == MODE_LEVEL:
                break
            self.frame()
        else:
            return None
        self.hold(2)
        self.save_level()
        return self.level_name
