"""Level tile map and per-generation paths for the map view."""
import json
from pathlib import Path

from .. import config, ram as R, train

ROWS = 27   # tile rows per page in horizontal levels
BOX_TILES = {0x24, 0x26}        # Map16 0x124 / 0x126: boxes seen turning into used blocks when hit
YOSHI_COIN_TILES = {0x2D, 0x2E}  # the two halves of a big coin


def _tile_char(low, high):
    if high == 1:
        if low in R.PIPE_TILES:
            return "P"                                # pipe piece (Map16 0x133-0x13F)
        return "?" if low in BOX_TILES else "#"      # box with something in it / solid ground and blocks
    if high == 0 and low == 0x25:
        return "."        # empty sky
    if high == 0 and low == R.COIN_TILE:
        return "c"        # coin
    if high == 0 and low in YOSHI_COIN_TILES:
        return "Y"        # big coin (both halves)
    return "o"            # decoration, pipes, other blocks


def level_map(run_dir):
    """Whole-level tile map as ROWS strings of pages*16 characters (cached per level)."""
    run_dir = Path(run_dir)
    cfg = config.load(run_dir / "config.toml")
    level = cfg.env.levels[0]
    cache = run_dir.parent / "_levels" / f"{level}.v5.json"     # v5: pipes marked (tile 0x133 included)
    if cache.exists():
        return cache.read_text()
    train.use_config(cfg.env)
    env = train._get_env()
    env.reset()
    wram = R.wram(env._env.unwrapped.get_ram())
    pages = max(1, int(wram[0x5D]))
    rows = []
    for y in range(ROWS):
        line = []
        for page in range(pages):
            base = page * R.TILES_PER_PAGE + y * 16
            for x in range(16):
                line.append(_tile_char(int(wram[0xC800 + base + x]), int(wram[R.MAP16_HIGH + base + x])))
        rows.append("".join(line))
    out = json.dumps({"level": level, "pages": pages, "rows": rows})
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(out)
    return out


def run_path(run_dir, generation, events=False):
    """x/y path of a generation's best run (recorded at training time, else replayed once).

    With events=True also returns what the run collected ([frame, kind, x, y]: coin, yoshi, powerup,
    block); runs recorded before events existed are replayed once and the result is cached."""
    run_dir = Path(run_dir)
    f = run_dir / "best" / f"gen_{generation:04d}.json"
    rec = json.loads(f.read_text())
    if "path" not in rec or (events and "events" not in rec):
        train.use_config(config.load(run_dir / "config.toml").env)
        _, info = train.replay_actions(rec["actions"])
        rec["path"] = rec.get("path") or info["path"]
        rec["events"] = info.get("events", [])
        f.write_text(json.dumps(rec))
    out = {"generation": rec["generation"], "fitness": rec["fitness"], "max_x": rec["max_x"],
           "cleared": rec["cleared"], "path": rec["path"]}
    if events:
        out["events"] = rec.get("events", [])
    return out
