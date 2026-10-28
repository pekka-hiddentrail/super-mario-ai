"""Level tile map and per-generation paths for the map view."""
import json
from pathlib import Path

from .. import config, ram as R, train

ROWS = 27   # tile rows per page in horizontal levels


def _tile_char(low, high):
    if high == 1:
        return "#"        # solid ground / blocks
    if high == 0 and low == 0x25:
        return "."        # empty sky
    return "o"            # decoration, coins, ? blocks, pipes top etc.


def level_map(run_dir):
    """Whole-level tile map as ROWS strings of pages*16 characters (cached per level)."""
    run_dir = Path(run_dir)
    cfg = config.load(run_dir / "config.toml")
    level = cfg.env.levels[0]
    cache = run_dir.parent / "_levels" / f"{level}.json"
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


def run_path(run_dir, generation):
    """x/y path of a generation's best run (recorded at training time, else replayed once)."""
    run_dir = Path(run_dir)
    f = run_dir / "best" / f"gen_{generation:04d}.json"
    rec = json.loads(f.read_text())
    if "path" not in rec:
        train.use_config(config.load(run_dir / "config.toml").env)
        _, info = train.replay_actions(rec["actions"])
        rec["path"] = info["path"]
        f.write_text(json.dumps(rec))
    return {"generation": rec["generation"], "fitness": rec["fitness"], "max_x": rec["max_x"],
            "cleared": rec["cleared"], "path": rec["path"]}
