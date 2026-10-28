# super-mario-ai

Teach an agent to play Super Mario World (USA) using NEAT and modern reinforcement learning,
in Python on a headless SNES emulator (planned: [stable-retro](https://github.com/Farama-Foundation/stable-retro)).

Goal order: Yoshi's Island 1 first, then more levels.

## Layout

| Path | Purpose |
|------|---------|
| `src/smwai/` | Python package (environment, agents, training) |
| `tests/` | pytest tests |
| `tools/` | One-off scripts (e.g. RAM map parser) |
| `docs/` | SMW RAM map (`smw_ram_map.html` source from SMW Central, `.json` parsed) |
| `legacy/` | Original EmuHawk MarI/O Lua script and its NEAT pool files (reference only) |
| `roms/`, `states/` | Local only, git-ignored (see their READMEs) |
| `runs/` | Training outputs, git-ignored |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
pytest
```

Put your ROM in `roms/` (not committed). Expected SHA1: _TBD after verification._

## Development

- Work on feature branches (`feat/...`, `fix/...`, `chore/...`), merge to `main` via PR.
- `ruff check .` and `pytest` before committing.

## Regenerating the RAM map

```bash
python tools/parse_ram_map.py docs/smw_ram_map.html docs/smw_ram_map.json
```
