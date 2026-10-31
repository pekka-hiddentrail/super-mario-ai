# super-mario-ai

Teach an agent to play Super Mario World (USA) using NEAT and modern reinforcement learning,
in Python on a headless SNES emulator (planned: [stable-retro](https://github.com/Farama-Foundation/stable-retro)).

Goal order: Yoshi's Island 1 first, then more levels.

See [docs/FEATURES.md](docs/FEATURES.md) for everything that exists and how the parts fit together, and
[docs/ROADMAP.md](docs/ROADMAP.md) for findings and plans.

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
bash scripts/wsl_install_deps.sh   # inside WSL
scripts/test.sh
```

Put your ROM in `roms/` (not committed). Expected SHA1: _TBD after verification._

## Development

- Work on feature branches (`feat/...`, `fix/...`, `chore/...`), merge to `main` via PR.
- `ruff check .` and `pytest` before committing.

## Regenerating the RAM map

```bash
python tools/parse_ram_map.py docs/smw_ram_map.html docs/smw_ram_map.json
```

## Daily use (inside WSL, from the repo root)

```bash
scripts/ui.sh                   # dashboard: http://localhost:8501 (curves, train form, replay, path map)
scripts/smwai.sh train          # or train from the CLI with configs/level1.toml
scripts/view.sh                 # replay viewer only: http://localhost:8765
scripts/test.sh                 # tests
```

## Experiments (inside WSL)

```bash
scripts/experiment.sh configs/experiments/ladder_yi3.toml    # variants x seeds, one summary table
scripts/experiment.sh configs/experiments/ladder_yi3.toml --summary-only
scripts/video.sh                                             # mp4 of each run's best generation
scripts/restart_viewer.sh                                    # replay / path map / videos on :8765
```
An experiment spec names a base config and variants that each override single settings. Every run is
saved under `runs/<YYYYMMDD-HHMM>-<experiment>-<variant>-s<seed>/` (the stamp is the launch time) and the table lands in `runs/_experiments/<stamp>-<name>/summary.md`. `--stamp` continues or re-summarises an earlier launch.
Metrics include emulator frames used (`frames`, `total_frames` in `metrics.csv`), so methods can be compared per frame of experience.
