# Features and how they fit together

What exists, where to find it, and what is still rough. Findings and plans are in [ROADMAP.md](ROADMAP.md).
Everything runs inside WSL (Ubuntu) except the browser; scripts are in `scripts/`.

## The game and what the agent sees (`env.py`, `ram.py`, `jumpview.py`, `physics.py`)
| Input mode (`env.inputs`) | What the network gets |
|---|---|
| `mixed` | one tile grid (solid, enemy, item), the original MarI/O view |
| `channels` | separate solid / enemy / item grids + in-water, on-Yoshi, power-up |
| `collect` | + a coin grid, typed sprites (enemies vs items; Yoshi and particles ignored) |
| `physics` | + speeds, on-ground flag, and for jumps A/B held 1/10/25 frames: lands / pit / enemy / coins / landing offset (from the measured jump formula) |
| `explore` (recommended) | `physics` + a **pipe** grid (green pipes are told apart from ground) |

- View size: `radius` (square) or `window = [left, right, up, down]` tiles. A taller or shifted window did not help (see roadmap).
- `room_progress = true`: every room (main level, bonus room) keeps its own furthest x. The no-progress cut-off pauses during pipe animations and room loads.
- Older modes keep their exact behaviour, so every earlier run still replays.

## Rewards (`[reward]` in a config; all default to 0 except progress and the clear bonus)
progress per pixel, clear bonus, time penalty, death penalty, power-up, coin, big (Yoshi) coin, box hit, **goal bar** (per bonus star: higher hit = more),
**climb** (per pixel of new standing height), **pipe** (entering a pipe, once per pipe; use about 3000), speed bonus, fast-clear bonus, water penalty (off, undecided).
Every collection is recorded as an event (coin, big coin, power-up, box, pipe, climb) with the run, for the path map.

## Learning (`train.py`, `ppo.py`, `experiment.py`)
- **NEAT**: `smwai train --config configs/...toml`; parallel emulators, per-generation best run (buttons, path, events), checkpoints, seeds, frame counting.
- **Adaptive mutation** (`[adapt] enabled = true`): mutate harder when the best has not gained `min_improvement` for N generations. Best finding so far on Yoshi's Island 1.
- **PPO** (CPU): `smwai train-ppo`; button combinations; same run format. First version, unstable.
- **Experiments**: `smwai experiment --spec configs/experiments/<name>.toml` runs variants x seeds and writes a table; `scripts/count_events.py`, `scripts/pipe_summary.py` tally pipe events.

## Looking at results (browser)
| Page | Address | What it shows |
|---|---|---|
| Dashboard | `localhost:8501` | curves of several runs, Train form (every setting above incl. input mode, adaptive mutation, per-room progress, window), start/stop, embedded pages |
| Replay viewer | `localhost:8765/` | a recorded run with the network drawn live (grids, hidden nodes, connections, buttons) |
| Path map | `localhost:8765/map` | whole level with paths (darker = fitter), deaths, goal, coins, big coins, boxes, **pipes**, what the best run collected; paths through pipes are cut at the room change |
| Videos | `localhost:8765/videos` | mp4 of each run's best generation with the network next to the game (`scripts/video.sh`) |

## Habits and helpers
`scripts/test.sh` (67 tests), `scripts/check_features.sh` (end-to-end check of replay, path, level map, videos, dashboard on a pipe run), `scripts/restart_ui.sh`,
`scripts/restart_viewer.sh`, `scripts/stop_run.sh`, `scripts/resume_run.sh`, `scripts/list_runs.sh`, probes (`probe_*.py`, `validate_jumps.py`) for measuring the game.
After changing code, restart the servers (`restart_ui.sh`): a running page does not pick up new settings or input modes by itself.

## Rough edges
- Downward pipes are hard to find (need a deliberate DOWN press); sideways pipes are found by running into them.
- Yoshi's Island 3 (a chasm to a higher ledge) is unsolved. Running past the pipe entrance on Yoshi's Island 4 is a dead end.
- The jump model ignores walls, bounces and moving enemies (land/pit right 97%, landing within a tile 65%); some sprites that are not enemies count as enemies.
- Room fingerprints (screen count + solid tile count) could collide for two different rooms; per-room progress only exists with `room_progress`.
- PPO has no resume and an unstable evaluation. Experiments use 2-5 seeds, so they rank ideas, they do not prove them.

## Live view (`smwai live`, `scripts/restart_live.sh`)
http://localhost:8767 plays the newest saved best genome of the newest run (or `--run NAME`) in real time, with the network next to the game, and switches to each new generation (and each new run of an experiment) as it is saved. It runs in its own process with its own emulator.
- Runs that save their population (`[live] save_population = true`, written to `population.pkl` each generation) are shown genome by genome, species by species, failures included; the caption says which species (k of n, how many genomes), which genome, its fitness and rank.
- Other runs: the best genome of the newest generation.
- `scripts/showcase.sh` starts a training run just for watching (Yoshi Island 1, 3 workers, `configs/showcase_yi1.toml`) together with the live view; other experiments can run alongside.

## Video network panel
Input windows that are tall or have many single values (`ride`, `run_options`, ...) no longer get cut off: the panel is laid out on a larger canvas and shrunk to fit.

