# Roadmap / feature list

Status: done = in `main`/branch, next = planned order, later = wanted but not scheduled.

## Done
- Python project, WSL2 + stable-retro emulator, manual play tool
- RAM reader, tile grid, Gymnasium env, configurable reward
- NEAT v1: CLI training, parallel workers, per-generation best-run recording, checkpoints

## Next
1. ~~**Replay viewer**~~ done (`scripts/view.sh`, http://localhost:8765) - play a recorded generation in the browser (game frame + network overlay
   like MarI/O: input grid, hidden nodes, button outputs, green/red weights, live activations).
2. ~~**Level map + path view**~~ done (http://localhost:8765/map). Still to do: real graphics, path marks for power-ups.
3. **Streamlit dashboard** - training curves, parameter editing, run browser, embeds the viewer.
4. Live parameter tuning between generations (reward weights, mutation rates).

## Level map and path view (requested)
- Draw the **whole level** as one picture (horizontal scrolling levels first).
  - Source: the Map16 tile table in RAM (already readable); draw solid vs. empty first,
    real graphics later (sprite sheets provided by the user - location to be confirmed).
- Overlay the **path Mario took**: x/y per frame from a replay of each recorded run.
  - Better runs drawn darker/more opaque, worse runs lighter (opacity from fitness rank).
  - Selectable: one generation, a range of generations, or best-per-generation.
  - Mark deaths (where each run ended), power-up pickups and the goal.
- Needs: store x/y trajectory per recorded run (cheap; add to `best/gen_NNNN.json`).

## Learning ideas (experiment ladder)
- Reward: time penalty so faster clears score higher (once the level is cleared, plain
  progress reward has nothing left to optimise).
- Reward: power-up and coin bonuses, off by default.
- Exploration: novelty or checkpoint rewards when progress stalls.
- Inputs: separate channels for solid / enemy / item instead of one mixed grid.
- Compare NEAT settings (species threshold, mutation rates) on the same chart.

## Later
- Multi-level sequences (advance to the next save state on clear).
- PPO via stable-baselines3 on the same env, same dashboard.
- Other levels and vertical levels (tile layout differs).
