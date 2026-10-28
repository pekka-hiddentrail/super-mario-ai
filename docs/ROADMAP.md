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
3. ~~**Streamlit dashboard**~~ done (`scripts/ui.sh`, http://localhost:8501) - training curves, parameter editing, run browser, embeds the viewer.
4. ~~Live parameter tuning~~ done (dashboard "Live tuning" tab; applied at the next generation).

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
Status: runner built (`smwai experiment`), pilot ladder on Yoshi Island 3 in `configs/experiments/ladder_yi3.toml`.
Speed rewards exist (`speed_bonus`, `clear_time_bonus`) but are not yet part of a ladder run.
- Reward: time penalty so faster clears score higher (once the level is cleared, plain
  progress reward has nothing left to optimise).
- Reward: power-up and coin bonuses, off by default.
- Exploration: novelty or checkpoint rewards when progress stalls.
- Inputs: separate channels for solid / enemy / item instead of one mixed grid.
- Compare NEAT settings (species threshold, mutation rates) on the same chart.

## Making the agent more intelligent (plan)

Today one network plays one level from its start. It only knows "go right". To handle other
levels, bonus rooms, mini games and things that are not to the right, the agent needs to know
**where it is** (context), **what it has already tried** (memory and exploration), and
**what a move will lead to** (prediction). The ideas below are ordered by value for effort.
None of this is built yet unless marked.

### 0. Principle: one model does not fit every situation
A NEAT network trained on a side-scrolling level does not work in a bonus room, a pipe
section, a boss room, the world map or a mini game: the goal, the coordinates and often the
controls differ. So the design is **a context router plus one specialist per context**:

| Context | How we detect it (game memory) | Specialist |
|---|---|---|
| Normal level, side-scrolling | game mode = in level, level/room id | NEAT / PPO policy per level or level family |
| Bonus room, pipe room, sub-area | room/screen id changes while the level continues | own small policy or scripted routine, goal = the exit |
| Boss room (e.g. Bowser, Koopalings) | boss sprite present, room id | own policy; boss sprite IDs still to be verified |
| Water level | in-water flags (`$75`, `$85`) | own policy (different physics) or same policy with a water input |
| World map / overworld | game mode != in level | map navigator (graph search to the next level), not a neural net |
| Mini game | game mode / room id | own policy, trained from imitation of human play |

- The router is rule-based (no learning): read game mode, level id, room id and a few flags,
  and pick the specialist. Every specialist keeps its own genome/network, config and run folder.
- Progress rewards must be per context: "furthest x" is meaningless in a room that does not
  scroll right. Each specialist defines its own goal (reach exit, collect coins, survive).
- Needs: context fields in the environment (level id, room id, game mode, in-water, boss),
  a `contexts.toml` mapping context -> specialist, and a runner that switches between them.

### 1. Context and speed inputs (cheap, do first)
- Add Mario's X/Y speed (`$7B`, `$7D`), the in-air flag (`$72`), facing direction, level id,
  room id and game mode to the observation. Version the observation layout so old runs replay.
- Make all observation changes in one step, because every change invalidates earlier comparisons.

### 2. Exploration: reaching things that are not to the right
- **Novelty reward**: pay for visiting a new (room, 16-pixel tile) cell, not only for a new furthest x.
  Rewards going up, back or into pipes. Works for NEAT and PPO.
- **Novelty search / MAP-Elites (NEAT)**: keep many *different* behaviours alive instead of one best.
- **Go-Explore style archive**: save emulator states at interesting cells, start new attempts from them.
  The emulator is deterministic, so this is exact. The same archive supplies start states for
  harder sections and for multi-level training.
- Curiosity/RND bonus for PPO.

### 3. Prediction: where will a jump land?
- Ask the emulator itself: save the state, press a candidate action for 30-60 frames, record the
  outcome (landing x/y, died in a pit, hit an enemy, coins collected), restore the state.
  Exact, but too slow to run at every step during training (about 10x slower).
- So: collect many such examples offline, train a small predictor, and give NEAT/PPO its answers for
  a few standard jumps (short, high, running, spin) as extra inputs. The agent then chooses among
  labelled options instead of reading physics from raw tiles.
- A hand-written physics model is the cheap alternative, but SMW's variable jump height, slopes
  and moving enemies make it inaccurate.
- Short horizon only. Water, Yoshi and cape flight need their own examples.

### 4. Memory
- Recurrent policy for PPO (`RecurrentPPO`, sb3-contrib). NEAT can also grow recurrent networks
  (set feed-forward off). Lets the agent remember "I already went through that pipe".

### 5. Learning from demonstrations
- NEAT -> PPO: replay NEAT's recorded best runs to pre-train the PPO network (behaviour cloning),
  then continue with PPO. Needs the same button layout in both (8 independent buttons).
- Human -> agent: record your own play with the manual-play tool, for mini games and secret exits
  where a reward is hard to write down. Beware: deterministic emulator means a recorded winner is
  a memorised key sequence; train on several levels or add small randomness for real skill.

### 6. Search with the emulator
- Beam search / MCTS over saved states, like a tool-assisted speedrun bot. Finds hidden routes and
  produces demonstrations for imitation. It plans for one level, it does not generalise by itself.

### 7. Hierarchy and goals (later)
- A high-level choice ("go to the pipe", "reach the goal") plus a low-level policy that gets there
  (goal-conditioned RL). Powerful but hard to get right; try 0-2 first.
- Overworld: breadth-first search over the world map graph to the next level node.

### What probably does not pay off yet
- World-model methods (Dreamer-style) and sequence models: heavy, need a lot of data.
- Learning from pixels: we have exact game memory. Only useful to work on other games.

### Suggested order
1. Fair benchmark: several seeds, clear rate, steps/generations to clear, clear time, coins
   (the experiment runner is the start of this).
2. Diagnose where runs die on the plateau levels (path map death marks) before choosing a fix.
3. Context + speed inputs and the novelty reward (item 1 + novelty part of 2).
4. Router and per-context specialists (item 0), starting with level vs. bonus room.
5. Multi-level sequences and the Go-Explore archive.
6. PPO baseline on the same benchmark, then imitation from NEAT (item 5).
7. Jump predictor (item 3), tested on a level where runs die to jumps.
8. Recurrent PPO, overworld navigator, hierarchy.

Rule: one change at a time, measured on at least three seeds and two or three levels.

## Later
- Multi-level sequences (advance to the next save state on clear).
- ~~PPO via stable-baselines3 on the same env, same dashboard.~~ first version done (`smwai train-ppo`, MLP policy on CPU). Ideas: CNN policy, frame stacking / velocity inputs, live tuning for PPO.
- Other levels and vertical levels (tile layout differs).
