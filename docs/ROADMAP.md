# Roadmap / feature list

Status key: **done** = built and committed, **next** = planned order, **later** = wanted but not scheduled.
Sections "Making the agent more intelligent" and "Suggested order" are a longer-term plan written in
a separate session; they are kept as written, with status notes added where something has since been built.

Naming note: the early configs and runs called `level1` use the save state `YoshiIsland2` (the easy
level). The experiments now use `YoshiIsland3` (`configs/yi3.toml`). `yi1*` runs use `YoshiIsland1`.

## Done

### Foundation
- Python project (`src/smwai`, tests, pyproject, gitignore), WSL2 Ubuntu with stable-retro (no Windows wheels), ROM imported.
- Manual play tool (`scripts/play_manual.*`), helper scripts for WSL (`scripts/*.sh`), test runner (`scripts/test.sh`, 36 tests).
- SMW RAM map parsed from the saved SMW Central page (`docs/smw_ram_map.json`).

### Environment (`env.py`, `ram.py`)
- Gymnasium env on stable-retro; state read from WRAM; 8 buttons; frame skip 4.
- Input modes (`env.inputs`): `mixed` (one tile grid, 169 inputs), `channels` (solid/enemy/item grids + water, Yoshi, power-up),
  `collect` (adds a coin grid and typed sprites: enemies, items, Yoshi/extended sprites skipped). Old modes keep their behaviour so old runs replay.
- Reward terms (all in the config): progress, clear bonus, time penalty, death penalty, power-up bonus,
  coin bonus, **big (Yoshi) coin bonus**, **box-hit bonus**, **speed bonus** (rightward speed, from `$7B`),
  **fast-clear bonus** (shrinks with frames), **goal bar bonus** (per bonus star; the game records how high the bar was hit in `$1900`). Collecting something worth reward resets the stuck timer, so going back for it is possible.
- Collection **events** (coin, big coin, power-up, box hit with position and frame) are recorded with every best run.
- Verified in game memory: coin = Map16 `0x2B`; big coin = pair `0x2D`/`0x2E` (counter `$1420`); a hit box turns into the
  used-block tile `0x132` (box tiles seen: `0x124`, `0x126`). Other box types are not yet verified.

### Training
- NEAT (neat-python): CLI `smwai train`, parallel workers, per-generation best-run recording (buttons, path, events), checkpoints,
  resume (`scripts/resume_run.sh`), seeds, frame counting (`frames`, `total_frames` in `metrics.csv`).
- **Adaptive mutation** (`[adapt]` in a config, off by default): when the best fitness has not improved by at least
  `min_improvement` for N generations, mutation is boosted until it does; logged to `events.jsonl`.
  Best result so far on Yoshi's Island 1 (3 of 3 seeds better than baseline); see Findings. Recommended config: `configs/yi1_adaptive.toml`.
- PPO (stable-baselines3, CPU only: this PC has only an Intel GPU): `smwai train-ppo`, button combos, same run folder format.
  First version; eval is unstable and there is no resume.
- Experiment runner (`smwai experiment`, `configs/experiments/`): variants x seeds under one budget, summary table in
  `runs/_experiments/<name>/summary.md`. Pilot ladder on Yoshi's Island 3 is defined (`ladder_yi3.toml`).

### Tools
- Replay viewer (`:8765`): game frame + network overlay (input grids, hidden nodes, buttons, weights, activations); PPO runs show buttons only.
- Path map (`:8765/map`, opens already drawn): whole level from the Map16 tables, paths per generation (darker = fitter, best in orange),
  death marks, goal flag, **big coins and boxes marked, and what the best run collected** (dot = coin, ring = big coin, triangle = power-up, diamond = box hit).
- Best-run videos (`scripts/video.sh`, `:8765/videos`): mp4 at 60 fps with a caption strip.
- Streamlit dashboard (`scripts/ui.sh`, `:8501`): curves for several runs, train form (NEAT and PPO), replay, path map, videos.
- Helper scripts: `stop_run.sh`, `list_runs.sh`, `restart_viewer.sh`, probes (`probe_levels.py`, `probe_collectibles.py`, ...).

## Findings so far
- Yoshi's Island 2 is easy: NEAT clears it in 2-6 generations; a naive hold-right script clears it. Not a useful benchmark.
- Time penalty 0.2/frame made clears about 10% faster (about 1900 -> 1700 frames), then plateaued.
- Yoshi's Island 3 (21 screens): every variant of the pilot ladder (collect inputs, lower species threshold, more patience) stalled
  near x 530-550 after 15 generations (means 528-550, within noise). It needs a structural fix (climbing), not tuning.
  The Yoshi's Island 1 route was chosen instead.
- Yoshi's Island 1: NEAT used to plateau around x 3550. Where it ends: around x 3580 a giant Banzai Bill flies in, then the run drops into a gap at x ~3677.
- **Yoshi's Island 1 ladder** (collect inputs, population 50, 14 generations, best x; seeds 1-2): mutate-more (static higher
  mutation) 2721 / 4826 (one clear); wide view (17x17) 3433 / 1266; speed bonus 852 / 2955; baseline 858 / 2210; adaptive (first
  version) 858 / 1758. The spread between seeds alone was larger than most differences.
- **Confirmation on seeds 3-5**: adaptive mutation (with the min-improvement fix) beat the baseline on all three seeds
  (2808 / 3674 / 3432 vs 1826 / 1890 / 1991; means 3305 vs 1902). The static higher mutation did **not** hold up
  (2456 / 980 / 847; mean 1428), so its earlier clear was luck. None of the confirmation runs cleared.
  The first adaptive version never triggered because tiny fitness gains reset its stall counter; fixed by `min_improvement`.
- **Yoshi's Island 1 cleared** by a long adaptive-mutation run (`configs/yi1_adaptive.toml`, population 100, seed 1): first clear at generation 23
  after about 889,000 emulator frames (about 11 minutes with 12 workers). Each jump in best fitness (gens 8, 14, 23) came right after a
  boost period. The clearing run takes about 2,240 frames; it happened to pick up 2 coins, 2 big coins and 1 power-up (no collect rewards were set).
  One seed; best-ever x had sat at 3039 -> 3416 -> 4152 before.
- PPO vs NEAT on Yoshi's Island 2 (same reward and `collect` inputs): NEAT cleared in 2 generations (roughly 100k frames, estimated);
  PPO cleared at about 270k steps (about 1.1M frames) and only in some evaluation rounds (3 of 20), so NEAT was about 10x more
  frame-efficient there. One seed each.
- Experiments are small (2-3 seeds, 14 generations): they rank ideas, they do not prove them.

## Known limits
- Background jobs stop when the Claude session ends and WSL is restarted; use checkpoints / `resume_run.sh`.
- Tests need WSL (Windows Python lacks gymnasium and stable-retro).
- Horizontal levels only (the tile map layout of vertical levels differs).
- The network sees +-6 tiles around Mario (96 px), so things farther back cannot be seen (matters for "go back for the coin").
- PPO sees one frame only (no speed, no memory).

## Next (in order)
1. Repeat the long adaptive run on Yoshi's Island 1 with more seeds (the single clear could be luck), then add speed and collect rewards on top.
2. More seeds for adaptive mutation (5+), tune `stall_generations` and the boost size; try it together with the wider view (mixed results so far).
3. Speed and ground inputs (X/Y speed, in-air flag) in one observation version (see "Making the agent more intelligent", item 1).
4. Try the box-hit and big-coin rewards (`collect` configs) and check that the path map shows them being collected.
5. Wider forward-looking input window (more tiles ahead than behind) as a ladder variant.
6. Multi-level sequences (advance to the next save state on clear).
7. PPO: resume, frame stacking, stable evaluation.
8. Real graphics for the level map from the sprite sheets (location of the sprite sheet downloads still to be decided).

## Level map and path view
Done (see above). Possible additions: compare two runs on one map, heat map of deaths over many runs, real graphics.

## Learning ideas (experiment ladder)
Status: runner built; ladders run on Yoshi's Island 3 (nothing helped) and Yoshi's Island 1 (adaptive mutation helped; see Findings).
- Reward: time penalty (done, modest effect), speed bonus and fast-clear bonus (built, not yet in a ladder run).
- Reward: power-up, coin, big coin and box-hit bonuses (built; try on `collect` runs).
- Exploration: novelty or checkpoint rewards when progress stalls (not built).
- Inputs: separate channels (done), coins (done), speed/ground (next).
- NEAT settings: species threshold (note: 1.5 gives one species per genome at population 60 and crashes NEAT), stuck patience,
  adaptive mutation (built, best so far), static higher mutation (did not replicate), wider view (mixed), speed bonus (mixed).

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
- Status: water and Yoshi flags are already inputs in `channels`/`collect` mode; the rest is not built.

### 1. Context and speed inputs (cheap, do first)
- Add Mario's X/Y speed (`$7B`, `$7D`), the in-air flag (`$72`), facing direction, level id,
  room id and game mode to the observation. Version the observation layout so old runs replay.
- Make all observation changes in one step, because every change invalidates earlier comparisons.
- Status: X speed is read for the speed reward only; it is not yet a network input. Layout versioning exists (`inputs` modes).

### 2. Exploration: reaching things that are not to the right
- **Novelty reward**: pay for visiting a new (room, 16-pixel tile) cell, not only for a new furthest x.
  Rewards going up, back or into pipes. Works for NEAT and PPO.
- **Novelty search / MAP-Elites (NEAT)**: keep many *different* behaviours alive instead of one best.
- **Go-Explore style archive**: save emulator states at interesting cells, start new attempts from them.
  The emulator is deterministic, so this is exact. The same archive supplies start states for
  harder sections and for multi-level training.
- Curiosity/RND bonus for PPO.
- Status: not built. (Collectible rewards plus the stuck-timer reset allow some going back, but only within the view window.)

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
  Note: PPO currently uses 10 button combinations, so a mapping would be needed.
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
- Other levels and vertical levels (tile layout differs).
- Live tuning for PPO runs.
