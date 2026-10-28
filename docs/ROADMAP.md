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
  **fast-clear bonus** (shrinks with frames), **goal bar bonus** (per bonus star; the game records how high the bar was hit in `$1900`), **climb bonus** (`height_bonus`: per pixel of
  new standing height, i.e. landing on something higher than ever before) and **pipe bonus** (`pipe_bonus`: entering a pipe, once per pipe per
  episode; pipe entry = pipe timer `$88` starting with action `$89` in 1-4: 3 = downward (Yoshi's Island 2, x 3878) and 1 = sideways
  right (Yoshi's Island 4, x 2723) are measured, 2 and 4 are assumed; coming out is 5 or 7 and pays nothing). Collecting something worth reward resets the stuck timer, so going back for it is possible.
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
- **The network cannot see the ground at the top of a high jump**: the view covers 6 tiles up and down, and at x 2621 on
  Yoshi's Island 1 the ground is 7 tiles below Mario at the apex (the solid grid is empty there). A taller or lower-reaching
  window is a cheap candidate improvement (see Next).
- **Repeat of the long adaptive run** (Yoshi's Island 1, collect inputs, population 100, 30 generations): seed 1 cleared at
  generation 23; seed 4 cleared at generation 9 (about 238,000 frames); seed 2 reached x 4169 and seed 3 x 3554 without clearing.
  So 2 of 4 seeds cleared and 3 of 4 got clearly past the old plateau (x ~3550). There is no no-boost baseline of the same size
  yet (the earlier no-boost runs were population 100 with other inputs and one seed: x 3538 / 3661).
- **Shifted view window** (`window = [4, 8, 4, 8]`: 8 tiles ahead and 8 below, same 169 cells per grid; collect inputs + adaptive mutation,
  seeds 3-7, population 50, 14 generations): mean best x 2583 against 2990 for the symmetric window; better on 3 seeds (+734, +122, +96),
  worse on 2 (-1760, -1225), no clear. No evidence that the taller window helps; the seed noise is larger than the effect. The symmetric
  runs on seeds 3-5 reproduced the earlier numbers exactly (2808 / 3674 / 3432), confirming runs are deterministic per seed.
- **Jump options as inputs** (`inputs = "physics"`, built; `jumpview.py`): from the measured jump formula, per jump (A/B held 1/10/25 frames)
  land-or-pit, enemy on the arc, coins, landing offset. Checked against the emulator on 216 jumps: land-or-pit right 97%, landing within
  one tile 65%. B jumps harder than A (takeoff 77..90 vs 71..82).
  **Result on Yoshi's Island 1** (seeds 3-7, same settings as the symmetric runs): mean best x 3690 against 2990; better on 4 of 5 seeds
  (+1788, +243, +192, +2794), worse on 1 (-1515); no clear in either. Encouraging but not proven at 5 seeds. Test on the gap levels
  (Yoshi's Island 3 and 4, `configs/experiments/gaps_yi34.toml`) is running.
- **Gap levels** (jump inputs against collect+adaptive baseline, seeds 1-3, 15 generations, population 60): Yoshi's Island 3 mean best x
  545 (physics) vs 546 (baseline): no difference, every one of the 6 runs stalls at x 540-550 where a chasm has to be crossed to a ledge
  higher than the takeoff. Yoshi's Island 4: 2793 vs 2609 (physics reached the plateau on 3 of 3 seeds, baseline on 2 of 3).
  On Yoshi's Island 4 the plateau (x ~2793) is caused by the way on being a **sideways pipe** at x ~2723, which the runs jump over.
  Pipes can be mandatory path elements, not only bonuses.
- **Pipe reward without perception does not work**: on Yoshi's Island 2 (baseline vs `pipe_bonus` 100000 + climb bonus, 3 seeds, 15
  generations) all 6 runs cleared the level but none of the 90 recorded best runs entered the pipe at x 3878; with only solid-block
  inputs a pipe looks like any other block. The `explore` input mode adds a pipe channel (Map16 0x134-0x13F) on top of `physics`;
  its test (`configs/experiments/explore_pipes.toml`) is queued behind the Yoshi's Island 4 pipe test.
- **Pipe bonus on Yoshi's Island 4** (collect inputs, baseline vs `pipe_bonus` 100000 vs pipe + jump inputs; 2 seeds, 12 generations, no pipe
  channel): all four pipe runs found a pipe (best fitness about 100,800-102,700); the baselines found none. Seed 2 entered the sideways pipe at
  x ~2723 (the way on); seed 1 took an earlier pipe near x ~810 and then never progressed (best x 811 / 809), because 100000 for one pipe beats
  any amount of progress. So sideways pipes are found by running into them, downward pipes (which need DOWN) were not found on Yoshi's Island 2.
  Lesson: a very large bonus can trap a run in a bonus room; progress inside the new room (per-room max x) is still unrewarded.
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
5. (done: shifted window did not help, see Findings) Input window variants.
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

**Ground rules (apply to every item below):**
- **AI only.** No human in the loop: no human demonstrations, no hand-labelled data. Everything is learned from
  the agent's own play and its reward.
- **Organic understanding, not tricks.** No tool-assisted speedrun methods: no save-state search or
  rewind-and-retry as a planner (beam search, MCTS over states, Go-Explore archives), no frame-perfect scripts, no
  hard-coded routes, no scripted bots. The agent only sees what a player could see (screen contents via game
  memory: tiles, sprites, speeds) and only presses buttons. Every attempt starts from the level start.
- **Knowledge of the game is allowed as a model, not as a script.** Physics measured from the game (item 3) is a
  tool the agent *uses to understand* a jump, the way a player learns jump arcs. It never decides the button presses.

### 0. Principle: one model does not fit every situation
A NEAT network trained on a side-scrolling level does not work in a bonus room, a pipe
section, a boss room, the world map or a mini game: the goal, the coordinates and often the
controls differ. So the design is **a context router plus one specialist per context**:

| Context | How we detect it (game memory) | Specialist |
|---|---|---|
| Normal level, side-scrolling | game mode = in level, level/room id | NEAT / PPO policy per level or level family |
| Bonus room, pipe room, sub-area | room/screen id changes while the level continues | own small learned policy, goal = the exit |
| Boss room (e.g. Bowser, Koopalings) | boss sprite present, room id | own policy; boss sprite IDs still to be verified |
| Water level | in-water flags (`$75`, `$85`) | own policy (different physics) or same policy with a water input |
| World map / overworld | game mode != in level | own learned policy, goal = enter the next level node (needs a reward for reaching a new level) |
| Mini game | game mode / room id | own policy, trained with RL from a reward read from game memory (score, lives, bonus) |

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
- Curiosity/RND bonus for PPO.
- Multi-level training starts every attempt at a level's start (several levels in rotation), not from saved mid-level states.
- Status: not built. (Collectible rewards plus the stuck-timer reset allow some going back, but only within the view window.)

### 3. Prediction: where will a jump land? (formula measured, built as `physics.py`)
A jump is plain constant-acceleration motion, so it is a formula, not a search. Measured from game memory with
`scripts/probe_jump.py` (units: 1/16 pixel per frame, positive = down):
- Every frame: `y_speed += gravity`, then `y += y_speed / 16`, and `x += x_speed / 16` (x speed stays constant in the air).
- Gravity is **3 while the jump button is held** (and fall speed is below 36), **6 once it is released**. Fall speed is capped at 70.
- Takeoff speed depends on how fast Mario is running: -71, -73, -75, -77, -79, -82 for run speed in bins of 8.
- So holding A for H frames is two parabolas joined at release (the t² law):
  `y(t) = v0*t + 1.5*t²` while held, then `y = y_r + v_r*s + 3*s²` after release (s = frames since release), all / 16.
  The jump height is therefore a function of how long the button is held: e.g. standing, a 1-frame tap rises 28 px,
  a 10-frame hold 45 px, and the maximum is 55 px (67 px at full run).
- Verified against the emulator: predicted apex height is within 1 px for 11 hold/speed combinations (`tests/test_physics.py`).
- Use for the agent: for a few standard holds (tap, short, medium, full) compute the arc from the current x speed,
  walk the arc over the tile grid and sprite positions, and give the network **per option** a landing summary as
  inputs: lands on solid ground / falls in a pit / touches an enemy / passes a coin, plus landing offset (dx, dy).
  The agent still decides *when* to jump and *how long to hold*. The model only helps it read the world, like a
  player who has learned how far Mario flies. Rollouts in the emulator are not used.
- Still to measure: air steering (holding left/right changes x speed slowly), spin jump, water, Yoshi, cape flight,
  bounce off enemies, slopes. Each needs its own probe, same method.
- Status: formula and tests done. Not yet wired into the observation.

### 4. Memory
- Recurrent policy for PPO (`RecurrentPPO`, sb3-contrib). NEAT can also grow recurrent networks
  (set feed-forward off). Lets the agent remember "I already went through that pipe".

### 5. Learning from its own experience
- NEAT -> PPO: replay NEAT's recorded best runs to pre-train the PPO network (behaviour cloning),
  then continue with PPO. This is AI teaching AI, no human involved. Needs the same button layout in both
  (8 independent buttons). Note: PPO currently uses 10 button combinations, so a mapping would be needed.
- Always train across several levels or with small randomness (sticky actions): the emulator is deterministic, so a
  single recorded winner would only be a memorised key sequence, not skill.

### 6. Hierarchy and goals (later)
- A high-level choice ("go to the pipe", "reach the goal") plus a low-level policy that gets there
  (goal-conditioned RL). Powerful but hard to get right; try 0-2 first.

### What probably does not pay off yet
- World-model methods (Dreamer-style) and sequence models: heavy, need a lot of data.
- Learning from pixels: we have exact game memory. Only useful to work on other games.

### Not doing (by the ground rules)
- Human demonstrations or human-in-the-loop labelling.
- Save-state search, MCTS/beam search over emulator states, Go-Explore state archives, rewind-and-retry planners.
- Scripted routes, frame-perfect input scripts, hard-coded navigators or glitch exploitation.

### Suggested order
1. Fair benchmark: several seeds, clear rate, steps/generations to clear, clear time, coins
   (the experiment runner is the start of this).
2. Diagnose where runs die on the plateau levels (path map death marks) before choosing a fix.
3. Context + speed inputs and the novelty reward (item 1 + novelty part of 2).
4. Jump-arc inputs from the physics model (item 3), tested on a level where runs die to jumps.
5. Router and per-context specialists (item 0), starting with level vs. bonus room.
6. Multi-level sequences (every attempt from a level start).
7. PPO baseline on the same benchmark, then pre-training from NEAT runs (item 5).
8. Recurrent PPO, hierarchy.

Rule: one change at a time, measured on at least three seeds and two or three levels.

## Later
- Other levels and vertical levels (tile layout differs).
- Live tuning for PPO runs.
