# Plan: overworld map navigation (not started)

Goal: Mario goes from level to level through the game's overworld map like a real playthrough, and eventually a **map navigation network** chooses where to go
(which level to enter, which route to take) on its own. Nothing here is built yet; this is the order of work.

Today: every level starts from a saved state (`env.levels`), the episode ends when the level is cleared, and the curriculum (`curriculum.py`) hands the
population from one level to the next in a fresh process. The map is never visited.

## Phase 0: confirm the curriculum hand-over (prerequisite)
- Replay the Yoshi's Island 2 "clear" of `20261031-1140-curriculum-nuke-s2-L2-YoshiIsland2` and confirm the level id and the finish (its x 4825 is one pixel
  from level 1's 4826, which is probably a coincidence but is unchecked).

## Phase 1: look at the game (research, no code in the training loop)
- Find in `docs/smw_ram_map.json` and by probing: the game mode (`$0100`) values for level, end-of-level screen, overworld; the overworld Mario position
  and the level tile he stands on; the "level cleared" flags; how the exit (normal / secret) changes the open paths. All of this is to be verified, not assumed.
- Record what the game does after the goal tape: how many frames until the map, whether any button is needed on the result screen.
- Output: a short probe script (like `scripts/probe_*.py`) and a table of the addresses in this file.

## Phase 2: the episode continues past the goal
- New env option (off by default so old runs replay): do not end the episode at the clear; run on until the game mode says "overworld".
- Keep the level-fitness untouched: the clear bonus and x progress are scored as now, the map part is not part of the level fitness.

## Phase 3: scripted map walk (the first useful result)
- A map controller that, from the overworld, walks to a target level tile and enters it. Start with a hardcoded route Yoshi's Island 1 -> 2.
- Then a generic version: read the map position and the tile graph from memory and walk a path (search over the tile graph, press the directions).
- Deliverable: a script that plays a genome on Yoshi's Island 1 from its start, walks the map, and the same population plays Yoshi's Island 2, in one
  continuous video, with no saved state for level 2.
- Open issue: Mario now arrives with whatever state he finished (big form, Yoshi), so level 2 differs from the saved state used in training.
  Decide whether training also starts from that carried state.

## Phase 4: map observation and a map network
- Observation for the map: Mario's tile, the neighbouring tiles and which are open / cleared / unreached, the level type (ghost house, castle, normal).
- A second NEAT network (own population, own config) whose outputs are the directions and "enter". Level play stays with the level network.
- Fitness for the map network: levels cleared, new levels reached, exits found, time or lives spent. Needs a way to run the level network inside the
  map episode (a fixed best genome per level first, then the curriculum population).
- Open issues: a failed level (death, game over) must feed back into the map fitness; the map network must not loop on the map (a stuck detector like
  `stuck_frames`).

## Phase 5: put them together
- One agent: map network picks the level, level network plays it, repeat until the budget ends.
- Videos and viewer: show the map episode (route taken) next to the level clips.

## Constraints that carry over from the current work
- One change at a time, at least 3 seeds, short runs on Yoshi's Island 1 first, commit and merge only when asked.
- Old configs must keep replaying: every new behaviour sits behind an option that is off by default.
