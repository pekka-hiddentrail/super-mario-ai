"""Replay a chain of checkpoint runs from the real start of the level, input by input, and see whether it clears the level.

With checkpoints on, a generation starts from a save a run of an earlier generation made (checkpoint_NN_xCX.state), and that save's folder keeps next to it
the emulator input that led there (checkpoint_NN_xCX.json: where that run started from, and its input frame by frame). A generation's best run records its
own input and its start. Walking the starts back to the level's own start and putting the inputs in order gives one continuous input sequence for the
real level; feeding it to a plain emulator shows what the chain really does, with no saved state after the first frame.
"""
import json
from pathlib import Path

import numpy as np

from . import config
from . import ram as R

GAME = "SuperMarioWorld-Snes-v0"


def segments_of(run_dir, generation):
    """The chain as a list of emulator-input lists in play order: the run to the first save, the run from it to the second, ..., the best run of `generation`."""
    run = Path(run_dir)
    rec = json.loads((run / "best" / f"gen_{generation:04d}.json").read_text())
    if not rec.get("trace"):
        raise SystemExit("this run recorded no input trace (it was trained without checkpoints)")
    segments, start = [rec["trace"]], rec.get("start")
    while start:
        side = json.loads(Path(start).with_suffix(".json").read_text())
        segments.append(side["trace"])
        start = side["start"]
    return list(reversed(segments))


def stretches_of(run_dir, generation):
    """The chain as a list of (emulator input, genome file) in play order; the genome file is None where an older run did not keep it."""
    run = Path(run_dir)
    rec = json.loads((run / "best" / f"gen_{generation:04d}.json").read_text())
    stretches, start = [(rec["trace"], run / "best" / f"gen_{generation:04d}.pkl")], rec.get("start")
    while start:
        side = json.loads(Path(start).with_suffix(".json").read_text())
        pkl = Path(start).with_suffix(".pkl")
        stretches.append((side["trace"], pkl if pkl.exists() else None))
        start = side["start"]
    return list(reversed(stretches))


def assemble(run_dir, generation):
    """(frames, checkpoints): the emulator input of the whole chain ending with the best run of `generation`, and how many saves it passes through."""
    segments = segments_of(run_dir, generation)
    return [mask for seg in segments for mask in seg], len(segments) - 1


def verify(run_dir, generation, level=None):
    """Feed the chain to a fresh emulator on the level's own start. Returns a dict: cleared, frames used, final x, checkpoints in the chain."""
    import stable_retro as retro

    frames, checkpoints = assemble(run_dir, generation)
    level = level or config.load(Path(run_dir) / "config.toml").env.levels[0]
    env = retro.make(GAME, state=level, use_restricted_actions=retro.Actions.ALL, render_mode=None)
    try:
        env.reset()
        used, cleared = 0, False
        for mask in frames:
            env.step(np.array([(mask >> b) & 1 for b in range(12)], dtype=np.int8))
            used += 1
            ram = R.wram(env.unwrapped.get_ram())
            if R.u8(ram, R.END_LEVEL_TIMER) != 0:
                cleared = True
                break
        st = R.read_state(R.wram(env.unwrapped.get_ram()))
        return {"level": level, "checkpoints": checkpoints, "frames_available": len(frames), "frames_used": used, "cleared": cleared, "x": st.x, "y": st.y}
    finally:
        env.close()


def render(run_dir, generation=None, out_path=None):
    """One video of the whole chain on the real level: every checkpoint run joined, the banner telling which stretch is playing. Returns the output path."""
    import stable_retro as retro
    from PIL import Image

    from . import video

    run_dir = Path(run_dir)
    if generation is None:
        generation = video.best_generation(run_dir)["generation"]
    segments = segments_of(run_dir, generation)
    level = config.load(run_dir / "config.toml").env.levels[0]
    env = retro.make(GAME, state=level, use_restricted_actions=retro.Actions.ALL, render_mode=None)
    names = list(env.buttons)
    out_path = Path(out_path or run_dir / "videos" / f"chain_gen_{generation:04d}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    head = f"{run_dir.name}   whole run, {len(segments) - 1} checkpoints"
    height = 224 * video.SCALE + video.BANNER
    writer = video._video_writer(out_path)
    try:
        obs, _ = env.reset()
        done = 0
        total = sum(len(s) for s in segments)
        for n, seg in enumerate(segments, 1):
            label = f"stretch {n}/{len(segments)}"
            for mask in seg:
                obs = env.step(np.array([(mask >> b) & 1 for b in range(12)], dtype=np.int8))[0]
                done += 1
                x = R.read_state(R.wram(env.unwrapped.get_ram())).x
                pressed = {names[b] for b in range(12) if (mask >> b) & 1}
                game = Image.fromarray(obs).resize((256 * video.SCALE, 224 * video.SCALE), Image.NEAREST)
                canvas = Image.new("RGB", (game.width, height))
                canvas.paste(game, (0, video.BANNER))
                video._caption(canvas, [head, f"{label}  x {x}  {done}/{total}"], pressed)
                writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        env.close()
    return out_path
