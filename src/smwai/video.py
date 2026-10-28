"""Render the best run of each training run to an mp4 (game at 60 fps with a caption strip)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config, train
from .env import BUTTONS

SCALE = 2
BANNER = 44


def best_generation(run_dir):
    """Generation with the highest recorded fitness, as (generation, record without actions)."""
    best = None
    for f in sorted(Path(run_dir, "best").glob("gen_*.json")):
        rec = json.loads(f.read_text())
        if best is None or rec["fitness"] > best["fitness"]:
            best = rec
    return best


def _caption(img, lines, pressed):
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=15)
    d.rectangle([0, 0, img.width, BANNER], fill=(20, 22, 26))
    d.text((8, 4), lines[0], fill=(230, 232, 236), font=font)
    d.text((8, 24), lines[1], fill=(139, 147, 161), font=font)
    x = img.width - 8
    for name in reversed(BUTTONS):      # pressed buttons, right-aligned, lit when active
        on = name in pressed
        w = int(d.textlength(name, font=font)) + 8
        x -= w
        d.text((x, 24), name, fill=(90, 169, 255) if on else (60, 66, 76), font=font)


def render_run(run_dir, out_path=None, generation=None):
    """Replay a recorded generation and write an mp4. Returns the output path."""
    import imageio.v2 as imageio

    run_dir = Path(run_dir)
    rec = None
    if generation is None:
        rec = best_generation(run_dir)
        generation = rec["generation"]
    rec = json.loads((run_dir / "best" / f"gen_{generation:04d}.json").read_text())
    cfg = config.load(run_dir / "config.toml")
    train.use_config(cfg.env)
    env = train._get_env()
    env.capture = []
    try:
        env.reset()
        per_step = []                     # buttons pressed during each agent step
        total, info = 0.0, {}
        for mask in rec["actions"]:
            before = len(env.capture)
            _, reward, term, trunc, info = env.step(train._from_mask(mask))
            total += reward
            per_step.append((len(env.capture) - before, train._from_mask(mask), total, info["x"]))
            if term or trunc:
                break
        frames = env.capture
    finally:
        env.capture = None

    out_path = Path(out_path or run_dir / "videos" / f"best_gen_{generation:04d}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = "cleared" if info.get("cleared") else "died" if info.get("died") else "timed out"
    head = f"{run_dir.name}   generation {generation}"
    writer = imageio.get_writer(str(out_path), fps=60, codec="libx264", quality=7, macro_block_size=None,
                                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    try:
        i = 0
        for n, bits, reward, x in per_step:
            for _ in range(n):
                img = Image.fromarray(frames[i]).resize((256 * SCALE, 224 * SCALE), Image.NEAREST)
                canvas = Image.new("RGB", (img.width, img.height + BANNER))
                canvas.paste(img, (0, BANNER))
                pressed = {b for b, on in zip(BUTTONS, bits) if on}
                _caption(canvas, [head, f"fitness {rec['fitness']:.0f} ({result})   x {x}   reward {reward:.0f}"],
                         pressed)
                writer.append_data(np.asarray(canvas))
                i += 1
    finally:
        writer.close()
    return out_path
