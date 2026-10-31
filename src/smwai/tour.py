"""A tour through a run's species: for each of the first generations, one random genome of each of the first species, played one
after another into a single mp4 (game + network, caption with generation, species and genome). Needs a run trained with
`[live] save_population = true` and `save_history = true`.

    smwai tour --run NAME [--generations 5] [--species 5] [--seed 1] [--steps 450]
"""
import pickle
import random
from pathlib import Path

import numpy as np

from .live import episode_frames, generation_death_map


def pick_tour(run_dir, generations=5, species=5, seed=1):
    """[(generation, species position, species count, species id, member)] in playing order."""
    rng = random.Random(seed)
    files = sorted(Path(run_dir).glob("population_[0-9]*.pkl"))[:generations]
    tour = []
    for f in files:
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        by_species = {}
        for m in data["members"]:
            by_species.setdefault(m["species"], []).append(m)
        ids = sorted(by_species)[:species]
        for k, sid in enumerate(ids):
            tour.append((data["generation"], k + 1, len(ids), sid, rng.choice(by_species[sid])))
    return tour


def make_tour(run_dir, generations=5, species=5, seed=1, max_steps=450, out=None):
    import imageio.v2 as imageio

    run_dir = Path(run_dir)
    tour = pick_tour(run_dir, generations, species, seed)
    if not tour:
        raise SystemExit(f"{run_dir} has no population_*.pkl files: train it with [live] save_population and save_history")
    out = Path(out or run_dir / "videos" / f"tour_gen_{tour[-1][0]:04d}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=60, codec="libx264", quality=7, macro_block_size=None,
                                ffmpeg_params=["-movflags", "+faststart"])
    try:
        for n, (gen, k, count, sid, m) in enumerate(tour, 1):
            head = f"TOUR {n}/{len(tour)}  gen {gen}  species {k}/{count} (id {sid})"
            note = f"genome {m['gid']}  fit {m['fitness']:.0f}"
            print(head, note, flush=True)
            last = None
            for frames in episode_frames(run_dir, m["genome"], head, note, generation_death_map(run_dir, gen), capture=True, max_steps=max_steps):
                for canvas in frames:
                    writer.append_data(np.asarray(canvas))
                    last = canvas
            for _ in range(45):                       # a short hold on the last frame before the next genome
                writer.append_data(np.asarray(last))
    finally:
        writer.close()
    return out
