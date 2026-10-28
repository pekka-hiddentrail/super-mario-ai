"""Re-run a recorded best genome and capture everything the browser viewer needs."""
import base64
import io
import json
import pickle
from pathlib import Path

import neat
import numpy as np
from PIL import Image

from .. import config, train
from ..env import BUTTONS


def genome_graph(genome, neat_cfg):
    """Nodes (with layer) and enabled connections that matter for the outputs."""
    gcfg = neat_cfg.genome_config
    inputs, outputs = list(gcfg.input_keys), list(gcfg.output_keys)
    conns = [(k, c.weight) for k, c in genome.connections.items() if c.enabled]
    required = neat.graphs.required_for_output(inputs, outputs, [k for k, _ in conns])
    hidden = sorted(n for n in required if n not in outputs and n not in inputs)
    layer = {n: 0 for n in inputs}
    for _ in range(len(hidden) + 1):          # longest path from inputs, relaxed to a fixpoint
        for (a, b), _w in conns:
            if b in hidden and a in layer:
                layer[b] = max(layer.get(b, 0), layer[a] + 1)
    top = max([layer.get(h, 1) for h in hidden], default=0) + 1
    for o in outputs:
        layer[o] = top
    edges = [{"from": a, "to": b, "w": round(w, 3)} for (a, b), w in conns
             if (b in required) and (a in inputs or a in required)]
    return {"inputs": inputs, "outputs": outputs, "hidden": hidden,
            "layer": {str(k): layer.get(k, 1) for k in hidden + outputs}, "edges": edges}


def _jpeg(frame):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def compute_replay(run_dir, generation, use_cache=True):
    run_dir = Path(run_dir)
    cache = run_dir / "replay_cache" / f"gen_{generation:04d}.json"
    if use_cache and cache.exists():
        return cache.read_text()
    cfg = config.load(run_dir / "config.toml")
    train.use_config(cfg.env)
    neat_cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                           neat.DefaultStagnation, str(run_dir / "neat.cfg"))
    with open(run_dir / "best" / f"gen_{generation:04d}.pkl", "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, neat_cfg)
    graph = genome_graph(genome, neat_cfg)
    watch = graph["hidden"] + graph["outputs"]
    env = train._get_env()
    obs, info = env.reset()
    steps, total = [], 0.0
    while True:
        out = net.activate(obs.ravel().astype(np.float64))
        action = np.array(out) > 0.5
        steps.append({
            "frame": _jpeg(env.frame),
            "grid": "".join(chr(ord("a") + int(v) + 1) for v in obs.ravel()),  # -1..2 -> a..d
            "act": [round(float(net.values.get(n, 0.0)), 3) for n in watch],
            "pressed": [int(a) for a in action],
            "x": info["x"], "y": info["y"], "reward": round(total, 1)})
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        if term or trunc:
            break
    steps.append({"frame": _jpeg(env.frame), "grid": steps[-1]["grid"], "act": steps[-1]["act"],
                  "pressed": [0] * len(BUTTONS), "x": info["x"], "y": info["y"],
                  "reward": round(total, 1)})
    result = json.dumps({
        "generation": generation, "fitness": total, "max_x": info["max_x"],
        "cleared": bool(info.get("cleared")), "died": bool(info.get("died")),
        "buttons": BUTTONS, "radius": cfg.env.radius, "watch": watch, "graph": graph,
        "steps": steps})
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(result)
    return result
