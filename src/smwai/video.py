"""Render the best run of each training run to an mp4: the game at 60 fps with a caption strip and, for NEAT
runs, the network next to it (input grids, hidden nodes, connections, buttons), as in MarI/O."""
import json
import pickle
from pathlib import Path

import neat
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config, train
from .env import BUTTONS, view as env_view
from .ram import grid_shape, window
from .viewer.replay import genome_graph

SCALE = 2
BANNER = 44
PANEL_W = 560
BG = (20, 22, 26)
MUTED, TEXT, ACCENT = (139, 147, 161), (230, 232, 236), (90, 169, 255)
POS, NEG = (63, 185, 80), (248, 81, 73)
GRID_COLORS = {"solid": (154, 163, 178), "enemy": (248, 81, 73), "item": (188, 140, 255), "coin": (242, 204, 96), "pipe": (63, 185, 80), "platform": (232, 131, 58), "reach": (88, 166, 255)}


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
    d.rectangle([0, 0, img.width, BANNER], fill=BG)
    d.text((8, 4), lines[0], fill=TEXT, font=font)
    d.text((8, 24), lines[1], fill=MUTED, font=font)
    if pressed is None:                 # NEAT videos show the buttons in the network panel instead
        return
    x = img.width - 8
    for name in reversed(BUTTONS):      # pressed buttons, right-aligned, lit when active
        on = name in pressed
        w = int(d.textlength(name, font=font)) + 8
        x -= w
        d.text((x, 24), name, fill=ACCENT if on else (60, 66, 76), font=font)


def _mix(fg, alpha):
    return tuple(int(b + (f - b) * alpha) for f, b in zip(fg, BG))


class NetworkPanel:
    """Draws the network of one genome for any step. Positions are computed once."""

    def __init__(self, graph, layout, view, height):
        left, _right, up, _down = window(view)
        self.rows, self.cols = grid_shape(view)
        self.mario = (up, left)                    # row, column of Mario's tile in each grid
        self.graph, self.layout, self.height = graph, layout, height
        grids = [item for item in layout if item["kind"] == "grid"]
        compact = any(":" in item["name"] for item in layout if item["kind"] == "scalar")
        self.cell = 14 if len(grids) == 1 else (4 if len(grids) >= 5 else 5) if compact else 9 if len(grids) <= 3 else 7
        # everything is laid out on a canvas that is 1/zoom larger than the panel and shrunk at the end, so that tall input windows
        # and many single values still fit
        names = [item["name"] for item in layout if item["kind"] == "scalar"]
        text_rows = (sum(1 for n in names if ":" not in n) + 3) // 4 + len({n.split(":")[0] for n in names if ":" in n}) if compact else 0
        needed = 34 + len(grids) * (self.rows * self.cell + (16 if len(grids) > 1 else 0)) + 12 * text_rows + 16
        self.zoom = min(1.0, height / needed) if compact else 1.0
        self.lw, self.lh = PANEL_W / self.zoom, height / self.zoom
        self.pos, self.groups = {}, []
        idx, y, sx, scalars = 0, 34, 18, []
        for item in layout:
            if item["kind"] == "grid":
                self.groups.append(("grid", item["name"], idx, 10, y))
                for i in range(self.rows * self.cols):
                    self.pos[graph["inputs"][idx + i]] = (10 + (i % self.cols) * self.cell + self.cell / 2,
                                                          y + (i // self.cols) * self.cell + self.cell / 2)
                idx += self.rows * self.cols
                y += self.rows * self.cell + (16 if len(grids) > 1 else 0)
            elif compact:
                scalars.append((idx, item["name"]))
                idx += 1
            else:
                self.pos[graph["inputs"][idx]] = (sx, height - 26)
                self.groups.append(("scalar", item["name"], idx, sx, height - 26))
                idx += 1
                sx += 66
        if compact:                      # many single values: text lines, base values first, then one line per jump option
            base = [(i, n) for i, n in scalars if ":" not in n]
            for k, (i, n) in enumerate(base):
                self._text_item(i, n.replace("_", " "), 10 + (k % 4) * 104, y + 4 + (k // 4) * 12)
            y += 4 + ((len(base) + 3) // 4) * 12 + 4
            rows_by_prefix = {}
            for i, n in scalars:
                if ":" in n:
                    rows_by_prefix.setdefault(n.split(":")[0], []).append((i, n.split(":")[1]))
            label_w = 62 if any(len(p) > 4 for p in rows_by_prefix) else 44
            for r, (prefix, items) in enumerate(rows_by_prefix.items()):
                self.groups.append(("label", prefix, 0, 10, y + r * 12))
                for k, (i, field) in enumerate(items):
                    self._text_item(i, field, label_w + k * 76, y + r * 12)
        cols = {}
        for h in graph["hidden"]:
            cols.setdefault(graph["layer"][str(h)], []).append(h)
        top = max(list(cols) + [1]) + 1
        for layer, nodes in cols.items():
            for i, h in enumerate(nodes):
                self.pos[h] = ((200 + (layer - 1) * (210 / max(1, top - 1))) / PANEL_W * self.lw,
                               50 + (i + 0.5) * (self.lh - 100) / len(nodes))
        for i, o in enumerate(graph["outputs"]):
            self.pos[o] = (480 / PANEL_W * self.lw, 56 + (i + 0.5) * (self.lh - 110) / len(graph["outputs"]))

    def _text_item(self, input_index, label, x, y):
        key = self.graph["inputs"][input_index]
        self.pos[key] = (x - 3, y + 5)
        self.groups.append(("text", label, input_index, x, y))

    def draw(self, obs, vals, pressed):
        g, cell = self.graph, self.cell
        img = Image.new("RGB", (int(self.lw), int(self.lh)), BG)
        d = ImageDraw.Draw(img)
        small, mid = ImageFont.load_default(size=11), ImageFont.load_default(size=13)
        value = dict(zip(g["inputs"], (int(v) for v in obs)))
        value.update(vals)
        d.text((10, 8), "Network: inputs around Mario, hidden nodes, buttons", fill=MUTED, font=mid)
        strong = []
        for e in g["edges"]:                       # quiet connections first, active ones on top
            a, b = self.pos.get(e["from"]), self.pos.get(e["to"])
            if a is None or b is None:
                continue
            v = value.get(e["from"], 0)
            active = v != 0 if e["from"] < 0 else v > 0.5
            if active:
                strong.append((a, b, e["w"]))
            else:
                d.line([a, b], fill=_mix(POS if e["w"] > 0 else NEG, 0.16), width=1)
        for a, b, w in strong:
            d.line([a, b], fill=_mix(POS if w > 0 else NEG, 0.95), width=2 if abs(w) > 1.5 else 1)
        for kind, name, start, x, y in self.groups:
            if kind == "grid":
                d.text((x, y - 13), name, fill=MUTED, font=small)
                for i in range(self.rows * self.cols):
                    cx, cy = self.pos[g["inputs"][start + i]]
                    v = value[g["inputs"][start + i]]
                    if name == "tiles":
                        fill = {1: GRID_COLORS["solid"], -1: GRID_COLORS["enemy"], 2: GRID_COLORS["item"]}.get(v)
                    else:
                        fill = GRID_COLORS.get(name) if v else None
                    box = [cx - cell / 2 + 1, cy - cell / 2 + 1, cx + cell / 2 - 1, cy + cell / 2 - 1]
                    d.rectangle(box, fill=fill, outline=(44, 49, 58))
                mx, my = self.pos[g["inputs"][start + self.mario[0] * self.cols + self.mario[1]]]
                d.rectangle([mx - cell / 2, my - cell / 2, mx + cell / 2, my + cell / 2], outline=ACCENT, width=2)
            elif kind == "text":
                v = value[g["inputs"][start]]
                d.text((x, y), f"{name} {v:+d}" if name in ("dx", "dy", "land") else f"{name} {v}",
                       fill=ACCENT if v else (90, 97, 110), font=small)
            elif kind == "label":
                d.text((x, y), name, fill=MUTED, font=small)
            else:
                v = value[g["inputs"][start]]
                d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=ACCENT if v else None, outline=MUTED, width=2)
                d.text((x - 8, y + 9), f"{name} {v}", fill=MUTED, font=small)
        for h in g["hidden"]:
            x, y = self.pos[h]
            v = max(0.0, min(1.0, float(value.get(h, 0.0))))
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=_mix(TEXT, 0.15 + 0.85 * v), outline=MUTED, width=2)
        for i, o in enumerate(g["outputs"]):
            x, y = self.pos[o]
            on = bool(pressed[i])
            v = max(0.0, min(1.0, float(value.get(o, 0.0))))
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=ACCENT if on else _mix(TEXT, 0.15 + 0.6 * v),
                      outline=ACCENT if on else MUTED, width=3 if on else 2)
            d.text((x + 13, y - 7), BUTTONS[i], fill=ACCENT if on else MUTED, font=mid)
        return img if self.zoom == 1.0 else img.resize((PANEL_W, self.height), Image.LANCZOS)


def _play_network(run_dir, generation, env):
    """Replay a recorded NEAT genome with its network, keeping every step's inputs and activations."""
    neat_cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                           neat.DefaultStagnation, str(run_dir / "neat.cfg"))
    with open(run_dir / "best" / f"gen_{generation:04d}.pkl", "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, neat_cfg)
    graph = genome_graph(genome, neat_cfg)
    watch = graph["hidden"] + graph["outputs"]
    env.death_map = json.loads((run_dir / "best" / f"gen_{generation:04d}.json").read_text()).get("death_map")
    obs, info = env.reset()
    steps, total = [], 0.0
    while True:
        out = net.activate(obs.ravel().astype(np.float64))
        action = env.decide(out)
        before = len(env.capture)
        steps.append({"obs": obs.ravel().copy(), "vals": {k: float(net.values.get(k, 0.0)) for k in watch},
                      "pressed": action, "x": info["x"], "before": before})
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        steps[-1].update(n=len(env.capture) - before, reward=total)
        if term or trunc:
            return steps, graph, info


def _play_masks(rec, env):
    """Replay recorded button presses (PPO runs have no network to draw)."""
    env.reset()
    steps, total, info = [], 0.0, {}
    for mask in rec["actions"]:
        before = len(env.capture)
        action = train._from_mask(mask)
        _, reward, term, trunc, info = env.step(action)
        total += reward
        steps.append({"pressed": action, "x": info["x"], "before": before, "n": len(env.capture) - before,
                      "reward": total})
        if term or trunc:
            break
    return steps, None, info


def render_run(run_dir, out_path=None, generation=None):
    """Replay a recorded generation and write an mp4. Returns the output path."""
    import imageio.v2 as imageio

    run_dir = Path(run_dir)
    if generation is None:
        generation = best_generation(run_dir)["generation"]
    rec = json.loads((run_dir / "best" / f"gen_{generation:04d}.json").read_text())
    cfg = config.load(run_dir / "config.toml")
    train.use_config(cfg.env)
    env = train._get_env()
    env.capture = []
    try:
        has_net = (run_dir / "neat.cfg").exists() and (run_dir / "best" / f"gen_{generation:04d}.pkl").exists()
        steps, graph, info = _play_network(run_dir, generation, env) if has_net else _play_masks(rec, env)
        frames = env.capture
        layout = [{"name": nm, "kind": k} for nm, k in env.layout]
    finally:
        env.capture = None

    out_path = Path(out_path or run_dir / "videos" / f"best_gen_{generation:04d}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = "cleared" if info.get("cleared") else "died" if info.get("died") else "timed out"
    head = f"{run_dir.name}   generation {generation}"
    height = 224 * SCALE + BANNER
    panel = NetworkPanel(graph, layout, env_view(cfg.env), height) if graph else None
    writer = imageio.get_writer(str(out_path), fps=60, codec="libx264", quality=7, macro_block_size=None,
                                ffmpeg_params=["-movflags", "+faststart"])
    try:
        for step in steps:
            pressed = {b for b, on in zip(BUTTONS, step["pressed"]) if on}
            side = panel.draw(step["obs"], step["vals"], step["pressed"]) if panel else None
            for k in range(step["n"]):
                game = Image.fromarray(frames[step["before"] + k]).resize((256 * SCALE, 224 * SCALE), Image.NEAREST)
                left = Image.new("RGB", (game.width, height))
                left.paste(game, (0, BANNER))
                _caption(left, [head, f"fitness {rec['fitness']:.0f} ({result})   x {step['x']}   "
                                      f"reward {step['reward']:.0f}"], None if side else pressed)
                canvas = Image.new("RGB", (game.width + (PANEL_W if side else 0), height))
                canvas.paste(left, (0, 0))
                if side:
                    canvas.paste(side, (game.width, 0))
                writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
    return out_path
