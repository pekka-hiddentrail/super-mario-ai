"""Watch training as it happens: replays the newest best genome of the newest (or a chosen) run in real time, with the network
next to the game, as a motion-JPEG stream for a browser. It runs in its own process with its own emulator, so it never
slows the viewer, and it follows the training: when a new generation (or a new run of an experiment) appears, the next
episode plays that one.

    smwai live [--run NAME] [--port 8767]   ->   http://localhost:8767
"""
import io
import json
import pickle
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import neat
import numpy as np
from PIL import Image

from . import config, train
from .env import BUTTONS, input_layout, view as env_view
from .video import BANNER, PANEL_W, SCALE, NetworkPanel, _caption
from .viewer.replay import genome_graph

FRAME_SECONDS = 4 / 60          # one step = 4 emulator frames: real time


def latest_run(runs_dir, only=None):
    """(run dir, generation) of the newest generation whose genome is saved; None when there is nothing to show."""
    best = None
    for pkl in Path(runs_dir).glob("*/best/gen_*.pkl"):
        run = pkl.parent.parent
        if only and run.name != only or not (run / "neat.cfg").exists():
            continue
        t = pkl.stat().st_mtime
        if best is None or t > best[0]:
            best = (t, run, int(pkl.stem.split("_")[1]))
    return best[1:] if best else None


class LivePlayer(threading.Thread):
    daemon = True

    def __init__(self, runs_dir, only=None, speed=1.0):
        super().__init__()
        self.runs_dir, self.only, self.speed = Path(runs_dir), only, speed
        self.jpeg = None
        self.status = "waiting for a run with saved generations ..."
        self.cond = threading.Condition()

    def publish(self, img):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        with self.cond:
            self.jpeg = buf.getvalue()
            self.cond.notify_all()

    def run(self):
        while True:
            pick = latest_run(self.runs_dir, self.only)
            if pick is None:
                time.sleep(2)
                continue
            try:
                if (pick[0] / "population.pkl").exists():
                    self.play_population(pick[0])
                else:
                    self.play_best(*pick)
            except (OSError, ValueError, EOFError, KeyError, pickle.UnpicklingError) as exc:      # files still being written
                self.status = f"waiting: {exc!r}"
                time.sleep(2)

    def play_best(self, run_dir, generation):
        """Runs without a saved population: the best genome of the newest generation."""
        stem = run_dir / "best" / f"gen_{generation:04d}"
        rec = json.loads(stem.with_suffix(".json").read_text())
        with open(stem.with_suffix(".pkl"), "rb") as f:
            genome = pickle.load(f)
        head = f"LIVE  {run_dir.name}   generation {generation}"
        self.episode(run_dir, genome, head, f"best of the generation, fitness {rec['fitness']:.0f}", rec.get("death_map"))

    def play_population(self, run_dir):
        """Runs that save their population ([live] save_population): every genome of the generation in turn, species by species,
        failures included. A new generation is picked up when the current genome's episode is over."""
        pop_file = run_dir / "population.pkl"
        with open(pop_file, "rb") as f:
            data = pickle.load(f)
        stamp = pop_file.stat().st_mtime
        members = sorted(data["members"], key=lambda m: (m["species"], -m["fitness"]))
        sizes = {}
        for m in members:
            sizes[m["species"]] = sizes.get(m["species"], 0) + 1
        ranks = {m["gid"]: r + 1 for r, m in enumerate(sorted(members, key=lambda m: -m["fitness"]))}
        species_ids = sorted(sizes)
        for m in members:
            sid = m["species"]
            head = (f"LIVE {run_dir.name[:20]}  gen {data['generation']}  species {species_ids.index(sid) + 1}/{len(species_ids)}"
                    f" ({sizes[sid]} genomes)")
            note = f"genome {m['gid']}  fit {m['fitness']:.0f}  rank {ranks[m['gid']]}/{len(members)}"
            self.episode(run_dir, m["genome"], head, note, None)
            if pop_file.stat().st_mtime != stamp:
                return

    def episode(self, run_dir, genome, head, note, death_map):
        cfg = config.load(run_dir / "config.toml")
        neat_cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                               neat.DefaultStagnation, str(run_dir / "neat.cfg"))
        net = neat.nn.FeedForwardNetwork.create(genome, neat_cfg)
        graph = genome_graph(genome, neat_cfg)
        watch = graph["hidden"] + graph["outputs"]
        train.use_config(cfg.env)
        env = train._get_env()
        env.layout = input_layout(cfg.env)
        env.death_map = death_map
        layout = [{"name": nm, "kind": k} for nm, k in env.layout]
        height = 224 * SCALE + BANNER
        panel = NetworkPanel(graph, layout, env_view(cfg.env), height)
        self.status = f"{head.replace('LIVE  ', '')}  |  {note}"
        obs, info = env.reset()
        total, result = 0.0, ""
        while True:
            t0 = time.time()
            out = net.activate(obs.ravel().astype(np.float64))
            pressed = np.array(out) > 0.5
            vals = {k: float(net.values.get(k, 0.0)) for k in watch}
            side = panel.draw(obs.ravel(), vals, pressed)
            obs, reward, term, trunc, info = env.step(pressed)
            total += reward
            if term or trunc:
                result = "cleared" if info.get("cleared") else "died" if info.get("died") else "stuck" if trunc else ""
            game = Image.fromarray(env.frame).resize((256 * SCALE, 224 * SCALE), Image.NEAREST)
            left = Image.new("RGB", (game.width, height))
            left.paste(game, (0, BANNER))
            _caption(left, [head, f"{note}  x {info['x']}  reward {total:.0f}" + (f"   ({result})" if result else "")], None)
            canvas = Image.new("RGB", (game.width + PANEL_W, height))
            canvas.paste(left, (0, 0))
            canvas.paste(side, (game.width, 0))
            self.publish(canvas)
            if term or trunc:
                time.sleep(1.0)
                return
            time.sleep(max(0.0, FRAME_SECONDS / self.speed - (time.time() - t0)))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMW Live</title><style>
:root{--bg:#14161a;--text:#e6e8ec;--muted:#8b93a1}
@media (prefers-color-scheme: light){:root{--bg:#f4f5f7;--text:#1b1f24;--muted:#5b6573}}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif}
header{padding:8px 16px;color:var(--muted)} img{width:100%;max-width:1400px;display:block;margin:0 auto;background:#000}
</style></head><body><header><b style="color:var(--text)">Live</b> &middot; replays the newest best genome of the training run as it appears
&middot; <span id="s"></span></header><img src="/stream" alt="live run">
<script>setInterval(()=>fetch('/status').then(r=>r.text()).then(t=>document.getElementById('s').textContent=t),2000)</script>
</body></html>"""


def make_handler(player):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = None
                try:
                    while True:
                        with player.cond:
                            player.cond.wait_for(lambda: player.jpeg is not last and player.jpeg is not None, timeout=5)
                            jpeg = player.jpeg
                        if jpeg is None or jpeg is last:
                            continue
                        last = jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            elif self.path.startswith("/status"):
                body = player.status.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def serve(runs_dir="runs", port=8767, only=None, speed=1.0):
    player = LivePlayer(runs_dir, only, speed)
    player.start()
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(player))
    print(f"live: http://localhost:{port}  (following {only or 'the newest run'} in {runs_dir})", flush=True)
    server.serve_forever()
