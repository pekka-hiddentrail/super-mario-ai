"""Tiny local web server for the replay viewer (stdlib only)."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .mapdata import level_map, run_path
from .replay import compute_replay

STATIC = Path(__file__).parent / "static"
_emu_lock = threading.Lock()      # one emulator per process
_NAME = re.compile(r"^(?!\.)[\w.\-]+$")      # no leading dot: blocks '..' path tricks


def _gen_summaries(best_dir):
    """[{gen, fitness, max_x, cleared}] for a run, read from a small index.json that is updated
    incrementally (the generation files are large; opening all of them made the page slow)."""
    index_file = best_dir / "index.json"
    try:
        index = json.loads(index_file.read_text())
    except (OSError, ValueError):
        index = {}
    present = {f.name: f for f in best_dir.glob("gen_*.json")}
    changed = False
    for name in sorted(present):
        if name in index:                        # known generation: trust it (a stat per file is slow on /mnt/c)
            continue
        try:
            rec = json.loads(present[name].read_text())
        except ValueError:                       # being written by a running trainer right now
            continue
        index[name] = {"gen": rec["generation"], "fitness": rec["fitness"], "max_x": rec["max_x"],
                       "cleared": rec["cleared"]}
        changed = True
    for name in [k for k in index if k not in present]:
        del index[name]
        changed = True
    if changed:
        index_file.write_text(json.dumps(index))
    return [{k: index[n][k] for k in ("gen", "fitness", "max_x", "cleared")} for n in sorted(index)]


def _run_time(run_dir):
    """When the run was last active: newest runs are listed first."""
    for name in ("metrics.csv", "best"):
        f = run_dir / name
        if f.exists():
            return f.stat().st_mtime
    return 0.0


def list_runs(runs_dir):
    """Runs with recorded generations, newest first."""
    runs = []
    for best in Path(runs_dir).glob("*/best"):
        runs.append({"name": best.parent.name, "gens": _gen_summaries(best), "_t": _run_time(best.parent)})
    runs.sort(key=lambda r: r["_t"], reverse=True)
    for r in runs:
        del r["_t"]
    return runs


def list_videos(runs_dir):
    """Runs that have rendered videos, with the metrics needed to describe them (newest run first)."""
    out = []
    vdirs = sorted(Path(runs_dir).glob("*/videos"), key=lambda v: _run_time(v.parent), reverse=True)
    for vdir in vdirs:
        run = vdir.parent
        for mp4 in sorted(vdir.glob("*.mp4")):
            tail = mp4.stem.split("_")[-1]
            gen = int(tail) if tail.isdigit() else None            # all_generations.mp4: the stitched video of the whole run
            rec_file = run / "best" / f"gen_{gen:04d}.json" if gen is not None else None
            rec = json.loads(rec_file.read_text()) if rec_file and rec_file.exists() else {}
            out.append({"run": run.name, "file": mp4.name, "generation": gen,
                        "fitness": rec.get("fitness"), "max_x": rec.get("max_x"),
                        "cleared": rec.get("cleared"), "size_mb": round(mp4.stat().st_size / 1e6, 1)})
    return out


def make_handler(runs_dir):
    runs_dir = Path(runs_dir)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype="application/json", status=200):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_video(self, rel):
            """Serve runs/<run>/videos/<file>.mp4 with Range support so browsers can seek."""
            run, _, name = rel.partition("/")
            path = runs_dir / run / "videos" / name
            if not (_NAME.match(run) and _NAME.match(name) and path.is_file()):
                return self._send('{"error": "not found"}', status=404)
            size = path.stat().st_size
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else end
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
            self.send_response(206 if m else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if m:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(end - start + 1))

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/runs":
                    self._send(json.dumps(list_runs(runs_dir)))
                elif url.path in ("/videos", "/videos.html"):
                    self._send((STATIC / "videos.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/videos":
                    self._send(json.dumps(list_videos(runs_dir)))
                elif url.path.startswith("/video/"):
                    return self._send_video(url.path[len("/video/"):])
                elif url.path in ("/map", "/map.html"):
                    self._send((STATIC / "map.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path in ("/api/level", "/api/path"):
                    name = q["run"][0]
                    if not _NAME.match(name) or not (runs_dir / name).is_dir():
                        return self._send('{"error": "unknown run"}', status=404)
                    with _emu_lock:
                        if url.path == "/api/level":
                            self._send(level_map(runs_dir / name))
                        else:
                            self._send(json.dumps(run_path(runs_dir / name, int(q["gen"][0]),
                                                           events="events" in q)))
                elif url.path == "/api/replay":
                    name, gen = q["run"][0], int(q["gen"][0])
                    if not _NAME.match(name) or not (runs_dir / name).is_dir():
                        return self._send('{"error": "unknown run"}', status=404)
                    with _emu_lock:
                        self._send(compute_replay(runs_dir / name, gen))
                else:
                    self._send('{"error": "not found"}', status=404)
            except Exception as exc:  # report to the page instead of dropping the connection
                self._send(json.dumps({"error": repr(exc)}), status=500)

        def log_message(self, *args):
            pass

    return Handler


def serve(runs_dir="runs", port=8765):
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(runs_dir))
    print(f"viewer: http://localhost:{port}  (runs in {runs_dir})", flush=True)
    server.serve_forever()
