"""Tiny local web server for the replay viewer (stdlib only)."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .replay import compute_replay

STATIC = Path(__file__).parent / "static"
_emu_lock = threading.Lock()      # one emulator per process
_NAME = re.compile(r"^[\w.\-]+$")


def list_runs(runs_dir):
    runs = []
    for run in sorted(Path(runs_dir).glob("*/best"), reverse=True):
        gens = []
        for p in sorted(run.glob("gen_*.json")):
            rec = json.loads(p.read_text())
            gens.append({"gen": rec["generation"], "fitness": rec["fitness"],
                         "max_x": rec["max_x"], "cleared": rec["cleared"]})
        runs.append({"name": run.parent.name, "gens": gens})
    return runs


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

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/runs":
                    self._send(json.dumps(list_runs(runs_dir)))
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
