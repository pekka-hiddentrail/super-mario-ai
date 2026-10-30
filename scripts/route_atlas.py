"""Draw a level's tile map (runs/_levels) with the best runs of several run folders on top: where they climb, where they die.
Usage: route_atlas.py <level> <out.png> <x-from> <x-to> <run-dir> [<run-dir> ...]   (x in pixels; scale 3 per pixel)"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

level, out, x0, x1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
runs = [Path(r) for r in sys.argv[5:]]
levels = Path("runs/_levels")
f = max(levels.glob(f"{level}.v*.json"), key=lambda p: p.name)
rows = json.loads(f.read_text())["rows"]
S = 3
COLORS = {"#": (150, 158, 172), "o": (92, 98, 110), "P": (63, 185, 80), "c": (242, 204, 96), "?": (232, 131, 58), "Y": (188, 140, 255)}
W = (x1 - x0) * S
img = Image.new("RGB", (W, len(rows) * 16 * S + 40), (20, 22, 26))
d = ImageDraw.Draw(img)
for r, line in enumerate(rows):
    for c, ch in enumerate(line):
        px = c * 16
        if ch != "." and x0 - 16 <= px < x1:
            d.rectangle([(px - x0) * S, r * 16 * S, (px - x0 + 16) * S - 1, (r + 1) * 16 * S - 1], fill=COLORS.get(ch, (110, 110, 110)))
for c in range(x0 // 100 * 100, x1, 100):
    d.line([((c - x0) * S, 0), ((c - x0) * S, 6)], fill=(200, 200, 200))
    d.text(((c - x0) * S + 3, 2), str(c), fill=(200, 200, 200))
palette = [(80, 255, 120), (255, 90, 90), (90, 170, 255), (255, 210, 60)]
for i, run in enumerate(runs):
    recs = [json.loads(p.read_text()) for p in (run / "best").glob("gen_*.json")]
    rec = max(recs, key=lambda r: r["fitness"])
    pts = [((x - x0) * S, (y + 16) * S) for x, y in rec["path"]]          # mid-body of Mario
    col = palette[i % 4]
    d.line(pts, fill=col, width=2)
    ex, ey = pts[-1]
    d.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], outline=col, width=2)
    d.text((6, len(rows) * 16 * S + 4 + 12 * i), f"{run.name}  gen {rec['generation']}  fitness {rec['fitness']:.0f}  ends x {rec['path'][-1][0]} y {rec['path'][-1][1]}", fill=col)
img.save(out)
print(out, img.size)
