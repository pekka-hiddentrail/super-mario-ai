"""How well does bridge_ext predict the turn-block bridge? Compares the prediction made at each step with what happened
`ahead` steps later (4 frames per step). Usage: validate_platforms.py <run-dir> [<steps ahead>]"""
import json
import sys
from pathlib import Path

from smwai import config, train
from smwai import ram as R

run, ahead = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 4
rec = max((json.loads(f.read_text()) for f in sorted((run / "best").glob("gen_*.json"))), key=lambda r: r["fitness"])
train.use_config(config.load(run / "config.toml").env)
env = train._get_env()
env.reset()
hist = []
for mask in rec["actions"]:
    env.step(train._from_mask(mask))
    hist.append({s.slot: (s.number, s.x, s.ext, R.bridge_ext(s, 4 * ahead)) for s in env.state.sprites if s.number == R.BRIDGE_SPRITE})
errs = []
for i in range(len(hist) - ahead):
    for slot, (num, x, ext, pred) in hist[i].items():
        later = hist[i + ahead].get(slot)
        if later and later[1] == x:
            errs.append(abs(later[2] - pred))
print(f"{run.name}: {len(errs)} predictions {ahead * 4} frames ahead, mean error {sum(errs) / max(1, len(errs)):.1f} px, "
      f"within 4 px: {100 * sum(e <= 4 for e in errs) / max(1, len(errs)):.0f}%")
