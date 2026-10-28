#!/usr/bin/env bash
# Restart the viewer server and check that a PPO run can be replayed and mapped.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
pkill -f "smwai.cli view" 2>/dev/null; sleep 1
rm -rf runs/ppo-smoke && cp -r ~/ppo_smoke runs/ppo-smoke
nohup python -W ignore -m smwai.cli view > /dev/null 2>&1 &
sleep 4
curl -s "http://localhost:8765/api/replay?run=ppo-smoke&gen=3" | python -c "import json,sys; d=json.load(sys.stdin); print('replay:', d.get('error') or (len(d['steps']), 'steps', 'graph', d['graph'], 'fitness', round(d['fitness'],1), 'max_x', d['max_x']))"
curl -s "http://localhost:8765/api/path?run=ppo-smoke&gen=3" | python -c "import json,sys; d=json.load(sys.stdin); print('path:', d.get('error') or (len(d['path']), 'points'))"
