#!/usr/bin/env bash
# Time the viewer's endpoints (where does the path map spend its time?).
run="${1:-yi1-adaptive-20261028-215704}"
t() { curl -s -o /dev/null -w "%{time_total}s  $1\n" "http://localhost:8765$1"; }
t /api/runs
t "/api/level?run=$run"
t "/api/path?run=$run&gen=5"
for g in 6 7 8; do t "/api/path?run=$run&gen=$g"; done
t "/api/path?run=$run&gen=23&events=1"
du -sh /mnt/c/Users/pmarj/super-mario-ai/runs/$run/best | tail -1
ls -S /mnt/c/Users/pmarj/super-mario-ai/runs/$run/best/*.json | head -1 | xargs ls -la
