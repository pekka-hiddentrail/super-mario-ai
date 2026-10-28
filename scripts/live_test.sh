#!/usr/bin/env bash
# End-to-end check of live tuning: train a tiny run, change the reward after generation 1.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
rm -rf ~/live_run
sed -e 's/^pop_size = .*/pop_size = 12/' -e 's/^name = .*/name = "live-test"/' configs/level1.toml > ~/live_test.toml
python -W ignore -m smwai.cli train --config ~/live_test.toml --generations 5 --workers 3 --run-dir ~/live_run > ~/live_run.log 2>&1 &
PID=$!
until grep -q "^gen    1 " ~/live_run.log 2>/dev/null; do sleep 2; done
echo '{"reward": {"progress": 3.0}, "neat": {"conn_add_prob": 0.9}}' > ~/live_run/overrides.json
echo "overrides written after gen 1"
wait $PID
cat ~/live_run.log
cat ~/live_run/events.jsonl
