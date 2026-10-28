#!/usr/bin/env bash
# Show which training processes are alive, grouped by config.
ps -eo args= | grep -E "smwai (train|train-ppo)|smwai.cli train" | grep -v grep \
  | sed -E 's/.*--config ([^ ]+).*/\1/' | sort | uniq -c
echo "load: $(cut -d' ' -f1-3 /proc/loadavg)  cores: $(nproc)"
