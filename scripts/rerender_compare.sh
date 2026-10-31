#!/usr/bin/env bash
# Render the compare_yi1 videos again with the current network display (bias, input sums, thicker active links), then stitch each run, detached:
# scripts/rerender_compare.sh   (log /tmp/rerender_compare.log). Waits for compare_yi1 to finish first.
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
setsid nohup bash -c '
  until grep -q "all done" /tmp/compare_yi1.log 2>/dev/null; do sleep 10; done
  for run in $(ls runs | grep -E "^[0-9]{8}-[0-9]{4}-compare-yi1-"); do
    FORCE=1 scripts/video_all_gens.sh $run
    python -W ignore -m smwai.cli stitch --run $run
  done
  echo "all done"
' > /tmp/rerender_compare.log 2>&1 < /dev/null &
disown
echo "started; log /tmp/rerender_compare.log"
