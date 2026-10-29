#!/usr/bin/env bash
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
python -W ignore -m smwai.cli replay ~/smoke_run 2
tail -4 ~/smoke_run/metrics.csv; ls ~/smoke_run | head
