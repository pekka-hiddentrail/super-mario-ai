#!/usr/bin/env bash
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
rm -rf ~/smoke_run
python -W ignore -m smwai.cli train --generations 3 --workers 10 --run-dir ~/smoke_run
echo "exit=$?"
