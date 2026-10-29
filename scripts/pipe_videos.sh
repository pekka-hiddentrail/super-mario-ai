#!/usr/bin/env bash
# Render the best run of the runs that go to a pipe (and extract a still of the moment Mario enters it).
. ~/venv/bin/activate
cd "$(dirname "$0")/.."
for r in exp-rooms-pipes-yi2-rooms-s2 exp-rooms-pipes-yi4-rooms-s1; do
  python -W ignore -m smwai.cli video --run "$r" --force 2>&1 | tail -2
done
