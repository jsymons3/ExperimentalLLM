#!/bin/bash
# The reading test: four fact runs, two seeds each, one after another.
# Re-running this command resumes the saved plan and unfinished checkpoints.
#
#   bash scripts/run_reading_test.sh                  # eight runs, about six hours
#   bash scripts/run_reading_test.sh --estimate-only  # time the runs and stop
#   bash scripts/run_reading_test.sh --hours 6        # a tighter budget
#
# A different budget, run list or seed list needs a fresh output directory (--out).
# No download is needed: the facts are in the repository.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python="$repo_root/.venv/bin/python"
run_dir="runs/reading-test"
mkdir -p "$run_dir"

/usr/bin/caffeinate -is "$python" -u -m phrase_scratchpad.sweep \
  --reading --seeds 0 1 --preset full --out "$run_dir" \
  --checkpoint-minutes 5 --log-every 500 \
  "$@" 2>&1 | tee -i -a "$run_dir/sweep.log"
