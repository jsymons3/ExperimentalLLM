#!/bin/bash
# The three-stage multi-token scratchpad runs, one after another.
# Re-running this command resumes the saved plan and unfinished checkpoints.
#
#   bash scripts/run_phrase_scratchpad.sh                  # eight hours, five runs
#   bash scripts/run_phrase_scratchpad.sh --hours 10       # a longer budget
#   bash scripts/run_phrase_scratchpad.sh --estimate-only  # time the runs and stop
#
# A different budget or run list needs a fresh output directory (--out).
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python="$repo_root/.venv/bin/python"
run_dir="runs/phrase-scratchpad"
mkdir -p "$run_dir"

# The family stories are downloaded once (about 20 MB) and checked.
"$python" -m phrase_scratchpad.clutrr

/usr/bin/caffeinate -is "$python" -u -m phrase_scratchpad.sweep \
  --preset full --out "$run_dir" --checkpoint-minutes 5 --log-every 500 \
  "$@" 2>&1 | tee -i -a "$run_dir/sweep.log"
