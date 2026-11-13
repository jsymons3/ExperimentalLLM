#!/bin/bash
# Re-running this command resumes the saved plan and unfinished checkpoints.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
run_dir="runs/overwrite-2026-11-13"
mkdir -p "$run_dir"
/usr/bin/caffeinate -is "$repo_root/.venv/bin/python" -u -m recurrent_lookup.sweep \
  --runs overwrite --preset full --hours 7 --max-steps 128200 \
  --pilot-steps 500 --out "$run_dir" --checkpoint-minutes 5 --log-every 500 \
  "$@" 2>&1 | tee -i -a "$run_dir/overnight.log"
