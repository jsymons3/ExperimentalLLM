#!/bin/bash
# Two seed-0 controls at the same 64,000 updates as the completed hard-slot run.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
run_dir="runs/followup-2026-11-13"
mkdir -p "$run_dir"
/usr/bin/caffeinate -is "$repo_root/.venv/bin/python" -u -m recurrent_lookup.sweep \
  --runs followups --preset full --steps 64000 --out "$run_dir" --pilot-steps 200 \
  --checkpoint-minutes 5 --log-every 500 \
  "$@" 2>&1 | tee -i -a "$run_dir/followups.log"

# Subsequent invocations reuse timings and resume unfinished checkpoints.
# Report the controls alongside the existing supervised scratchpad experiments.
if [[ -f "$run_dir/dense_hard_s0/final.json" ]]; then
  "$repo_root/.venv/bin/python" -m recurrent_lookup.report \
    "$run_dir" runs/overwrite-2026-11-13 > "$run_dir/comparison.log"
fi
