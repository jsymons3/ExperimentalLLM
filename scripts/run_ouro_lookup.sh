#!/bin/bash
# Stage 1 on a pretrained model: Ouro-1.4B with a lookup table.
# Downloads the model on first use (about 2.9 GB), checks it, then runs the
# sweep. Re-running the same command resumes the saved plan and checkpoints.
#
#   bash scripts/run_ouro_lookup.sh --check-only     # a few minutes: does it learn at all?
#   bash scripts/run_ouro_lookup.sh --estimate-only  # time the runs and stop
#   bash scripts/run_ouro_lookup.sh                  # eight hours, four runs
#   bash scripts/run_ouro_lookup.sh --hours 10       # a longer budget
#
# A different budget or run list needs a fresh output directory (--out).
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python="$repo_root/.venv/bin/python"
run_dir="runs/ouro-lookup"
mkdir -p "$run_dir"

if ! "$python" -c "import tokenizers" 2>/dev/null; then
  echo "installing ouro_lookup/requirements.txt"
  "$python" -m pip install -q -r ouro_lookup/requirements.txt
fi

# Fetches the weights if they are missing, verifies them, and asks the model
# to complete a well-known sentence.
"$python" -m ouro_lookup.download

/usr/bin/caffeinate -is "$python" -u -m ouro_lookup.sweep \
  --preset full --out "$run_dir" --checkpoint-minutes 5 \
  "$@" 2>&1 | tee -i -a "$run_dir/sweep.log"
