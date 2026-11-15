#!/bin/bash
# The two-day Ouro run: take over eight layers, then test flexibility.
# Downloads the model (2.9 GB) and the text (160 MB) on first use and checks
# both. Re-running the same command resumes the saved plan and checkpoints.
#
#   bash scripts/run_ouro_twoday.sh --check-only     # minutes: does each kind of run train?
#   bash scripts/run_ouro_twoday.sh --estimate-only  # time the runs, print the plan, stop
#   bash scripts/run_ouro_twoday.sh                  # 21 runs, at most about 44 hours
#   bash scripts/run_ouro_twoday.sh --hours 30       # a tighter budget
#
# A different budget or run list needs a fresh output directory (--out).
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python="$repo_root/.venv/bin/python"
run_dir="runs/ouro-twoday"
mkdir -p "$run_dir"

if ! "$python" -c "import tokenizers, pyarrow" 2>/dev/null; then
  echo "installing ouro_lookup/requirements.txt"
  "$python" -m pip install -q -r ouro_lookup/requirements.txt
fi

# Fetch and verify the model, then fetch, verify and tokenise the text.
"$python" -m ouro_lookup.download
"$python" -m ouro_lookup.text

/usr/bin/caffeinate -is "$python" -u -m ouro_lookup.twoday \
  --out "$run_dir" "$@" 2>&1 | tee -i -a "$run_dir/sweep.log"
