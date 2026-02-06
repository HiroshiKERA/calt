#!/usr/bin/env bash
# Run full evaluation for all tasks/patterns and write success_rate table.
# Uses conda env calt-env. Execute from issac2026_experiments/:  bash evaluate/run_all_eval.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate calt-env
fi

python evaluate/run_all_eval.py "$@"
