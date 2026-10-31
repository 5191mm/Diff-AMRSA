#!/usr/bin/env bash
# Diff-AMRSA stage runner. Required flags are documented in docs/EXPERIMENT_GUIDE.md.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
STEP="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$STEP" in
  export-splits)  python scripts/diff_amrsa/export_splits.py "$@" ;;
  train-encoder)  python scripts/diff_amrsa/train_graph_encoder.py "$@" ;;
  train-diffusion) python scripts/diff_amrsa/train_conditioned_diffusion.py "$@" ;;
  export-conformers) python scripts/diff_amrsa/export_conformers.py "$@" ;;
  train-refiner)  python scripts/diff_amrsa/train_flow.py "$@" ;;
  generate)       python scripts/diff_amrsa/sample_pipeline.py "$@" ;;
  train-proxy)    python scripts/diff_amrsa/train_activity_proxy.py "$@" ;;
  rank)           python scripts/diff_amrsa/screen_and_rank.py "$@" ;;
  test)           python -m pytest -q tests/diff_amrsa "$@" ;;
  help|-h|--help)
    cat <<EOF
Diff-AMRSA full-framework entrypoints
Usage: bash scripts/run_pipeline.sh <stage> [stage arguments]
Stages: export-splits train-encoder train-diffusion export-conformers
        train-refiner generate train-proxy rank test
For exact stage flags see: docs/EXPERIMENT_GUIDE.md
Older adaptation-only baseline: scripts/run_legacy_pipeline.sh
EOF
    ;;
  *) echo "Unknown stage: $STEP" >&2; exit 2 ;;
esac
