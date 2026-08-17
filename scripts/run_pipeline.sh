#!/usr/bin/env bash
# Full reproduction: data -> features -> corpus -> three training stages ->
# out-of-sample evaluation -> ablations -> latency -> report.
#
# Usage:  scripts/run_pipeline.sh [--skip-data] [--skip-ablations]
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"
SKIP_DATA=0
SKIP_ABLATIONS=0

for arg in "$@"; do
  case "$arg" in
    --skip-data) SKIP_DATA=1 ;;
    --skip-ablations) SKIP_ABLATIONS=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

if [ "$SKIP_DATA" -eq 0 ]; then
  step "Market data"
  $PY -m gendesk.cli data build
  step "Features and regimes"
  $PY -m gendesk.cli data features
fi

step "Page corpus"
$PY -m gendesk.cli corpus build

step "Stage 1 - pretraining"
$PY -m gendesk.cli train pretrain

step "Stage 2 - weighted binary classification"
$PY -m gendesk.cli train wbc

step "Stage 3 - Dr. GRPO page-level RL"
$PY -m gendesk.cli train rl

step "Out-of-sample backtest"
$PY -m gendesk.cli eval backtest --window test

step "Validation-window backtest"
$PY -m gendesk.cli eval backtest --window valid

step "Serving latency"
$PY -m gendesk.cli eval latency

if [ "$SKIP_ABLATIONS" -eq 0 ]; then
  step "Ablation grid"
  $PY -m gendesk.cli eval ablations
fi

step "Report"
$PY -m gendesk.cli eval report

printf '\n\033[1;32mDone.\033[0m Artifacts are in artifacts/reports/.\n'
