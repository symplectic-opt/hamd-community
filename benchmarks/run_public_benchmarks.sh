#!/usr/bin/env bash
# =============================================================================
# run_public_benchmarks.sh
# HAMD Research Release — public benchmark suite
#
# Runs three experiments demonstrating the HAMD algorithm on cubic
# combinatorial optimisation problems:
#   1. Exact small-size validation (correctness proof, n=20/25/30)
#   2. Small HUBO cubic benchmark (n=50/75/100, HAMD vs SA vs Tabu)
#   3. Small cubic portfolio benchmark (n=200, HAMD vs SA vs Tabu)
#
# Usage:
#   bash benchmarks/run_public_benchmarks.sh [--quick]
#
# Options:
#   --quick    10 s budget, 1 seed  (~2 min total)
#   (default)  30 s budget, 3 seeds (~18 min total)
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

QUICK=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

if [[ $QUICK -eq 1 ]]; then
  BUDGET=10; SEEDS=1
  echo "=== QUICK MODE: budget=${BUDGET}s, seeds=${SEEDS} ==="
else
  BUDGET=30; SEEDS=3
fi

OUTDIR="results_public"
mkdir -p "$OUTDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUTDIR/run_${STAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "HAMD Research Release — Public Benchmark Suite"
log "budget=${BUDGET}s  seeds=${SEEDS}  output=${OUTDIR}"
log "=========================================================="

# ── Experiment 1: Exact small-size validation ─────────────────────────────
log ""
log "── Experiment 1/3: Exact Small-Size Validation (n=20,25,30) ──"
python3 benchmarks/run_exact_small.py \
    --sizes 20 25 30 \
    --budget-sec "$BUDGET" \
    --seeds "$SEEDS" \
  2>&1 | tee "$OUTDIR/exact_small.log"

# ── Experiment 2: HUBO cubic benchmark ────────────────────────────────────
log ""
log "── Experiment 2/3: HUBO Cubic Benchmark (n=50,75,100) ──"
python3 benchmarks/run_hubo_small.py \
    --budget-sec "$BUDGET" \
    --seeds "$SEEDS" \
  2>&1 | tee "$OUTDIR/hubo_small.log"

# ── Experiment 3: Cubic portfolio benchmark ───────────────────────────────
log ""
log "── Experiment 3/3: Cubic Portfolio Benchmark (n=200, K=40) ──"
python3 benchmarks/run_portfolio_small.py \
    --budget-sec "$BUDGET" \
    --seeds "$SEEDS" \
  2>&1 | tee "$OUTDIR/portfolio_small.log"

log ""
log "=========================================================="
log "All experiments complete.  Logs: $OUTDIR/"
