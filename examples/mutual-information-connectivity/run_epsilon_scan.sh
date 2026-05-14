#!/usr/bin/env bash
# Test 2 — ε-scan at fixed (N=60, K=9).
# Multi-ε per TDVP run: TDVP runs once per m/g; graph build, spectral
# dimension, diameter, and histograms are computed at five thresholds.
# H_4D passes iff peak D_S plateaus near 4 over an intermediate ε
# window.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10
export BLIS_NUM_THREADS=10

OUT_DIR=/tmp/temporal-entangled/epsilon_scan
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/scan.log"
: > "$LOG"

N=60
T=2.0
DT=0.25

# Five ε values spanning into the tail of the MI distribution.
EPSILONS="1e-8 1e-6 1e-4 1e-3 1e-2"

for MG in 0.125 0.25 0.5; do
  BASE_OUT="$OUT_DIR/N${N}_K9_mg_${MG}.json"
  echo "=== N=$N K=9 mg=$MG (multi-ε) -> $BASE_OUT (split by ε) ===" \
       | tee -a "$LOG"
  SECONDS=0
  python examples/quantum/temporally_connected_entangled_spacetime.py \
      --N "$N" --m-over-g "$MG" \
      --T "$T" --dt "$DT" \
      --max-temporal-stride 0 \
      --epsilon-i $EPSILONS \
      --out-json "$BASE_OUT" >> "$LOG" 2>&1
  rc=$?
  echo "    rc=$rc  elapsed=${SECONDS}s" | tee -a "$LOG"
done
echo "DONE" | tee -a "$LOG"
