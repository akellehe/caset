#!/usr/bin/env bash
# Test 1 — K-only scan at fixed N=40.
# Tests whether the N=60 K=9 overshoot was driven by K rather than N.
# H_4D passes iff peak D_S plateaus near 4 as K grows.
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10
export BLIS_NUM_THREADS=10

OUT_DIR=/tmp/temporal-entangled/k_scan
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/scan.log"
: > "$LOG"

N=40
EPS=1e-6
DT=0.25

# K = T/dt + 1.  Targets K ∈ {5,7,9,11,13} → T ∈ {1.0, 1.5, 2.0, 2.5, 3.0}.
for T in 1.0 1.5 2.0 2.5 3.0; do
  for MG in 0.125 0.25 0.5; do
    K=$(python -c "import math; print(int(math.floor(${T}/${DT})) + 1)")
    OUT="$OUT_DIR/N${N}_K${K}_mg_${MG}.json"
    echo "=== N=$N K=$K (T=$T) mg=$MG -> $OUT ===" | tee -a "$LOG"
    SECONDS=0
    python examples/quantum/temporally_connected_entangled_spacetime.py \
        --N "$N" --m-over-g "$MG" \
        --T "$T" --dt "$DT" \
        --max-temporal-stride 0 \
        --epsilon-i "$EPS" \
        --out-json "$OUT" >> "$LOG" 2>&1
    rc=$?
    echo "    rc=$rc  elapsed=${SECONDS}s" | tee -a "$LOG"
  done
done
echo "DONE" | tee -a "$LOG"
