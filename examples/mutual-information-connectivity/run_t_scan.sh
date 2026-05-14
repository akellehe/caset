#!/usr/bin/env bash
# Test 1 — T-scan (evolution-time scan) at fixed N=40.
# Varies the physical evolution time T with dt held constant.
# Tests whether peak D_S converges with longer evolution: does the
# spectral dimension reach a plateau, and if so at what value?
#
# The snapshot count K = T/dt + 1 follows from T; it's a derived
# discretization observable, not an independent variable.
set -u
cd "$(dirname "$0")/../.."
: "${OMP_NUM_THREADS:=10}"
: "${OPENBLAS_NUM_THREADS:=10}"
: "${MKL_NUM_THREADS:=10}"
: "${BLIS_NUM_THREADS:=10}"
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS BLIS_NUM_THREADS

OUT_DIR=/tmp/temporal-entangled/t_scan
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/scan.log"
: > "$LOG"

N=40
EPS=1e-6
DT=0.25

# Evolution-time targets at fixed dt=0.25.
for T in 1.0 1.5 2.0 2.5 3.0; do
  for MG in 0.125 0.25 0.5; do
    OUT="$OUT_DIR/N${N}_T${T}_mg_${MG}.json"
    echo "=== N=$N T=$T mg=$MG -> $OUT ===" | tee -a "$LOG"
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
