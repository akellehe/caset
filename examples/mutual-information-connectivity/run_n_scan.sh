#!/usr/bin/env bash
# Test 3 — N-only scan at fixed K=5.
# Tests the asymptotic behaviour of peak D_S as the boundary lattice
# grows, holding the snapshot count fixed.
# H_4D passes iff peak D_S plateaus near 4 from below as N grows.
set -u
cd "$(dirname "$0")/../.."
: "${OMP_NUM_THREADS:=10}"
: "${OPENBLAS_NUM_THREADS:=10}"
: "${MKL_NUM_THREADS:=10}"
: "${BLIS_NUM_THREADS:=10}"
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS BLIS_NUM_THREADS

OUT_DIR=/tmp/temporal-entangled/n_scan
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/scan.log"
: > "$LOG"

EPS=1e-6
T=1.0
DT=0.25
MG=0.5
MAX_BOND_DIM=128

for N in 50 60 80 100; do
  OUT="$OUT_DIR/N${N}_K5_mg_${MG}.json"
  echo "=== N=$N K=5 mg=$MG -> $OUT ===" | tee -a "$LOG"
  SECONDS=0
  python examples/quantum/temporally_connected_entangled_spacetime.py \
      --N "$N" --m-over-g "$MG" \
      --T "$T" --dt "$DT" \
      --max-bond-dim "$MAX_BOND_DIM" \
      --max-temporal-stride 0 \
      --epsilon-i "$EPS" \
      --out-json "$OUT" >> "$LOG" 2>&1
  rc=$?
  echo "    rc=$rc  elapsed=${SECONDS}s" | tee -a "$LOG"
done
echo "DONE" | tee -a "$LOG"
