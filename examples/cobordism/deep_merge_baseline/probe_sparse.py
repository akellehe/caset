"""Probe v5: (a) sparse null-space for ker L1 (fast, replaces the 16s dense
eigendecomp), verified to span the dense harmonics; (b) the KEY physics check —
does the SELF-CONSISTENT matter energy (recomputed register) give the conformal
free energy G=ReS+kappa E an INTERIOR minimum? i.e. does the mass regulate the
runaway to a convergent geometry (vs fixed-h, which ran to the bound)?"""
import os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.argv = ["probe"]
exec(open("/tmp/probe_deep.py").read().split("for level in")[0])

st, nreg, holes, hole_vs, cells = build_merge(2)
st.materializeFacets()
R = 2 * nreg
es = cob.EigenstateSynthesis(st, 1)
cells_k1 = [tuple(int(v) for v in c) for c in es.cellSimplices()]
idx = {c: j for j, c in enumerate(cells_k1)}
circles = [tuple(sorted(v + off for v in h)) for off in (0, nreg, R) for h in holes]
emap = {}
for e in st.getEdgeList().toVector():
    a, b = e.getSource().getId(), e.getTarget().getId()
    emap[(min(a, b), max(a, b))] = e


def set_conformal(s):
    for k, e in emap.items():
        e.setPhase(0.0)
        e.setSquaredLength(-float(s) if (k[0] >= R) != (k[1] >= R) else 1.0)
    st.materializeFacets()


def my_periods(H):
    """Oriented period of each harmonic around each hole circle — a consistent
    convention (target & P use the same one, so it cancels)."""
    P = np.zeros((H.shape[0], len(circles)), dtype=complex)
    for ci, (a, b, c) in enumerate(circles):
        for (x, y) in ((a, b), (b, c), (c, a)):
            key = (min(x, y), max(x, y)); sgn = 1.0 if x < y else -1.0
            if key in idx:
                P[:, ci] += sgn * H[:, idx[key]]
    return P


def harmonics_sparse(k=8):
    L = np.asarray(cob.HodgeLaplacian(st).laplacian(1), dtype=float)
    n = len(cells_k1); L = L.reshape(n, n)
    Ls = sp.csr_matrix(L)
    vals, vecs = spla.eigsh(Ls, k=k, sigma=0.0, which="LM")
    null = vecs[:, np.abs(vals) < 1e-7]
    return null.T.astype(complex)   # (dim, ncells)


set_conformal(1.0)
# --- (a) sparse vs dense ---
t0 = time.time()
Hd = np.asarray(cob.HodgeLaplacian(st).harmonicMatrix(1, 1e-9, False),
                dtype=complex).reshape(-1, len(cells_k1))
td = time.time() - t0
t1 = time.time(); Hs = harmonics_sparse(); ts = time.time() - t1
# does sparse span dense? project each dense harmonic onto sparse null space
Q, _ = np.linalg.qr(Hs.T)                  # ncells x dim ON basis
resid = np.linalg.norm(Hd.T - Q @ (Q.conj().T @ Hd.T)) / np.linalg.norm(Hd.T)
print(f"dense harmonicMatrix: {td:.1f}s (dim {Hd.shape[0]}) | "
      f"sparse null-space: {ts:.2f}s (dim {Hs.shape[0]}) | "
      f"span residual {resid:.2e}", flush=True)

w1 = np.asarray(cob.HodgeLaplacian(st).weights(1), dtype=float)
target = my_periods(Hs)[0]


def energy_sc(s):
    """Self-consistent matter energy at conformal scale s: recompute the register,
    carry the fixed target periods, Dirichlet energy on the live metric."""
    set_conformal(s)
    H = harmonics_sparse()
    P = my_periods(H)
    c, *_ = np.linalg.lstsq(P.T, target, rcond=None)
    h = c @ H
    w = np.asarray(cob.HodgeLaplacian(st).weights(1), dtype=float)
    return float(np.real(np.vdot(h, w * h)))


def action(s):
    set_conformal(s)
    return complex(tessera.ReggeSolver(st, tessera.MatterConfiguration())
                   .dualReggeAction())


# --- (b) the restoring force: scan the conformal scale ---
print("\nconformal scan — does the mass regulate the runaway to an interior min?",
      flush=True)
print(f"{'s':>6} {'ReS':>12} {'|ImS|':>10} {'E_selfcons':>12}", flush=True)
grid = np.linspace(0.4, 3.0, 14)
rows = []
te = time.time()
for s in grid:
    S = action(s); E = energy_sc(s)
    rows.append((s, S.real, abs(S.imag), E))
    print(f"{s:6.2f} {S.real:12.2f} {abs(S.imag):10.2f} {E:12.5f}", flush=True)
print(f"(scan {time.time()-te:.0f}s, {len(grid)} pts → {(time.time()-te)/len(grid):.1f}s/eval)",
      flush=True)

arr = np.array(rows)
for kappa in (0.0, 1e2, 1e3, 1e4):
    G = arr[:, 1] + kappa * arr[:, 3] + 1.0 * arr[:, 2]
    i = int(np.argmin(G)); interior = 0 < i < len(grid) - 1
    print(f"  kappa={kappa:7g}: argmin s*={arr[i,0]:.2f}  "
          f"{'INTERIOR (convergent!)' if interior else 'on the boundary (runaway)'}",
          flush=True)
