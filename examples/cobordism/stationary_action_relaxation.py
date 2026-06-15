# MIT License
# Copyright (c) 2025 Andrew Kelleher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Per-edge stationary-action backreaction relaxation on the merge substrate.

Objective: **Φ = ‖∇S‖² + Γ·r_U** — the STATIONARY action, the discrete Einstein
equation δS = 0 for the FULL COMPLEX Lorentzian (Sorkin) action. This is NOT
minimize-|S| (that collapses to the action's zero set |S|→0) and NOT ‖∇Re S‖²
(that drops the imaginary part and drifts up in overall scale).

  ``‖∇S‖² = ‖∇Re S‖² + ‖∇Im S‖² → 0`` : the Regge action is stationary in the
        variable (per-edge l²) directions — the discrete Einstein equation. The
        imaginary part (boost / spacelike-hinge physics) is INCLUDED — it is
        real physics and constrains the overall scale that the real part alone
        leaves loose (‖∇S‖² has a finite-scale minimum; r_U pins the uniform
        scale). The action is never reduced to Re S.
  ``Γ·r_U`` : the realizability penalty — the geometry must carry the register
        (the charge). Matter is r_U ONLY (``residualForPeriods``); the Dirichlet
        energy E = ⟨ψ, w₁ ψ⟩ is NEVER computed here.

Gradient: ∇Φ = ∇(‖∇S‖²) + Γ·∇r_U. d(‖∇S‖²) = 2 Re(H · conj(g)) =
2[Re(H·Re g) + Im(H·Im g)], g = ∇_VAR S (complex), H = the complex Hessian of S;
each Hessian-vector is ONE finite-difference step of the EXACT analytic
``ReggeSolver.actionGradientExact`` (clean → accurate, no analytic Hessian).
∇r_U is the analytic low-rank perturbation theory (dM/dl² → dψ → d r_U).

Guards against silently re-adding a Dirichlet source or dropping the imaginary
part:
  (1) no energy()/grad_E here — the matter term is a single source;
  (2) ``objective`` asserts Φ == ‖dS‖² + Γ·residualForPeriods every evaluation
      (full complex |dS|²);
  (3) ``check_gradient`` FD-verifies ∇Φ against an FD of Φ before minimizing.

The substrate is the level-0 merge cobordism (:class:`MergeCobordism`): two
inputs merge into a result block; the variable DOF are the per-edge squared
lengths l² of every edge touching the result block (the emergent-dual geometry).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MC = _load("merge_cobordism")
tessera = MC.tessera
cob = tessera.cobordism

NULL_TOL = 1e-7    # |λ| below this is a kernel (harmonic) eigenvalue of M = L₁


class StationaryActionRelaxer:
    """Relax the per-edge geometry of the merge substrate to δS = 0 (stationary
    complex action) subject to the carried register, by minimizing
    Φ = ‖∇S‖² + Γ·r_U over the result-block squared lengths.

    The variable edges ``VAR`` are every edge with at least one endpoint in the
    result block (the input→result worldtubes and the result-internal edges) —
    the emergent-dual geometry. The input (slice-t) spatial edges are held fixed.
    """

    def __init__(self, gamma=1e3, merge=None):
        self.gamma = float(gamma)
        self.m = merge if merge is not None else MC.MergeCobordism()
        self.m.st.materializeFacets()
        self.st = self.m.st
        self.es = self.m.es
        self.dim = self.m.dim
        self.holes = [list(t) for t in self.m.hole_circles]      # 9 circles
        self.M_HOLES = len(self.holes)

        self.cells1 = self.m.cells_k1                             # k=1 cells
        self.n1 = len(self.cells1)
        self.cidx1 = {c: i for i, c in enumerate(self.cells1)}

        self.cc = cob.ChainComplex.fromSpacetime(self.st)
        self.tris = [tuple(sorted(int(v) for v in t))
                     for t in self.cc.kSimplexVertices(2)]
        self.n0 = len(self.cc.kSimplexVertices(0))
        self.n2 = len(self.tris)
        self.d1 = np.asarray(self.cc.boundaryMatrix(1), float).reshape(self.n0, self.n1)
        self.d2 = np.asarray(self.cc.boundaryMatrix(2), float).reshape(self.n1, self.n2)
        self.K1 = self.d1.T @ self.d1

        self.emap, self.EIDX = {}, {}
        for i, e in enumerate(self.st.getEdgeList().toVector()):
            a, b = e.getSource().getId(), e.getTarget().getId()
            self.emap[(min(a, b), max(a, b))] = e
            self.EIDX[(min(a, b), max(a, b))] = i        # position in actionGradientExact()

        self.tris_of = defaultdict(list)                 # triangles touching each edge
        for ti, t in enumerate(self.tris):
            for i in range(3):
                for j in range(i + 1, 3):
                    self.tris_of[(min(t[i], t[j]), max(t[i], t[j]))].append(ti)

        # VAR: every edge touching the result block (the emergent-dual DOF).
        res = self.m._is_result
        self.VAR = sorted(k for k in self.emap if res(k[0]) or res(k[1]))

        # Q: the hole-boundary covector per holonomy circle; leakCol: the cell
        # that absorbs the un-carried period so ψ realizes the target exactly.
        self.Q = np.zeros((self.M_HOLES, self.n1))
        self.leakCol = []
        for q, h in enumerate(self.holes):
            for j in range(3):
                facet = tuple(sorted(h[i] for i in range(3) if i != j))
                self.Q[q, self.cidx1[facet]] += (1.0 if j % 2 == 0 else -1.0)
            self.leakCol.append(self.cidx1[tuple(sorted((h[0], h[1])))])

        self.x0 = np.array([self.emap[k].getSquaredLength().real for k in self.VAR])
        periods0 = np.asarray(self.es.cyclePeriods(self.holes), dtype=complex)
        self.target = periods0.reshape(self.dim, self.M_HOLES)[0].copy()
        self.target_c = [complex(z) for z in self.target]

    # ---- geometry -------------------------------------------------------- #
    def set_var(self, x):
        for k, v in zip(self.VAR, x):
            v = float(v)
            if abs(v) < 1e-6:
                raise RuntimeError(f"null edge {k} (l^2={v:.2e})")
            if abs(v) > 1e6:
                warnings.warn(f"overflow edge {k} (l^2={v:.2e})")
            self.emap[k].setPhase(0.0)
            self.emap[k].setSquaredLength(v)
        self.st.materializeFacets()

    def _L2(self, a, b):
        return 0.0 if a == b else self.emap[(min(a, b), max(a, b))].getSquaredLength().real

    def _weights(self):
        hl = cob.HodgeLaplacian(self.st)
        return (np.asarray(hl.weights(1), float), np.asarray(hl.weights(2), float))

    def _metricL1(self):
        return np.asarray(cob.HodgeLaplacian(self.st).laplacian(1, True, False),
                          dtype=complex).reshape(self.n1, self.n1).real

    def _dS_VAR(self, x):
        """∂S/∂l² over VAR edges (FULL complex, exact analytic Part A) at x; also S."""
        self.set_var(x)
        rs = tessera.ReggeSolver(self.st, tessera.MatterConfiguration())
        dS = rs.actionGradientExact()
        return (np.array([complex(dS[self.EIDX[k]]) for k in self.VAR]),
                complex(rs.dualReggeAction()))

    # ---- analytic dM/dl² (low rank) for the r_U gradient ----------------- #
    def _dM_factors(self, ek, W1, W2, K2, D1d, D1pd):
        """dM/dl²_ek = fa @ fbᵀ (low rank, never densified): the diagonal-weight
        derivatives give one row + one column of M (rank 2: e_je and a vector w,
        which the row and column share by symmetry), plus one rank-1 term per
        triangle touching ek from dW2."""
        je = self.cidx1[ek]
        l2 = self.emap[ek].getSquaredLength().real
        dW1je = np.sign(l2) / (2.0 * np.sqrt(abs(l2)))
        s1 = -0.5 * dW1je / W1[je]**1.5             # d(1/sqrt W1)[je]
        s2 = 0.5 * dW1je / np.sqrt(W1[je])          # d(sqrt W1)[je]
        w = s1 * self.K1[je] * D1d + s2 * K2[je] * D1pd
        e = np.zeros(self.n1)
        e[je] = 1.0
        cols_a, cols_b = [e, w], [w, e]
        for ti in self.tris_of[ek]:
            t = self.tris[ti]
            G = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    G[i, j] = 0.5 * (self._L2(t[0], t[i + 1]) + self._L2(t[0], t[j + 1])
                                     - self._L2(t[i + 1], t[j + 1]))
            detG = np.linalg.det(G)
            if abs(np.sqrt(abs(detG)) / 2.0 - W2[ti]) > 1e-9 or abs(detG) < 1e-12:
                continue
            ind = lambda p, q: (1.0 if (min(t[p], t[q]), max(t[p], t[q])) == ek and p != q
                                else 0.0)
            dG = np.array([[0.5 * (ind(0, i + 1) + ind(0, j + 1) - ind(i + 1, j + 1))
                            for j in range(2)] for i in range(2)])
            dW2ti = (W2[ti] / 2.0) * np.trace(np.linalg.inv(G) @ dG)
            cj = D1pd * self.d2[:, ti]              # D1' @ (column ti of d2)
            cols_a.append(cj)
            cols_b.append((-dW2ti / W2[ti]**2) * cj)
        return np.asarray(cols_a).T, np.asarray(cols_b).T

    # ---- objective ------------------------------------------------------- #
    def objective(self, x):
        """Φ = ‖∇S‖² + Γ·r_U and its gradient at geometry x. Returns
        (Φ, grad, statres=‖∇S‖², r_U, S)."""
        self.set_var(x)
        W1, W2 = self._weights()
        sW1 = np.sqrt(W1)
        M = self._metricL1()
        lam, U = np.linalg.eigh(M)
        null = np.where(np.abs(lam) < NULL_TOL)[0]
        notnull = np.where(np.abs(lam) >= NULL_TOL)[0]
        Un = U[:, null]
        A = self.Q @ Un
        AtAi = np.linalg.inv(A.T @ A)
        Aplus = AtAi @ A.T
        c = Aplus @ self.target
        h = Un @ c
        carried = self.Q @ h
        psi = h.astype(complex).copy()
        for q in range(self.M_HOLES):
            psi[self.leakCol[q]] += self.target[q] - carried[q]
        nrm = np.linalg.norm(psi)
        p = psi / nrm
        Mp = M @ p
        lamR = np.real(np.vdot(p, Mp))
        rho = Mp - lamR * p
        rU = float(np.real(np.vdot(rho, rho)))

        # GRAVITY: g = ∂S over VAR (full complex, exact analytic Part A).
        # statres = Σ|dS|² = ‖∂ Re S‖² + ‖∂ Im S‖² — the Im part is kept.
        rs = tessera.ReggeSolver(self.st, tessera.MatterConfiguration())
        S = complex(rs.dualReggeAction())
        dS = rs.actionGradientExact()
        g = np.array([complex(dS[self.EIDX[k]]) for k in self.VAR])
        statres = float(np.vdot(g, g).real)

        # MATTER: d r_U over VAR via the single-spectrum low-rank dM.
        invlam = 1.0 / (0.0 - lam[notnull])
        Unn = U[:, notnull]
        K2 = self.d2 @ np.diag(1.0 / W2) @ self.d2.T
        D1d, D1pd = 1.0 / sW1, sW1
        drU = np.empty(len(self.VAR))
        for vi, ek in enumerate(self.VAR):
            fa, fb = self._dM_factors(ek, W1, W2, K2, D1d, D1pd)
            dMp = fa @ (fb.T @ p)
            core = (Unn.T @ fa) @ (fb.T @ Un)
            dUn = Unn @ (core * invlam[:, None])
            dA = self.Q @ dUn
            dAplus = -AtAi @ (dA.T @ A + A.T @ dA) @ AtAi @ A.T + AtAi @ dA.T
            dc = dAplus @ self.target
            dh = dUn @ c + Un @ dc
            dcarried = self.Q @ dh
            dpsi = dh.astype(complex).copy()
            for q in range(self.M_HOLES):
                dpsi[self.leakCol[q]] += -dcarried[q]
            drU[vi] = (2.0 * np.real(np.vdot(rho, dMp))
                       + (2.0 / nrm) * np.real(np.vdot(rho, M @ dpsi - lamR * dpsi))
                       - (2.0 * rU / nrm) * np.real(np.vdot(p, dpsi)))

        # d statres = 2 Re(H conj(g)) = 2[Re(H·Re g) + Im(H·Im g)]; each
        # Hessian-vector is ONE FD step of the exact complex actionGradientExact.
        reg, img = g.real.copy(), g.imag.copy()
        nre, nim = np.linalg.norm(reg), np.linalg.norm(img)
        HRe = np.zeros(len(self.VAR), complex)
        HIm = np.zeros(len(self.VAR), complex)
        if nre > 0:
            tre = 1e-6 / nre
            gre, _ = self._dS_VAR(x + tre * reg)
            HRe = (gre - g) / tre
        if nim > 0:
            tim = 1e-6 / nim
            gim, _ = self._dS_VAR(x + tim * img)
            HIm = (gim - g) / tim
        self.set_var(x)                               # restore geometry to x
        grad_stat = 2.0 * (HRe.real + HIm.imag)

        Phi = statres + self.gamma * rU
        grad = grad_stat + self.gamma * drU
        # GUARD: the matter term is exactly Γ·r_U (residualForPeriods); no Dirichlet.
        Phi_guard = statres + self.gamma * self.es.residualForPeriods(
            self.holes, self.target_c)
        assert abs(Phi - Phi_guard) < 1e-5 * (1 + abs(Phi)), (
            f"GUARD TRIPPED: Phi={Phi:.6e} != "
            f"||dS||^2+GAMMA*r_U={Phi_guard:.6e} (stray matter term)")
        return Phi, grad, statres, rU, S

    # ---- finite-difference gradient check (guard 3) ---------------------- #
    def check_gradient(self, x=None, n=5, h=1e-6):
        """FD-verify ∇Φ against an FD of Φ on the first ``n`` VAR edges. Returns
        the max |analytic − FD|. The tolerance is loose: the Hessian-vector term
        in the analytic gradient is itself a finite difference."""
        x = self.x0 if x is None else x
        _, g0, *_ = self.objective(x)
        worst = 0.0
        for i in range(min(n, len(self.VAR))):
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            fd = (self.objective(xp)[0] - self.objective(xm)[0]) / (2 * h)
            worst = max(worst, abs(g0[i] - fd))
        return worst

    # ---- relaxation ------------------------------------------------------ #
    def relax(self, x0=None, maxiter=1000, ftol=1e-12, gtol=1e-10, callback=None):
        """Minimize Φ over the VAR squared lengths with L-BFGS-B (jac from
        ``objective``). Returns the scipy OptimizeResult; the geometry is left at
        the minimizer."""
        x0 = self.x0 if x0 is None else x0

        def fg(x):
            Phi, g, *_ = self.objective(x)
            if callback is not None:
                callback(x, Phi, g)
            return Phi, g

        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       bounds=[(None, None)] * len(self.VAR),
                       options={"maxiter": maxiter, "maxfun": 12 * maxiter,
                                "ftol": ftol, "gtol": gtol})
        self.set_var(res.x)
        return res

    # ---- diagnostics ----------------------------------------------------- #
    def _shells(self):
        """BFS distance (over the spatial subgraph) from the hole vertices —
        the charge sits on the holonomy cycles, so this bins edges by distance
        from the charge."""
        holev = {v for h in self.holes for v in h}
        adj = defaultdict(set)
        for (a, b) in self.emap:
            adj[a].add(b)
            adj[b].add(a)
        dist = {v: 0 for v in holev}
        frontier = list(holev)
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        return dist

    def curvature(self):
        """Mean Re(deficit) and max |Im(deficit)| of the edges, binned by
        distance from the charge — the matter's bending of the dual geometry."""
        dist = self._shells()
        self.st.materializeFacets()
        bins = defaultdict(list)
        for sx in self.st.getSimplices():
            vs = [v.getId() for v in sx.getVertices()]
            if len(vs) == 2:
                d = min(dist.get(vs[0], 99), dist.get(vs[1], 99))
                bins[d].append(sx.lorentzianDeficitAngle())
        return {d: (float(np.mean([z.real for z in v])),
                    float(np.max([abs(z.imag) for z in v])))
                for d, v in sorted(bins.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma", type=float, default=1e3,
                    help="realizability coupling Γ in Φ = ||∇S||² + Γ·r_U")
    ap.add_argument("--maxiter", type=int, default=1000)
    args = ap.parse_args()

    rx = StationaryActionRelaxer(gamma=args.gamma)
    print(f"merge substrate: n1={rx.n1} edges, {len(rx.VAR)} variable, "
          f"Γ={rx.gamma:g}, register dim={rx.dim}", flush=True)
    P0, g0, sr0, rU0, S0 = rx.objective(rx.x0)
    print(f"Φ(x0)={P0:.4f}  (||∇S||²={sr0:.4f}, r_U={rU0:.3e}, S={S0:.2f})", flush=True)
    print(f"gradient check (analytic vs FD): max |Δ| = {rx.check_gradient():.2e}",
          flush=True)

    ref = rx.curvature()
    n = [0]

    def cb(x, Phi, g):
        n[0] += 1
        if n[0] % 10 == 0:
            print(f"  [eval {n[0]}] Φ={Phi:.3f} |grad|={np.linalg.norm(g):.1e} "
                  f"l²∈[{x.min():.2f},{x.max():.2f}]", flush=True)

    res = rx.relax(maxiter=args.maxiter, callback=cb)
    Pf, _, srf, rUf, Sf = rx.objective(res.x)
    rel = rx.curvature()
    print(f"\n=== DONE ({res.nit} iters, success={res.success}) ===", flush=True)
    print(f"  {res.message}", flush=True)
    print(f"  Φ* = {Pf:.4f}  (||∇S||²: {sr0:.3f} → {srf:.3f}, r_U={rUf:.3e}, S={Sf:.2f})",
          flush=True)
    print(f"  l² range [{res.x.min():.3f}, {res.x.max():.3f}]", flush=True)
    print("  ref   Re-deficit(d):", " ".join(f"{ref[d][0]:+.3f}" for d in sorted(ref)),
          flush=True)
    print("  relax Re-deficit(d):", " ".join(f"{rel[d][0]:+.3f}" for d in sorted(rel)),
          flush=True)
    print("  Δ(matter)(d):       ",
          " ".join(f"{rel[d][0] - ref[d][0]:+.3f}" for d in sorted(ref)), flush=True)


if __name__ == "__main__":
    main()
