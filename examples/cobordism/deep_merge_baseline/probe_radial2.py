"""Probe v4: radial relaxation with a FIXED matter field (computed once on the
reference geometry) — E(s)=Σ w1(s)|h|², cheap per eval. Minimize the matter-coupled
free energy over the radial profile; read curvature(d) of the emergent dual."""
import os, sys, time
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
sys.argv = ["probe"]
exec(open("/tmp/probe_deep.py").read().split("for level in")[0])

LEVEL, LAM = 2, 1.0


class RadialDeep:
    def __init__(self, level=LEVEL):
        self.st, self.nreg, self.holes3, self.hole_vs, self.cells = build_merge(level)
        self.R = 2 * self.nreg
        self.st.materializeFacets()
        self.circles = [list(tuple(sorted(v + off for v in h)))
                        for off in (0, self.nreg, self.R) for h in self.holes3]
        self.dist = shells(self.st, self.hole_vs)
        self.emap = {}
        for e in self.st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            self.emap[(min(a, b), max(a, b))] = e
        self.tl = {k: min(self.dist.get(k[0], 99), self.dist.get(k[1], 99))
                   for k in self.emap if (k[0] >= self.R) != (k[1] >= self.R)}
        self.ds = sorted(set(self.tl.values()))
        self.es = cob.EigenstateSynthesis(self.st, 1)
        self.apply(np.ones(len(self.ds)))
        # the FIXED matter field h: the carried register harmonic at the
        # reference (uniform) geometry — computed once.
        H = np.asarray(cob.HodgeLaplacian(self.st).harmonicMatrix(1, 1e-9, False),
                       dtype=complex).reshape(-1, len(self.es.cellSimplices()))
        P = np.asarray(self.es.cyclePeriods(self.circles),
                       dtype=complex).reshape(H.shape[0], len(self.circles))
        c, *_ = np.linalg.lstsq(P.T, P[0], rcond=None)
        h = c @ H
        self.h2 = np.abs(h)**2 / float(np.real(np.vdot(h, h)))   # unit-norm |h|²
        self.dim = int(H.shape[0])

    def apply(self, s):
        for k, e in self.emap.items():
            e.setPhase(0.0)
            if (k[0] >= self.R) != (k[1] >= self.R):
                e.setSquaredLength(-float(s[self.ds.index(self.tl[k])]))
            else:
                e.setSquaredLength(1.0)
        self.st.materializeFacets()

    def action(self):
        return complex(tessera.ReggeSolver(self.st, tessera.MatterConfiguration())
                       .dualReggeAction())

    def energy(self):
        w1 = np.asarray(cob.HodgeLaplacian(self.st).weights(1), dtype=float)
        return float(np.sum(w1 * self.h2))

    def G(self, s, kappa):
        self.apply(s); S = self.action()
        return S.real + kappa * self.energy() + LAM * abs(S.imag)

    def curvature(self):
        bins = defaultdict(list)
        for sx in self.st.getSimplices():
            vs = [v.getId() for v in sx.getVertices()]
            if len(vs) == 2:
                d = min(self.dist.get(vs[0], 99), self.dist.get(vs[1], 99))
                bins[d].append(sx.lorentzianDeficitAngle().real)
        return {d: float(np.mean(v)) for d, v in bins.items()}


t0 = time.time(); rd = RadialDeep()
print(f"setup {time.time()-t0:.0f}s; shells {rd.ds}; ker L1={rd.dim}", flush=True)
te = time.time(); rd.apply(np.ones(len(rd.ds))); _ = rd.G(np.ones(len(rd.ds)), 1.0)
print(f"one G eval: {time.time()-te:.2f}s", flush=True)
s0 = np.ones(len(rd.ds))
for kappa in (0.0, 1e4, 1e5):
    t1 = time.time()
    res = minimize(lambda s: rd.G(s, kappa), s0, method="Nelder-Mead",
                   bounds=[(0.3, 2.5)] * len(rd.ds),
                   options={"maxiter": 350, "xatol": 3e-3, "fatol": 2e-3})
    rd.apply(res.x); cur = rd.curvature()
    print(f"\nkappa={kappa:g} ({time.time()-t1:.0f}s, nit={res.nit}): G*={res.fun:.2f}",
          flush=True)
    print("  s*(d) =", " ".join(f"{v:.3f}" for v in res.x), flush=True)
    print("  cur(d)=", " ".join(f"{cur[d]:+.3f}" for d in sorted(cur)), flush=True)
