"""Probe v3: the radial relaxation. Generalize #312's (s_wt,s_bulk) to a radial
profile s(d) over the level-2 shells; relax the matter-coupled free energy
G = Re S + kappa E + lam |Im S|; read curvature(d) of the emergent dual. Does the
matter source propagate a falloff?"""
import importlib.util, os, sys, time
from collections import defaultdict, deque
import numpy as np
from scipy.optimize import minimize
sys.argv = ["probe"]
exec(open("/tmp/probe_deep.py").read().split("for level in")[0])  # helpers

LEVEL, LAM = 2, 1.0


class RadialDeep:
    def __init__(self, level=LEVEL):
        self.st, self.nreg, self.holes3, self.hole_vs, self.cells = build_merge(level)
        self.st.materializeFacets()
        A, B, R = 0, self.nreg, 2 * self.nreg
        self.circles = [list(tuple(sorted(v + off for v in h)))
                        for off in (A, B, R) for h in self.holes3]
        self.dist = shells(self.st, self.hole_vs)
        self.emap = {}
        for e in self.st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            self.emap[(min(a, b), max(a, b))] = e
        # timelike (result-crossing) edges, binned by shell distance
        self.tl_by_d = defaultdict(list)
        for (a, b), e in self.emap.items():
            if (a >= R) != (b >= R):
                d = min(self.dist.get(a, 99), self.dist.get(b, 99))
                self.tl_by_d[d].append((a, b))
        self.ds = sorted(self.tl_by_d)
        self.es = cob.EigenstateSynthesis(self.st, 1)
        self.set_profile(np.ones(len(self.ds)))
        P = np.asarray(self.es.cyclePeriods(self.circles),
                       dtype=complex).reshape(self.dim, len(self.circles))
        self.target = P[0]

    def set_profile(self, s):
        for (a, b), e in self.emap.items():
            e.setPhase(0.0)
            if (a >= 2 * self.nreg) != (b >= 2 * self.nreg):
                d = min(self.dist.get(a, 99), self.dist.get(b, 99))
                e.setSquaredLength(-float(s[self.ds.index(d)]))
            else:
                e.setSquaredLength(1.0)
        self.st.materializeFacets()
        self.H = np.asarray(cob.HodgeLaplacian(self.st).harmonicMatrix(1, 1e-9, False),
                            dtype=complex).reshape(-1, len(self.es.cellSimplices()))
        self.dim = int(self.H.shape[0])

    def action(self):
        return complex(tessera.ReggeSolver(self.st, tessera.MatterConfiguration())
                       .dualReggeAction())

    def energy(self):
        w1 = np.asarray(cob.HodgeLaplacian(self.st).weights(1), dtype=float)
        P = np.asarray(self.es.cyclePeriods(self.circles),
                       dtype=complex).reshape(self.dim, len(self.circles))
        c, *_ = np.linalg.lstsq(P.T, self.target, rcond=None)
        h = c @ self.H
        return float(np.real(np.vdot(h, w1 * h)))

    def G(self, s, kappa):
        self.set_profile(s)
        S = self.action()
        return S.real + kappa * self.energy() + LAM * abs(S.imag)

    def curvature_by_shell(self):
        bins = defaultdict(list)
        for sx in self.st.getSimplices():
            vs = [v.getId() for v in sx.getVertices()]
            if len(vs) != 2: continue
            d = min(self.dist.get(vs[0], 99), self.dist.get(vs[1], 99))
            bins[d].append(sx.lorentzianDeficitAngle().real)
        return {d: float(np.mean(v)) for d, v in bins.items()}


rd = RadialDeep()
print(f"shells with timelike edges: {rd.ds}  (counts "
      f"{[len(rd.tl_by_d[d]) for d in rd.ds]})")
t0 = time.time(); s0 = np.ones(len(rd.ds)); g0 = rd.G(s0, 0.0)
print(f"one G eval: {time.time()-t0:.2f}s ; ker L1={rd.dim} ; "
      f"S={rd.action():.2f} ; E={rd.energy():.4f}")

for kappa in (0.0, 50.0, 200.0):
    t1 = time.time()
    res = minimize(lambda s: rd.G(s, kappa), s0, method="Nelder-Mead",
                   bounds=[(0.3, 2.5)] * len(rd.ds),
                   options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-4})
    rd.set_profile(res.x)
    cur = rd.curvature_by_shell()
    print(f"\nkappa={kappa}: relaxed in {time.time()-t1:.0f}s, G*={res.fun:.3f}")
    print("  s*(d)  =", " ".join(f"{v:.3f}" for v in res.x))
    print("  curv(d)=", " ".join(f"{cur[d]:.3f}" for d in sorted(cur)))
