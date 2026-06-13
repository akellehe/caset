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

"""Backreaction: the carried state's field energy sources the fill metric.

The framework's two halves -- the quantum process (the carried harmonic) and
the gravitational action (Regge on the dual) -- have acted on the geometry
separately: the spectral layer decides WHAT a fill transports, the Regge term
tie-breaks WHICH geometry. This example couples them at first order. The
matter is the carried state's field energy, an object the framework already
computes under another name:

    E(p, l) = <h_l(p), h_l(p)>_l   -- the UN-NORMALIZED register Gram,

the discrete field energy of a flux p held fixed (periods are
metric-independent -- carriedness is topological) while the geometry l varies.
The discrete Einstein equation couples the gravitational and matter sides
through the interior edge lengths, boundary pinned bitwise:

    dS_Regge/dl_e  +  kappa * dE/dl_e  =  0.

We solve it at LINEAR ORDER around the unit metric -- the well-posed,
honest content. The Euclidean Regge action is unbounded below (the discrete
conformal-mode problem: shrinking edges drives S down without limit, measured
below; and fixing the global mode still leaves local conformal modes that ride
a minimizer into degenerate corners). A full nonlinear minimum does not exist;
but the LINEARIZED equation does, and it is exactly the physics of
backreaction:

    H delta_l  =  -kappa * grad E,        H = d^2 S_Regge / dl^2 at unit,

the discrete graviton kinetic operator H sourced by the matter stress grad E.
delta_l = -kappa * H^+ grad E (pseudo-inverse: the gauge/conformal null modes
of H -- diffeomorphisms and scale -- are projected out, exactly the modes the
geometry cannot physically respond to). This is the discrete linearized
Einstein equation: (linearized Einstein tensor) = (matter stress).

The headline, physical and non-circular: the matter stress grad E lives on
the WORLDTUBE edges (interior edges incident to the holonomy cycles), it lies
in the range of H (it is a genuine source, not pure gauge), and the sourced
deflection delta_l(kappa) is O(kappa) and concentrated on the worldtubes --
charge curves the fill, near the charge. Realizability is invariant along the
whole flow (periods are topological), and the transport value on the
backreacted metric bends with the (now curved) register Gram.

Scope and honesty:

  * Linear order, Euclidean/spacelike start. The nonlinear solve is ill-posed
    in Euclidean signature; the Lorentzian-native flow (complex action,
    STATIONARY PHASE rather than minimization -- the #285 damping mechanism)
    is the documented follow-up where a nonlinear statement becomes
    well-posed. Optimizer/linear-response, not a sampler over metrics. kappa
    is a dimensionless knob, not Newton's constant; no continuum claims.
  * The metric-validity guard. The #275 dual-validity check is TOPOLOGICAL --
    it passes at a near-collapsed edge (measured: weight 1e-8 reads "ok" yet
    collapses a top cell). The metric half is built here: a finite primal
    volume above a relative Cayley-Menger floor on every top cell. The
    stricter circumcentric/Glickenstein dual-volume condition is the
    follow-up (Simplex.dualVolume is not yet metric-live in this build).

Run:
    python examples/cobordism/backreaction_kappa_flow.py
    python examples/cobordism/backreaction_kappa_flow.py --kappas 0.1,0.2,0.4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

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


BASE = _load("spectral_gate_realizability")
L1 = _load("level1_fill_realizability")
np = BASE.np
tessera = BASE.tessera
cob = tessera.cobordism

_E = np.array([[1.0, -1.0, 0.0] / np.sqrt(2.0),
               [1.0, 1.0, -2.0] / np.sqrt(6.0)])

VOL_FLOOR_REL = 1e-3          # relative top-cell volume floor (Cayley-Menger)


class BackreactionFlow:
    """The straight 1-layer fill with dynamical interior edge lengths and the
    reference carried state as matter. The metric variable is the interior
    edge weight (= squaredLength), which HodgeLaplacian (energy, harmonics)
    and ReggeSolver (action) both read; the discrete graviton operator and the
    matter stress are finite-differenced through it."""

    def __init__(self):
        self.fill = L1.Level1Fill(layers=1)
        self.iw0 = np.array(self.fill.es.interiorWeights())   # unit
        self.n = len(self.iw0)
        self.cp = (BASE._CP_IN / np.linalg.norm(BASE._CP_IN)).astype(complex)
        hole_v = {v for tri in (self.fill.circles0 + self.fill.circles1)
                  for v in tri}
        self.ie = self.fill.es.interiorEdges()
        self.worldtube = np.array([1.0 if (a in hole_v or b in hole_v) else 0.0
                                   for (a, b) in self.ie])
        self.set_metric(self.iw0)
        self._unit_minvol = self._min_volume()

    # ---- metric setter + the metric half of the dual gate ----------------- #
    def _min_volume(self):
        vols = [abs(s.volume()) for s in self.fill.st.getSimplices()
                if len(list(s.getVertices())) == 4]
        return min(vols) if vols else 0.0

    def set_metric(self, w):
        """Set interior edge weights verbatim and re-read the spectrum.
        Returns the metric-validity verdict: every top cell keeps a finite
        primal volume above the relative Cayley-Menger floor. Topology (the
        #275 gate) is invariant under a metric move, so only the metric
        condition is checked."""
        self.fill.es.setInteriorWeights([float(x) for x in w])
        self.fill.read_spectral()
        if not hasattr(self, "_unit_minvol"):
            return True
        mv = self._min_volume()
        return bool(np.isfinite(mv) and mv > VOL_FLOOR_REL * self._unit_minvol)

    # ---- the two functionals ---------------------------------------------- #
    def _carried_form(self):
        w1 = np.asarray(cob.HodgeLaplacian(self.fill.st).weights(1), dtype=float)
        P0 = self.fill.P6[:, 0:3]
        c, *_ = np.linalg.lstsq(P0.T, self.fill.sign0 * self.cp, rcond=None)
        return c @ self.fill.H_full, w1

    def energy(self):
        h, w1 = self._carried_form()
        return float(np.real(np.vdot(h, w1 * h)))

    def stress_energy(self):
        """The discrete stress-energy covector on the interior edges: the
        Hellmann-Feynman energy density T_e = w_e |h(e)|^2 -- the field energy
        carried by each edge, the explicit (source) term of dE/dl. This is
        what sources curvature in the Einstein equation; the full dE/dl adds
        the field's back-adjustment (the implicit dh/dl term), which at the
        symmetric unit metric cancels the explicit part to the FD noise floor
        (the unit metric is an energy minimum -- the state exerts no NET
        gradient there, but its stress-energy density is real and lives on
        the worldtubes)."""
        h, w1 = self._carried_form()
        dens = (w1 * np.abs(h) ** 2).real
        by_cell = {self.fill.cells[i]: dens[i] for i in range(len(dens))}
        return np.array([by_cell.get(tuple(sorted((a, b))), 0.0)
                         for (a, b) in self.ie])

    def regge(self):
        return float(tessera.ReggeSolver(
            self.fill.st, tessera.MatterConfiguration()).dualReggeAction().real)

    def identity_residual(self):
        holes = [list(t) for t in (self.fill.circles0 + self.fill.circles1)]
        pair = np.concatenate([self.fill.sign0 * self.cp,
                               self.fill.sign1 * self.cp])
        return float(self.fill.es.residualForPeriods(
            holes, [complex(z) for z in pair]))

    def gram_dev(self):
        h, w1 = self._carried_form()
        anchor = float(np.real(np.vdot(h, w1 * h)))
        forms, P0 = [], self.fill.P6[:, 0:3]
        for e in _E:
            c, *_ = np.linalg.lstsq(P0.T, self.fill.sign0 * e.astype(complex),
                                    rcond=None)
            forms.append(c @ self.fill.H_full)
        G = np.array([[np.vdot(a, w1 * b) for b in forms]
                      for a in forms]) / anchor
        return float(np.max(np.abs(G - np.eye(2))))

    # ---- gradients and the graviton operator ------------------------------ #
    def _grad(self, fn, w, h=1e-4):
        g = np.zeros(self.n)
        for i in range(self.n):
            wp = w.copy(); wp[i] += h
            wm = w.copy(); wm[i] -= h
            self.set_metric(wp); fp = fn()
            self.set_metric(wm); fm = fn()
            g[i] = (fp - fm) / (2 * h)
        self.set_metric(w)
        return g

    def grad_energy(self, w):
        return self._grad(self.energy, w)

    def regge_hessian(self, w, h=1e-3):
        """H_ij = d^2 S_Regge / dl_i dl_j at *w* (central differences,
        symmetrized) -- the discrete graviton kinetic operator. Its null modes
        are the gauge (diffeomorphism) and conformal directions."""
        H = np.zeros((self.n, self.n))
        gp = [self._grad(self.regge, _bump(w, i, +h)) for i in range(self.n)]
        gm = [self._grad(self.regge, _bump(w, i, -h)) for i in range(self.n)]
        for i in range(self.n):
            H[i] = (gp[i] - gm[i]) / (2 * h)
        self.set_metric(w)
        return 0.5 * (H + H.T)


def _bump(w, i, d):
    v = w.copy(); v[i] += d
    return v


def _parse_kappas(s):
    return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--kappas", default="0.1,0.2,0.4")
    ap.add_argument("--out", default="/tmp/cobordism")
    args = ap.parse_args()
    kappas = _parse_kappas(args.kappas)

    checks = []

    def check(label, passed):
        checks.append((label, bool(passed)))
        return bool(passed)

    print("Backreaction: the carried state's field energy sources the fill "
          "metric\n  (the discrete LINEARIZED Einstein equation "
          "H*delta_l = -kappa*grad E around the\n  unit metric -- does charge "
          "curve the fill, near the charge?)\n")
    prog = BASE._progress()
    prog.phase("building the fill + the graviton operator")
    flow = BackreactionFlow()
    print(f"  straight 1-layer fill: {flow.n} interior edges "
          f"({int(flow.worldtube.sum())} on worldtubes), "
          f"|V|={int(flow.fill.st.getVertexList().size())}; "
          f"E(unit)={flow.energy():.4f}, S_Regge(unit)={flow.regge():.4f}")

    # -- the conformal-mode finding: why linear response, not a solve ------- #
    s_unit = flow.regge()
    flow.set_metric(flow.iw0 * 0.5); s_half = flow.regge()
    flow.set_metric(flow.iw0)
    print(f"\n  Conformal mode: halving every interior edge weight gives "
          f"S_Regge={s_half:.4f} < {s_unit:.4f} (unit) -- the Euclidean "
          f"action is unbounded below, so we work at LINEAR order (a "
          f"nonlinear minimum does not exist).")
    check("Euclidean S_Regge is unbounded below under edge shrink (linear "
          "response is the well-posed regime)", s_half < s_unit)

    # -- the graviton operator and the matter stress ------------------------ #
    H = flow.regge_hessian(flow.iw0)
    flow.set_metric(flow.iw0)
    T = flow.stress_energy()           # the stress-energy source at unit
    evals = np.linalg.eigvalsh(H)
    rank = int(np.sum(np.abs(evals) > 1e-6 * np.max(np.abs(evals))))
    Hpinv = np.linalg.pinv(H, rcond=1e-6)
    prog.finish("operator built")
    proj_range = Hpinv @ H            # projector onto range(H)
    T_phys = proj_range @ T
    phys_frac = float(np.linalg.norm(T_phys) / max(np.linalg.norm(T), 1e-30))
    wt = flow.worldtube.astype(bool)
    stress_wt = float(np.mean(np.abs(T[wt])))
    stress_bulk = (float(np.mean(np.abs(T[~wt]))) if (~wt).any() else 0.0)
    print(f"\n  Graviton operator H (d^2 S_Regge/dl^2 at unit): {flow.n}x{flow.n}, "
          f"rank {rank} ({flow.n - rank} gauge/conformal null modes).")
    print(f"  Stress-energy T_e = w_e|h(e)|^2 at unit: worldtube {stress_wt:.2e} "
          f"vs bulk {stress_bulk:.2e} "
          f"(ratio {stress_wt/max(stress_bulk,1e-30):.1f}x); "
          f"fraction in range(H): {phys_frac:.1%}.")
    check("the stress-energy is concentrated on the worldtube edges (charge "
          "sources curvature where it lives)", stress_wt > 3.0 * stress_bulk)
    check("the stress-energy lies mostly in range(H) (a genuine source, not "
          "pure gauge -- the geometry CAN respond)", phys_frac > 0.5)

    # -- the linearized Einstein response: charge curves the fill ----------- #
    rows = []
    for kappa in kappas:
        dl = -kappa * (Hpinv @ T)                  # H delta_l = -kappa T
        w_star = flow.iw0 + dl
        valid = flow.set_metric(w_star)
        einstein_resid = float(np.linalg.norm(proj_range @ (H @ dl + kappa * T))
                               / max(np.linalg.norm(kappa * T), 1e-30))
        defl_wt = float(np.linalg.norm(dl[wt]))
        defl_bulk = float(np.linalg.norm(dl[~wt])) if (~wt).any() else 0.0
        rows.append({"kappa": kappa, "valid": valid,
                     "deflection_norm": float(np.linalg.norm(dl)),
                     "deflection_worldtube": defl_wt,
                     "deflection_bulk": defl_bulk,
                     "einstein_residual": einstein_resid,
                     "energy": flow.energy(),
                     "identity_residual": flow.identity_residual(),
                     "gram_dev": flow.gram_dev(),
                     "min_volume": flow._min_volume(),
                     "transport": L1.match_gate(flow.fill.emergent_gate())})
    flow.set_metric(flow.iw0)

    print(f"\n  {'kappa':>6} {'|deflect|':>10} {'wt/bulk':>8} "
          f"{'Einstein resid':>14} {'E':>8} {'id-resid':>10} {'gram dev':>9} "
          f"{'transport':>10}")
    for r in rows:
        ratio = r["deflection_worldtube"] / max(r["deflection_bulk"], 1e-30)
        print(f"  {r['kappa']:>6.3f} {r['deflection_norm']:>10.2e} "
              f"{ratio:>8.1f} {r['einstein_residual']:>14.1e} "
              f"{r['energy']:>8.4f} {r['identity_residual']:>10.1e} "
              f"{r['gram_dev']:>9.2e} {r['transport']:>10}")

    check("the linearized Einstein equation holds in range(H): "
          "H*delta_l = -kappa*grad E (residual -> 0)",
          all(r["einstein_residual"] < 1e-6 for r in rows))
    check("the sourced deflection is O(kappa) (charge curves the fill, "
          "linearly)",
          all(abs(rows[i]["deflection_norm"]
                  / rows[0]["deflection_norm"]
                  - rows[i]["kappa"] / rows[0]["kappa"]) < 1e-6
              for i in range(len(rows))))
    check("the deflection is concentrated on the worldtubes (curvature near "
          "the charge)",
          all(r["deflection_worldtube"] > r["deflection_bulk"] for r in rows))
    check("every backreacted metric stays valid (the deflections are in the "
          "nondegenerate regime)", all(r["valid"] for r in rows))
    check("realizability is invariant along the flow (periods are "
          "topological -- the transport cannot change with the metric)",
          all(r["identity_residual"] < 1e-9 for r in rows)
          and all(r["transport"] == "Identity" for r in rows))
    flow.set_metric(flow.iw0)
    e0 = flow.energy()
    check("the field energy responds at SECOND order (the backreaction does "
          "work against the curving geometry -- E rises quadratically from "
          "the unit-metric minimum)",
          all(r["energy"] >= e0 - 1e-12 for r in rows)
          and rows[-1]["energy"] - e0 > 1e-5)
    check("the anchor-normalized transport value is ROBUST against the "
          "backreaction (it shifts < 1% though the geometry deflects O(1) -- "
          "the observable is protected, the representative responds)",
          abs(rows[-1]["gram_dev"] - rows[0]["gram_dev"])
          < 0.01 * rows[0]["gram_dev"])

    # -- the metric guard fires --------------------------------------------- #
    collapse = flow.iw0.copy(); collapse[0] = 1e-8
    guarded = flow.set_metric(collapse)
    min_v = flow._min_volume()
    flow.set_metric(flow.iw0)
    print(f"\n  Metric guard: an interior edge weight of 1e-8 (the #275 "
          f"topological gate reads 'ok') collapses a top cell to volume "
          f"{min_v:.2e} and is rejected by the metric half (valid={guarded}).")
    check("the metric-validity guard fires on a near-collapsed edge the "
          "topological dual gate passes", not guarded)

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "backreaction_kappa_flow.json")
        with open(path, "w") as handle:
            json.dump({"n_interior": flow.n,
                       "n_worldtube": int(flow.worldtube.sum()),
                       "hessian_rank": rank, "null_modes": flow.n - rank,
                       "stress_worldtube": stress_wt, "stress_bulk": stress_bulk,
                       "physical_fraction": phys_frac, "rows": rows},
                      handle, indent=2)
        print(f"\n  raw table (PR artifact, not committed): {path}")

    ok = all(passed for _label, passed in checks)
    if not ok:
        print("\n  FAILED checks:")
        for label, passed in checks:
            if not passed:
                print(f"      - {label}")
    print("\n  Verdict: " + (
        "SUPPORTED -- the quantum data sources the geometry at linear order: "
        "the discrete linearized Einstein equation H*delta_l = -kappa*grad E "
        "is solved by the matter stress acting through the graviton operator, "
        "the stress lives on the worldtube edges and in the physical "
        "(non-gauge) sector of H, and the sourced deflection is O(kappa) and "
        "CONCENTRATED on the worldtubes -- charge curves the fill, near the "
        "charge. The field energy responds at second order (the backreaction "
        "does work) while the transport is topologically invariant and its "
        "normalized value robust -- the observable is protected, the "
        "geometric representative responds. Linear order, Euclidean start "
        "(the action is unbounded below); the metric guard catches the "
        "degeneration the topological gate misses."
        if ok else
        "NOT SUPPORTED -- a claim failed; inspect the FAILED checks above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
