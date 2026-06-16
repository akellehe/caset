# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Backreaction: matter sources the emergent dual on the merge substrate.

The framework's two halves couple, in the only well-posed way: the carried
state's stress-energy **shifts the action-selected, damping-regulated emergent
dual**. We do NOT solve a discrete Einstein equation from scratch — the bulk
is intrinsically curved (no flat vacuum), and the Lorentzian Regge action has
a conformal mode that runs away. Instead, following the stationary-phase /
Sorkin-damping selection (the merged `level1_stationary_phase` result): the
emergent dual is the carrier the damped action concentrates on, and the matter
re-ranks the carriers.

The substrate is the **merge** cobordism (two co-incoming states on slice t,
the bulk merging them to a single object at t+1; coordinate-free, the
input→result edges timelike). The carriers are merges of varying bulk-step
geometry — parametrized here by two scales: the **worldtube** timelike edges
(those incident to the holonomy cycles, where the register lives) and the
**bulk** timelike edges (the rest). Every carrier realizes the merge (the
period residual is machine-zero — the carried space is topological), so the
selection is driven purely by

    A_kappa(W)  =  S_Regge(W)  +  kappa * E(W),

the complex dual Lorentzian (Sorkin) action plus the coupling to the carried
state's field energy E (the Riemannian register norm — the matter). The
emergent dual is the damping-regulated selection

    W*(kappa)  =  argmin_W  [ Re S_Regge(W) + kappa * E(W) + lambda * |Im S_Regge(W)| ],

with |Im S| the boost damping (from the spacelike-hinge rapidities). The
headline, physical and well-posed:

  * **The matter regulates the conformal mode.** At kappa = 0 both Re S and
    the damping |Im S| fall monotonically with the conformal scale — the
    action alone runs the emergent dual to the largest geometry (the
    conformal runaway; the damping does not pin it either). The matter is the
    only term with an interior minimum, so it provides the **restoring force**
    that makes the emergent dual finite — stress-energy sources the scale of
    spacetime, rather than deflecting an already-finite one.
  * **It sources the geometry where the charge lives.** The carried register
    sits on the holonomy worldtubes, so E depends far more strongly on the
    worldtube scale than the bulk scale; the matter pins the worldtube
    direction of the emergent dual first and tightest — charge curves the
    fill, near the charge.
  * **Lorentzian and realizable throughout.** S is complex (Im S the
    spacelike-hinge boosts); every carrier's period residual is machine-zero
    (realizability is topological, independent of the metric).

Run:
    python examples/cobordism/backreaction_emergent_dual.py
    python examples/cobordism/backreaction_emergent_dual.py --plot
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
MC = _load("merge_cobordism")
np = BASE.np
tessera = MC.tessera
cob = tessera.cobordism

LAM = 1.0                       # the boost-damping regulator weight
GRID = np.linspace(0.5, 2.5, 9)  # carrier scales (worldtube x bulk)


class EmergentDual:
    """The merge substrate with a two-scale carrier family: the worldtube
    timelike edges (incident to a holonomy cycle — where the register sits)
    and the remaining bulk timelike edges. Each (s_wt, s_bulk) is a carrier
    of the merge; the action and the matter energy are read off it."""

    def __init__(self):
        self.m = MC.MergeCobordism()
        self.m.st.materializeFacets()
        self.holes = [list(t) for t in self.m.hole_circles]
        self.emap = {}
        for e in self.m.st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            self.emap[(min(a, b), max(a, b))] = e
        tl = [k for k in self.emap
              if self.m._is_result(k[0]) != self.m._is_result(k[1])]
        holev = {v for c in self.m.hole_circles for v in c}
        self.wt = [k for k in tl if (k[0] in holev or k[1] in holev)]
        self.bulk = [k for k in tl if k not in set(self.wt)]
        self.set_scales(1.0, 1.0)
        # a genuinely carried 9-period target (a register harmonic's own
        # periods) — the realizability gate, machine-zero at every carrier
        P = np.asarray(self.m.es.cyclePeriods(self.holes),
                       dtype=complex).reshape(self.m.dim, 9)
        self.target = P[0]

    def set_scales(self, s_wt, s_bulk):
        for k in self.wt:
            self.emap[k].setSquaredLength(-float(s_wt))
        for k in self.bulk:
            self.emap[k].setSquaredLength(-float(s_bulk))
        self.m.st.materializeFacets()
        self.m.read_spectral()

    def action(self):
        return self.m.regge_action()                       # complex (Sorkin)

    def energy(self):
        w1 = np.asarray(cob.HodgeLaplacian(self.m.st).weights(1), dtype=float)
        P = np.asarray(self.m.es.cyclePeriods(self.holes),
                       dtype=complex).reshape(self.m.dim, 9)
        c, *_ = np.linalg.lstsq(P.T, self.target, rcond=None)
        h = c @ self.m.H
        return float(np.real(np.vdot(h, w1 * h)))

    def residual(self):
        return float(self.m.es.residualForPeriods(
            self.holes, [complex(z) for z in self.target]))

    def scan(self, grid=GRID):
        """The carrier landscape over (s_wt, s_bulk)."""
        out = {}
        for sw in grid:
            for sb in grid:
                self.set_scales(sw, sb)
                S = self.action()
                out[(round(float(sw), 3), round(float(sb), 3))] = {
                    "ReS": S.real, "ImS": S.imag,
                    "E": self.energy(), "residual": self.residual()}
        return out

    @staticmethod
    def select(scan, kappa, lam=LAM):
        """The emergent dual: argmin of the damping-regulated, matter-coupled
        free energy over the carriers."""
        best, bg = None, np.inf
        for k, v in scan.items():
            g = v["ReS"] + kappa * v["E"] + lam * abs(v["ImS"])
            if g < bg:
                bg, best = g, k
        return best


def _interior(grid):
    return (grid[0], grid[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="/tmp/cobordism")
    args = ap.parse_args()

    checks = []

    def check(label, passed):
        checks.append((label, bool(passed)))
        return bool(passed)

    print("Backreaction: matter sources the emergent dual on the merge "
          "substrate\n  (the stress-energy regulates the action's conformal "
          "runaway and pins the\n  damping-selected emergent geometry — "
          "shifting it with the coupling kappa)\n")
    prog = BASE._progress()
    prog.phase("scanning the carrier family")
    ed = EmergentDual()
    scan = ed.scan()
    prog.finish(f"{len(scan)} carriers scored")

    lo, hi = _interior(GRID)
    res_max = max(v["residual"] for v in scan.values())
    im_min = min(abs(v["ImS"]) for v in scan.values())
    print(f"  carriers: {len(scan)} merges over (s_wt, s_bulk) in "
          f"[{lo},{hi}]²; max period residual = {res_max:.1e} "
          f"(realizable); min |Im S| = {im_min:.2f} (>0: Lorentzian).")
    check("every carrier realizes the merge (period residual machine-zero — "
          "realizability is topological)", res_max < 1e-9)
    check("the action is complex on every carrier (Lorentzian — the "
          "spacelike-hinge boosts)", im_min > 1e-6)

    # -- the conformal runaway at kappa = 0 --------------------------------- #
    w0 = EmergentDual.select(scan, 0.0)
    on_edge = (w0[0] in (lo, hi)) or (w0[1] in (lo, hi))
    print(f"\n  kappa = 0 (action + damping only): emergent dual at "
          f"(s_wt, s_bulk) = {w0} — {'on the grid edge: the CONFORMAL RUNAWAY' if on_edge else 'interior'}.")
    check("at kappa=0 the action+damping run the emergent dual to the edge "
          "(the conformal mode is not pinned)", on_edge)

    # -- the matter is the regulator (interior minimum, worldtube-dominated)  #
    Emin = min(scan.values(), key=lambda v: v["E"])
    wEmin = [k for k, v in scan.items() if v["E"] == Emin["E"]][0]
    E_edge = min(v["E"] for k, v in scan.items()
                 if k[0] in (lo, hi) or k[1] in (lo, hi))
    e_int = wEmin[0] not in (lo, hi) and wEmin[1] not in (lo, hi)
    print(f"  matter energy E: interior minimum at {wEmin} "
          f"(E={Emin['E']:.5f}); E rises toward the edges "
          f"(min edge E={E_edge:.5f}) — the restoring force.")
    # worldtube-dominance: E varies more along s_wt than s_bulk
    mid = GRID[len(GRID) // 2]
    e_along_wt = [scan[(round(float(s), 3), round(float(mid), 3))]["E"] for s in GRID]
    e_along_bulk = [scan[(round(float(mid), 3), round(float(s), 3))]["E"] for s in GRID]
    span_wt = max(e_along_wt) - min(e_along_wt)
    span_bulk = max(e_along_bulk) - min(e_along_bulk)
    print(f"  matter sensitivity: ΔE along worldtube scale = {span_wt:.5f} vs "
          f"along bulk scale = {span_bulk:.5f} "
          f"(ratio {span_wt/max(span_bulk,1e-12):.1f}× — the charge lives on "
          f"the worldtubes).")
    check("the matter energy has an interior minimum (the regulator that "
          "pins the runaway)", e_int)
    check("the matter sources the worldtube direction far more than the bulk "
          "(charge curves the fill near the charge)", span_wt > 3 * span_bulk)

    # -- the emergent dual: pinned and shifted by the matter ---------------- #
    print("\n  emergent dual W*(kappa) = argmin[Re S + kappa·E + "
          f"{LAM}|Im S|]:")
    traj = []
    for kappa in (0.0, 100.0, 300.0, 600.0, 1200.0, 2500.0, 5000.0):
        w = EmergentDual.select(scan, kappa)
        traj.append((kappa, w))
        edge = (w[0] in (lo, hi)) or (w[1] in (lo, hi))
        print(f"      kappa={kappa:>6.0f}: (s_wt, s_bulk) = {w}"
              f"{'  [edge: runaway]' if edge else '  [interior: matter-pinned]'}")
    pinned = [k for k, w in traj if not ((w[0] in (lo, hi)) or (w[1] in (lo, hi)))]
    moved = traj[-1][1] != traj[0][1]
    check("a finite coupling pins the emergent dual to a finite interior "
          "geometry (the matter regulates the conformal mode)", len(pinned) > 0)
    check("the emergent dual shifts with kappa (the matter sources it)", moved)
    # the pinned worldtube scale tracks the matter's preferred scale
    final = traj[-1][1]
    check("the matter-pinned emergent dual sits at the matter's preferred "
          "worldtube scale", abs(final[0] - wEmin[0]) <= (GRID[1] - GRID[0]) + 1e-9)

    if args.plot:
        from backreaction_emergent_dual_plot import render
        path = render(scan, traj, GRID, args.out)
        print(f"\n  landscape + trajectory plot: {path}")

    if not args.no_write:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "backreaction_emergent_dual.json"),
                  "w") as h:
            json.dump({"scan": {f"{k[0]},{k[1]}": v for k, v in scan.items()},
                       "trajectory": [(kp, list(w)) for kp, w in traj],
                       "E_min_at": list(wEmin), "kappa0_at": list(w0),
                       "span_wt": span_wt, "span_bulk": span_bulk}, h, indent=2)

    ok = all(p for _l, p in checks)
    if not ok:
        print("\n  FAILED checks:")
        for label, passed in checks:
            if not passed:
                print(f"      - {label}")
    print("\n  Verdict: " + (
        "SUPPORTED — backreaction on the merge substrate: every carrier "
        "realizes the merge, the action is Lorentzian (complex), and the "
        "matter stress-energy regulates the action's conformal runaway — "
        "providing the restoring force that pins the damping-selected "
        "emergent dual at a finite scale and shifts it with the coupling, "
        "sourced most strongly on the holonomy worldtubes where the charge "
        "lives. The stress-energy sources the scale of the emergent "
        "spacetime."
        if ok else
        "NOT SUPPORTED — a claim failed; inspect the FAILED checks above."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
