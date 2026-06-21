# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Flavor as a split of the existing register --- the negative result (#414).

This reads the two candidate ``u``/``d`` flavor measurables OFF the relaxed ``W_ABC``
junction (`examples/cobordism/proton_observables.py`), per input window (A, B, C),
emergent-first --- and shows both collapse to a window-INDEPENDENT value, so neither
distinguishes the proton (``uud``, ``Q = +1``) from the neutron (``udd``, ``Q = 0``):

  * candidate (i)  spatio-temporal split --- split each window's signed-edge period,
    and its field strength ``F = dpsi``, by the causal type (timelike/spacelike) of
    the cells it rides on. On the relaxed geometry there are ZERO timelike edges, so
    the electric (timelike-leg) sector is empty: every window's split is 100% spacelike
    and the per-window timelike fraction is identically 0 (margin 0 across A, B, C).

  * candidate (ii) Dirac-Kahler taste --- the per-window conserved charge
    ``q_k = <Phi_k, Phi_k>_W`` (#415). It is EQUAL across the three windows to machine
    precision (the A4-symmetric apex interior, #413) AND positive-definite (a norm /
    constituent count), so it can neither separate the windows nor sign the opposite
    charges ``u: +2/3`` vs ``d: -1/3``. ``multiplicity() == 4`` is a fixed framework
    constant (4 lattice tastes, not a 2-valued isospin), with no per-window taste
    projector.

The root cause is structural, not dimensional (the dimensional reach is #429's): the
three quark windows are ONE A4 orbit (the window-cycling ``g`` is a 3-cycle, transitive
on {A, B, C}), and the transport intertwines color Z3 (``M P_in = P_out M``, residual
~1e-14). Any per-window measurable that is invariant under base-vertex relabeling --- the
#412 G6 property the flavor audit (F4) REQUIRES --- is therefore constant on the orbit,
so its u-vs-d discriminator margin is identically 0. Relabeling-invariance (F4) and a
real discriminator (F3, margin >= 0.1) are mutually exclusive on the symmetric register:
flavor is a symmetry-ODD label, the singlet/confinement is a symmetry-EVEN fact, and they
cannot share the same A4-symmetric geometry. The resolution is a symmetry-BREAKING isospin
structure (new construction), not a split of the existing register.

Run:  python examples/cobordism/proton_flavor_split.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import proton_observables as P  # noqa: E402

import tessera  # noqa: E402

cob = tessera.cobordism

# the per-flavor electric charges we would need to assign (the target, never reached)
_Q_UP, _Q_DOWN = 2.0 / 3.0, -1.0 / 3.0


def build(max_iters=25):
    """The relaxed symmetric W_ABC junction with the natural omega-rep proton input."""
    seed = cob.TransportCobordism([[1, -1, 0], [1, 0, -1], [0, 1, -1]],
                                  max_iters=0, seed=0,
                                  topology=cob.TripartiteRegisterTopology())
    states = P._omega_rep_input(P._windows(seed))
    return cob.TransportCobordism(states, max_iters=max_iters, seed=0,
                                  topology=cob.TripartiteRegisterTopology())


def causal_census(m):
    """(timelike, spacelike, null) edge counts on the relaxed metric."""
    tl = sp = nu = 0
    for e in m.cobordism.getEdgeList().toVector():
        if e.isNull():
            nu += 1
        elif e.isTimelike():
            tl += 1
        else:
            sp += 1
    return tl, sp, nu


def _per_window_carry(m):
    """Carry a unit input on each input window k -> carried representative psi_k."""
    es1 = cob.EigenstateSynthesis(m.cobordism, 1)
    windows = P._windows(m)
    psis = [np.array(es1.carriedRepresentative([list(h) for h in windows[k]],
                                               [1.0, 1.0, 1.0])) for k in range(3)]
    return es1, windows, psis


def candidate_i_period_split(m):
    """(i) per-window signed-edge period, split by the causal type of each edge.

    Returns the per-window timelike fraction tl/(tl+sp); identically 0 (every window
    boundary edge is spacelike)."""
    es1, windows, psis = _per_window_carry(m)
    eidx = {(min(c), max(c)): i
            for i, c in enumerate(es1.cellSimplices()) if len(c) == 2}
    timelike = {}
    for e in m.cobordism.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        timelike[(min(a, b), max(a, b))] = e.isTimelike()
    fracs = []
    for k in range(3):
        tl = sp = 0.0
        for (a, b, c) in windows[k]:
            for (x, y) in [(a, b), (b, c), (a, c)]:
                key = (min(x, y), max(x, y))
                mag = abs(psis[k][eidx[key]])
                if timelike.get(key, False):
                    tl += mag
                else:
                    sp += mag
        fracs.append(tl / (tl + sp + 1e-30))
    return fracs


def candidate_i_field_split(m):
    """(i') per-window field strength F = dpsi, split into E (timelike-leg) and B.

    Returns (electric_fraction, Q_electric, n_electric_cells) per window."""
    es1, windows, psis = _per_window_carry(m)
    es2 = cob.EigenstateSynthesis(m.cobordism, 2)
    out = []
    for k in range(3):
        F = list(es2.curvatureFromConnection(list(psis[k])))
        split = es2.fieldStrengthSplit(F)
        ne = float(np.linalg.norm(np.array(split.electric)))
        nb = float(np.linalg.norm(np.array(split.magnetic)))
        enclosed = sorted(set(v for h in windows[k] for v in h))
        qe = abs(es2.gaussLawCharge(F, enclosed, True))
        out.append((ne / (ne + nb + 1e-30), qe, len(split.electricCells)))
    return out


def candidate_ii_dk_charge(m):
    """(ii) per-window Dirac-Kahler charge q_k and its (non-negative) density bounds."""
    es1, windows, psis = _per_window_carry(m)
    dk = cob.DiracKahler(m.cobordism)
    out = []
    for k in range(3):
        field = dk.lift(1, list(psis[k]))
        q = dk.charge(field)
        dens = np.array(dk.chargeDensity(field))
        out.append((q, float(dens.min())))
    return out, dk.multiplicity()


def label_vector(m):
    """The would-be per-window u/d label = sign of (q_k - mean q).  On the symmetric
    interior the three q_k are equal, so this is the zero vector --- it cannot encode
    either uud or udd, and the two are indistinguishable (margin 0)."""
    dkr, _mult = candidate_ii_dk_charge(m)
    qs = np.array([q for q, _ in dkr])
    margin = float(qs.max() - qs.min())
    labels = np.sign(qs - qs.mean()) * (np.abs(qs - qs.mean()) > 0.1)
    return labels.tolist(), margin


def main():
    m = build(25)
    tl, sp, nu = causal_census(m)
    print("FLAVOR AS A SPLIT OF THE EXISTING REGISTER --- the negative result (#414)\n")
    print(f"causal census (relaxed metric): timelike={tl}  spacelike={sp}  null={nu}")
    print("  -> the electric (timelike-leg) sector is unpopulated; there is no causal")
    print("     split to read off candidate (i) on the relaxed geometry.\n")

    fracs = candidate_i_period_split(m)
    print("candidate (i)  period causal-split   [per-window timelike fraction]:")
    print("  A,B,C = " + ", ".join(f"{x:.3e}" for x in fracs)
          + f"   margin = {max(fracs) - min(fracs):.3e}")

    fi = candidate_i_field_split(m)
    print("candidate (i') field E/B split       [E-fraction, Q_electric, #E-cells]:")
    for k, (ef, qe, ne) in enumerate(fi):
        print(f"  {'ABC'[k]}: Efrac={ef:.3e}  Q_E={qe:.3e}  nE={ne}")
    print(f"  margin(Efrac) = {max(e for e, _, _ in fi) - min(e for e, _, _ in fi):.3e}")

    dkr, mult = candidate_ii_dk_charge(m)
    print("candidate (ii) Dirac-Kahler charge   [q_k, density min]:")
    for k, (q, dmin) in enumerate(dkr):
        print(f"  {'ABC'[k]}: q={q:.6f}  dens_min={dmin:.3e}  (>= 0: a norm, never -1/3)")
    qs = [q for q, _ in dkr]
    print(f"  margin(q) = {max(qs) - min(qs):.3e}   multiplicity = {mult} (fixed; not 2)")

    labels, margin = label_vector(m)
    print(f"\nwould-be per-window u/d label vector = {labels}   (margin {margin:.3e})")
    print("  -> the label vector is constant: it is neither uud nor udd, and the two")
    print("     assignments are indistinguishable.  No symmetry-respecting split of the")
    print("     register carries flavor (F4 relabeling-invariance forces margin 0,")
    print("     mutually exclusive with F3's required discriminator margin >= 0.1).")
    print(f"\ntarget electric charges (unreachable here): u={_Q_UP:+.3f}  d={_Q_DOWN:+.3f}")
    print("  Q(uud) = +1, Q(udd) = 0 require a symmetry-BREAKING isospin structure")
    print("  (new construction), NOT a split of the A4-symmetric color register.")


if __name__ == "__main__":
    main()
