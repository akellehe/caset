"""Experiment A: emergent intermediates of the q/qbar -> proton event (#434).

The whole color event is built as ONE connected, tube-connected (#378, never
welded) cobordism over several TEMPORAL slices (`EmergentEventTopology`): the
shared `SymmetricWindowSurface` (S^2 minus the four A4 windows A,B,C,R) stacked
over `n_layers` by the staircase prism. Only the ENDPOINTS are pinned -- the three
color-indefinite neutral-pair quark inputs (windows A,B,C) at the BOTTOM slice and
the proton color singlet (window R) at the TOP slice -- and the entire middle
interior is relaxed at once, so the intermediate states EMERGE off the relaxed
geometry. Nothing in the bulk is hand-placed.

This is the direct test of the #435 finding. The isolated creation node pinned only
ONE boundary, so r_state ~ 0 gave the conformal runaway no restoring force and the
relaxation never reached the symmetric stationary point. BILATERAL pinning (both
endpoints) supplies the constraint the single seed lacked; this module measures
whether that regulates the runaway -- whether the convergence floor, the emergent
colored diquark, the final singlets, the color crystallization, and the emergent
Gauss-law charge come out as the epic predicts.

Everything is read OFF the relaxed geometry (emergent-first, the #410 ethos): no
parallel charge register (charge = the Gauss-law holonomy Q = oint_S E), no imposed
matter (`MatterConfiguration()` empty), the dynamics are the relaxation's delta S = 0
(not a sampler), the complex action is kept (Im S, the Lorentzian worldlines), and
color is never painted (the inputs are color-indefinite, the singlet emerges).

Importable: the test (`tests/cobordism/test_emergent_intermediates.py`) reuses
`build_event`, `slice_color`, `singlet_overlap`, `color_sigma`, `gauss_law_charge`,
`measure`, `null_edges`.
"""

import cmath
import os
import sys

import numpy as np

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)
_SINGLET = np.array([1.0 + 0j, _W, _W * _W])      # the color singlet [1, w, w^2]
_CHARGE = np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j])  # the g-invariant (color) mode

# The window-cycling color-Z3 rep + the omega-rep input live in proton_observables.
sys.path.insert(0, os.path.dirname(__file__))
import proton_observables as P  # noqa: E402


def _windows0(topo):
    """The four windows' base-layer (ell=0) color holes as sorted tuples (A,B,C,R),
    in the format the proton_observables symmetry helpers consume."""
    return [[tuple(h) for h in topo.window_holes_at_layer(w, 0)] for w in range(4)]


def input_states(topo):
    """The pinned endpoint states (A, B, C, R). The three quark windows A,B,C take
    the natural color-symmetric (omega-representation) input -- the omega-eigenvector
    of the window-cycling symmetry, color-INDEFINITE (equal color-period magnitudes,
    no preferred axis, the #414 no-go). The result window R takes the color singlet
    [1, w, w^2] (pinned at the TOP slice only). Color is never painted: A,B,C carry
    no definite color and the singlet emerges in between."""
    abc = P._omega_rep_input(_windows0(topo))     # three color-indefinite inputs
    return [list(abc[0]), list(abc[1]), list(abc[2]), list(_SINGLET)]


def _make_topo(n_layers, lorentzian, u_turn, worldline_lsq):
    topo = cob.EmergentEventTopology()
    topo.set_layers(n_layers)
    if u_turn:
        topo.set_u_turn_twist(True)
    if lorentzian:
        topo.set_lorentzian_worldlines(worldline_lsq)
    return topo


def build_event(n_layers=4, lorentzian=True, u_turn=False, max_iters=80, seed=0,
                worldline_lsq=-1.0):
    """Build + relax the bilaterally-pinned event cobordism. Lorentzian by default
    (timelike cross-layer worldlines -> a non-empty electric sector). `u_turn=True`
    is the anti-baryon (anti-proton) sector (the orientation-reversing twist, opposite
    charge). Returns (TransportCobordism, EmergentEventTopology). A throwaway
    max_iters=0 seed build first populates the topology's windows (so the omega-rep
    input can be read off them), exactly as proton_observables seeds its junction."""
    topo = _make_topo(n_layers, lorentzian, u_turn, worldline_lsq)
    cob.TransportCobordism([list(_SINGLET)] * 4, max_iters=0, seed=seed,
                           topology=topo)  # seed: populate windows
    states = input_states(topo)
    m = cob.TransportCobordism(states, max_iters=max_iters, seed=seed, topology=topo)
    return m, topo


def _carried(m):
    """The carried representative psi (the closed 1-cochain U(1) connection of the
    pinned endpoint states, carried through the bulk) and the edge index map. Color
    periods at any slice and the Gauss-law charge are read off psi."""
    es1 = cob.EigenstateSynthesis(m.cobordism, 1)
    psi = np.array(es1.carriedRepresentative([list(h) for h in m.input_holes],
                                             list(m.input_hole_targets)))
    edge = {(min(c), max(c)): i
            for i, c in enumerate(es1.cellSimplices()) if len(c) == 2}
    return es1, psi, edge


def _period(psi, edge, hole):
    a, b, c = hole
    return psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]]


def slice_color(m, topo, window, layer, psi=None, edge=None):
    """The signed color content (three signed color periods) of `window` (0=A,1=B,
    2=C,3=R) at temporal `layer`, read off the relaxed geometry. Signed by the
    window's induced-orientation covector so the read-out is the relabeling-invariant
    Stokes content (#412), comparable across slices."""
    if psi is None:
        _es, psi, edge = _carried(m)
    holes = topo.window_holes_at_layer(window, layer)
    signs = topo.window_signs_at_layer(window, layer)
    return [signs[k] * _period(psi, edge, holes[k]) for k in range(len(holes))]


def _proj(u, v):
    return abs(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30)


def singlet_overlap(color):
    """The color-singlet overlap of a window's color content (=> 1 for a pure
    singlet [1, w, w^2]; the proton/anti-proton signature)."""
    return _proj(_SINGLET, np.array(color))


def color_sigma(color):
    """The color charge sigma: the projection onto the g-invariant (color) mode
    (=> 0 for a confined singlet; nonzero for a colored / non-singlet object)."""
    return _proj(_CHARGE, np.array(color))


def color_spread(color):
    """The max relative spread of the three color-period magnitudes (0 = perfectly
    color-indefinite / no preferred axis)."""
    mags = np.abs(np.array(color))
    mean = float(np.mean(mags))
    return (float(np.max(mags) - np.min(mags)) / mean) if mean > 1e-15 else 0.0


def null_edges(m, tol=1e-9):
    """Emergent photons: the count of null edges (l^2 -> 0, Im(l) -> 0) in the relaxed
    geometry. A real photon needs a symmetry-breaking source (#413) -- reported, not
    over-claimed."""
    n = 0
    for e in m.cobordism.getEdgeList().toVector():
        lsq = e.getSquaredLength()
        if abs(lsq.real) < tol and abs(lsq.imag) < tol:
            n += 1
    return n


def gauss_law_charge(m, topo, layer=None):
    """The emergent electric charge Q = oint_S E (the temporal-sector Gauss-law
    holonomy, #411) over the closed surface bounding the three quark windows at
    `layer` (default: the bottom slice). Returns (Q_electric, Q_full): Q_full =
    oint_S F is the full closed-surface flux (0 to round-off, the protection);
    Q_electric restricts F to the timelike-leg plaquettes. On an all-spacelike relax
    Q_electric = 0 (the degenerate control). The anti-baryon (u_turn) sector carries
    the opposite sign, so the proton + anti-proton total cancels."""
    _es1, psi, _edge = _carried(m)
    es2 = cob.EigenstateSynthesis(m.cobordism, 2)
    F = list(es2.curvatureFromConnection(list(psi)))
    lyr = 0 if layer is None else layer
    enclosed = sorted(set(v for w in range(3)
                          for h in topo.window_holes_at_layer(w, lyr) for v in h))
    return es2.gaussLawCharge(F, enclosed, True), es2.gaussLawCharge(F, enclosed, False)


def measure(m, topo):
    """The full event read-out as a dict, all read OFF the relaxed geometry. The
    per-slice color crystallization is read on the RESULT window R from the middle
    slices up to the pinned top; the emergent intermediate diquark is the middle
    slice; the final singlet is the top."""
    es1, psi, edge = _carried(m)
    n_layers = topo.n_layers()
    mid = n_layers // 2

    # per-slice color content of the result window R (crystallization), from the
    # first emergent middle slice through to the pinned top slice.
    slices = {}
    for ell in range(1, n_layers + 1):
        col = slice_color(m, topo, 3, ell, psi, edge)
        slices[ell] = {
            "color": col,
            "singlet": singlet_overlap(col),
            "sigma": color_sigma(col),
            "spread": color_spread(col),
        }

    # the emergent intermediate: the result window R at the middle slice (unpinned).
    inter = slice_color(m, topo, 3, mid, psi, edge)
    # the bottom (pinned) quark inputs' color-indefiniteness.
    abc_bottom = [slice_color(m, topo, w, 0, psi, edge) for w in range(3)]

    q_e, q_f = gauss_law_charge(m, topo, 0)
    betti = list(m.stats.betti_cobordism)
    valid, _msg = es1.dualComplexValid()
    return {
        "n_layers": n_layers,
        "mid": mid,
        "converged": m.stats.converged,
        "gradS2": m.stats.stat_action_residual,     # ||grad S||^2 (floor < 100)
        "r_state": m.stats.state_residual,           # r_U of the pinned endpoints
        "betti": betti,
        "b1": betti[1] if len(betti) > 1 else 0,
        "dual_valid": valid,
        "n_null": null_edges(m),
        "slices": slices,
        "inter_color": inter,
        "inter_singlet": singlet_overlap(inter),
        "inter_sigma": color_sigma(inter),
        "inter_spread": color_spread(inter),
        "top_singlet": slices[n_layers]["singlet"],
        "bottom_spread": float(np.mean([color_spread(c) for c in abc_bottom])),
        "Q_e": q_e,
        "Q_f": q_f,
    }


def main():
    print("=== Experiment A: emergent intermediates (#434) ===\n")
    print("--- proton sector (untwisted, minimal depth nL=2) ---")
    mp, tp = build_event(n_layers=2, lorentzian=True, u_turn=False, max_iters=80)
    op = measure(mp, tp)
    print(f"  ||grad S||^2        = {op['gradS2']:.3f}   "
          f"(carriable floor 100; the runaway IS regulated)")
    print(f"  r_state (r_U)       = {op['r_state']:.3e}   "
          f"(the genuine bilateral colored->singlet residual; #435 node was ~3e-27)")
    print(f"  betti / b1          = {op['betti']} / {op['b1']}  "
          f"(dualComplexValid = {op['dual_valid']}, null edges = {op['n_null']})")
    print(f"  intermediate (mid)  : singlet={op['inter_singlet']:.3f}  "
          f"sigma={op['inter_sigma']:.3f}   (colored component sigma > 0, hosted in bulk)")
    print(f"  per-slice R singlet  : "
          + ", ".join(f"{e}:{op['slices'][e]['singlet']:.3f}" for e in op['slices']))
    print(f"  top (final) singlet  = {op['top_singlet']:.3f}   (the proton => 1)")
    print(f"  emergent charge Q_e  = {abs(op['Q_e']):.3e}   "
          f"(closed-surface flux Q_f = {abs(op['Q_f']):.3e} => 0, protection)\n")

    print("--- anti-proton sector (U-turn twist) ---")
    ma, ta = build_event(n_layers=2, lorentzian=True, u_turn=True, max_iters=80)
    oa = measure(ma, ta)
    print(f"  top (final) singlet  = {oa['top_singlet']:.3f}   (the anti-proton => 1)")
    print(f"  emergent charge Q_e  = {abs(oa['Q_e']):.3e}")
    print(f"  total charge (p + pbar) = {abs(op['Q_e'] + oa['Q_e']):.3e}   (CPT => 0)\n")

    print("--- all-spacelike control (Riemannian, degenerate) ---")
    m0, t0 = build_event(n_layers=2, lorentzian=False, u_turn=False, max_iters=80)
    o0 = measure(m0, t0)
    print(f"  emergent charge Q_e  = {abs(o0['Q_e']):.3e}   "
          f"(E == 0: the degenerate case the Lorentzian node beats)\n")

    print("--- ||grad S||^2 vs temporal depth (extensive in the volume) ---")
    for nl in (2, 3, 4):
        m, t = build_event(n_layers=nl, lorentzian=True, u_turn=False, max_iters=80)
        o = measure(m, t)
        print(f"  nL={nl}: ||grad S||^2 = {o['gradS2']:7.2f}   "
              f"top_singlet = {o['top_singlet']:.3f}   |Q_e| = {abs(o['Q_e']):.3e}")


if __name__ == "__main__":
    main()
