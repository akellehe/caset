"""Bipartite q/qbar creation node + the charge<->color bridge (#435).

The creation node (`BipartiteCreationTopology`) is the time U-turn of one fermion
line, realized as a SPLIT of the color register: one seed window relaxes into TWO
emergent windows (a quark q and an antiquark qbar) on one connected surface. It is
the mirror of the trivalent W_ABC junction (`TripartiteRegisterTopology`, three
inputs -> one result), and the elementary building block #434/#438 instantiate.

Three faithfulness commitments (the epic #410 ethos), all read OFF the relaxed
geometry, never hand-placed:

  * pair NEUTRALITY is the Stokes/confinement theorem -- on the connected
    surface-minus-holes the induced periods sum to zero, so
    sigma_q + sigma_qbar = -sigma_seed; a neutral seed gives a neutral pair;
  * the pair is COLOR-INDEFINITE at birth -- a color-symmetric seed [1, w, w^2]
    transports (C3-equivariantly) to equal-magnitude color content with no preferred
    axis; color crystallizes later from assembly context (the #414 no-go);
  * electric charge is EMERGENT -- Q = oint_S E (`gaussLawCharge`) read off the
    relaxed Lorentzian connection (the timelike creation-vertex worldlines populate
    the electric sector), NOT a parallel Q-hat register. An all-spacelike (Riemannian)
    relax gives E = 0 -- the degenerate case to detect, not the target. The
    orientation-reversing U-turn twist (#416) makes qbar the time-reversed (opposite)
    charge of q, so the pair charge cancels (qbar = q backward in time).

The Charged-Cartan assumptions (a parallel Q-hat register, an imposed
pair-Hamiltonian, Metropolis Monte-Carlo, a hand-dialed CP bias) are QUARANTINED:
this is a one-shot relaxation (delta S = 0) and the charge is read, never imposed.

This module is importable: the test (`tests/cobordism/test_bipartite_creation.py`)
reuses `relax_creation`, `emergent_color`, `gauss_law_charges`, `creation_pair_states`
(the bridge handing the two windows to a downstream `TransportCobordism`), and
`measure`.
"""

import cmath

import numpy as np

import tessera

cob = tessera.cobordism
_W = cmath.exp(2j * cmath.pi / 3)  # the cube root of unity (color phase)

# The color-symmetric neutral seed [1, w, w^2]: Sigma = 1 + w + w^2 = 0 (neutral) and
# |each| = 1 (no preferred color axis), so the emergent pair is color-indefinite.
_SEED = [1.0 + 0j, _W, _W * _W]


def relax_creation(worldline_lsq=-1.0, max_iters=25, seed=0, lorentzian=True,
                   u_turn=True, frequency=2):
    """Build + relax the creation node. Lorentzian by default (timelike
    creation-vertex worldlines -> a non-empty electric sector). Returns
    (TransportCobordism, BipartiteCreationTopology): the topology is needed to read
    the SECOND emergent window (qbar), which the single-result read-out cannot hold."""
    topo = cob.BipartiteCreationTopology()
    topo.set_u_turn_twist(u_turn)
    if frequency != 2:
        topo.set_frequency(frequency)
    if lorentzian:
        topo.set_lorentzian_worldlines(worldline_lsq)
    m = cob.TransportCobordism([list(_SEED)], max_iters=max_iters, seed=seed,
                               topology=topo)
    return m, topo


def _carried_seed(m):
    """The carried representative psi -- the kernel (closed) 1-cochain U(1) connection
    of the seed pinned on the seed window, carried through the bulk to the emergent
    windows. F = d psi (its E/B split and the Gauss-law Q) is read off it."""
    es1 = cob.EigenstateSynthesis(m.cobordism, 1)
    seed_holes = [list(h) for h in m.input_holes]
    seed_targets = list(m.input_hole_targets)
    return es1, np.array(es1.carriedRepresentative(seed_holes, seed_targets))


def _period(psi, edge, hole):
    """The signed period of the 1-cochain psi around a hole triangle (a,b,c)."""
    a, b, c = hole
    return psi[edge[(a, b)]] + psi[edge[(b, c)]] - psi[edge[(a, c)]]


def emergent_color(m, topo):
    """The two emergent windows' color content (signed color periods), q then qbar,
    each a length-3 complex list -- the quark from the single-result read-out
    (`m.result()`) and the antiquark read separately over `antiquark_window()` with
    its (U-turn-reversed) signs, off the relaxed geometry."""
    es1, psi = _carried_seed(m)
    edge = {(min(c), max(c)): i
            for i, c in enumerate(es1.cellSimplices()) if len(c) == 2}
    color_q = list(m.result)
    qbar_holes = [list(h) for h in topo.antiquark_window()]
    qbar_signs = list(topo.antiquark_signs())
    color_qbar = [qbar_signs[k] * _period(psi, edge, qbar_holes[k])
                  for k in range(len(qbar_holes))]
    return color_q, color_qbar


def gauss_law_charges(m, topo):
    """Per-window emergent electric charge Q = oint_S E (the temporal-sector Gauss-law
    holonomy, #411) and the full closed-surface flux oint_S F. Returns
    [(Q_e_q, Q_f_q), (Q_e_qbar, Q_f_qbar)]: Q_e restricts F to the timelike-leg
    (electric) plaquettes; on an all-spacelike relax Q_e = 0 (the degenerate case).
    The antiquark charge carries the U-turn sign (qbar = q backward in time), so a
    color-symmetric pair cancels: Q_e_q + Q_e_qbar = 0."""
    _es1, psi = _carried_seed(m)
    es2 = cob.EigenstateSynthesis(m.cobordism, 2)
    F = list(es2.curvatureFromConnection(list(psi)))
    sign_qbar = -1.0 if topo.u_turn_twisted() else 1.0
    out = []
    for win, sgn in ((topo.quark_window(), 1.0),
                     (topo.antiquark_window(), sign_qbar)):
        enclosed = sorted(set(v for h in win for v in h))
        out.append((sgn * es2.gaussLawCharge(F, enclosed, True),
                    sgn * es2.gaussLawCharge(F, enclosed, False)))
    return out


def creation_pair_states(m, topo):
    """THE CHARGE<->COLOR BRIDGE: the two emergent windows as color states ready to
    hand to a downstream `TransportCobordism` (e.g. q+q -> diquark(3bar) in
    #434/#438). Each is the window's three signed color periods (the emergent color
    content); the accompanying emergent charge is `gauss_law_charges(m, topo)`."""
    cq, cqbar = emergent_color(m, topo)
    return [list(cq), list(cqbar)]


def color_spread(color):
    """The max relative spread of the three color-period magnitudes (0 = perfectly
    color-indefinite / no preferred axis)."""
    mags = np.abs(np.array(color))
    mean = float(np.mean(mags))
    return (float(np.max(mags) - np.min(mags)) / mean) if mean > 1e-15 else 0.0


def measure(m, topo):
    """The full creation-node read-out as a dict (the example main + the test). All
    quantities are read off the relaxed geometry."""
    cq, cqbar = emergent_color(m, topo)
    charges = gauss_law_charges(m, topo)
    sigma_q, sigma_qbar = sum(cq), sum(cqbar)
    betti = list(m.stats.betti_cobordism)
    return {
        "converged": m.stats.converged,
        "stat_action_residual": m.stats.stat_action_residual,
        "state_residual": m.stats.state_residual,
        "betti": betti,
        "b1": betti[1] if len(betti) > 1 else 0,
        "temporal_flips": topo.temporal_flip_count(),
        "color_q": cq,
        "color_qbar": cqbar,
        "sigma_q": sigma_q,
        "sigma_qbar": sigma_qbar,
        "sigma_pair": sigma_q + sigma_qbar,        # => 0 (pair neutrality)
        "spread_q": color_spread(cq),              # => 0 (color-indefinite)
        "spread_qbar": color_spread(cqbar),
        "Q_q": charges[0][0],                      # emergent electric charge oint E
        "Q_qbar": charges[1][0],
        "Q_pair": charges[0][0] + charges[1][0],   # => 0 (real cancellation)
        "flux_q": charges[0][1],                   # full closed-surface flux (protection)
        "flux_qbar": charges[1][1],
    }


def main():
    print("=== Bipartite q/qbar creation node (#435) ===\n")
    for label, lorentzian in (("Lorentzian (timelike worldlines)", True),
                              ("all-spacelike (Riemannian, degenerate)", False)):
        m, topo = relax_creation(lorentzian=lorentzian, max_iters=25)
        o = measure(m, topo)
        print(f"--- {label} ---")
        print(f"  converged           = {o['converged']}  "
              f"(||grad S||^2 = {o['stat_action_residual']:.3e}, "
              f"r_state = {o['state_residual']:.3e})")
        print(f"  betti / b1          = {o['betti']} / {o['b1']}  "
              f"(temporal flips = {o['temporal_flips']})")
        print(f"  color q             = {[f'{c:.3f}' for c in o['color_q']]}")
        print(f"  color qbar          = {[f'{c:.3f}' for c in o['color_qbar']]}")
        print(f"  color spread q/qbar = {o['spread_q']:.3e} / {o['spread_qbar']:.3e}"
              f"   (=> 0: color-indefinite)")
        print(f"  sigma pair          = {abs(o['sigma_pair']):.3e}"
              f"   (=> 0: pair neutrality / confinement)")
        print(f"  emergent charge |Q| = q:{abs(o['Q_q']):.3e}  qbar:{abs(o['Q_qbar']):.3e}"
              f"  pair:{abs(o['Q_pair']):.3e}")
        print(f"  closed-surface flux = q:{abs(o['flux_q']):.3e}  "
              f"qbar:{abs(o['flux_qbar']):.3e}   (protection => 0)\n")


if __name__ == "__main__":
    main()
