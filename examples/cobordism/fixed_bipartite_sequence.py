"""Experiment B: the fixed bipartite sequence (pinned intermediates) (#438).

The SAME connected, tube-connected (#378, never welded) event cobordism as
Experiment A (`EmergentEventTopology`, #434) -- the shared `SymmetricWindowSurface`
(S^2 minus the four A4 windows A,B,C,R) stacked over `n_layers` temporal slices --
reused VERBATIM (`FixedBipartiteSequenceTopology` is a subclass; the build is
inherited), but with the intermediate window ADDITIONALLY pinned to the known
bipartite sequence. Where A lets the intermediate EMERGE, B imposes it and asks
whether the geometry accepts it.

The bipartite sequence (the experiment's independent variable):
  (q_A, q_B, q_C) at the bottom  ->  q_A + q_B -> diquark(3bar) (spectator q_C)
  -> diquark + q_C -> proton at the top.
The colored 3bar diquark is the antisymmetric (wedge / cross-product) combination
d = q_A ^ q_B of the two quark inputs, carried in the conjugate (anti-triplet) rep
via the orientation-reversing #416 twist, pinned on the result window R at every
strictly-interior temporal layer (the bulk, not a stage boundary). The endpoints
A,B,C @ bottom (color-indefinite omega-rep) and R @ top (the color singlet) stay
color-emergent, exactly as in A.

The three questions (all read OFF the converged relaxed geometry, never a
preliminary checkpoint -- ||grad S||^2 is EXTENSIVE in temporal volume, so the
verdict is read only at convergence at the proper depth):
  1. realizability of the known path -- ||grad S||^2 below the per-depth carriable
     floor at convergence (< 100 at nL=2, matching A's 71);
  2. is the pinned colored 3bar diquark HOSTED -- its own r_U (residualForPeriods
     over the diquark holes alone, at the relaxed metric) comparable to A's
     connected-bulk 0.3-0.5 -- or FLOORED (a much higher, free-quark-like residual);
  3. the A-vs-B comparison -- A's EMERGENT intermediate vs B's IMPOSED diquark in a
     common frame (both read by slice_color with the endSignCovector signing).

Everything is emergent-first except the deliberately-pinned intermediate diquark
(the independent variable): no parallel registers (charge = the Gauss-law holonomy),
no imposed matter, the dynamics are the relaxation's delta S = 0, the complex action
is kept (Im S, the Lorentzian worldlines), color is never painted at the endpoints.

Importable: the test reuses `build_event_B`, `diquark_rU`, `measure_B`,
`compare_A_vs_B`.
"""

import os
import sys

import numpy as np

import tessera

sys.path.insert(0, os.path.dirname(__file__))
import emergent_intermediates as A  # noqa: E402  (Experiment A: helpers + baseline)
import proton_observables as P  # noqa: E402

cob = tessera.cobordism


def _make_topo_B(n_layers, lorentzian, u_turn, worldline_lsq, pin_intermediate):
    topo = cob.FixedBipartiteSequenceTopology()
    topo.set_layers(n_layers)
    if u_turn:
        topo.set_u_turn_twist(True)
    if lorentzian:
        topo.set_lorentzian_worldlines(worldline_lsq)
    topo.set_pin_intermediate(pin_intermediate)
    return topo


def build_event_B(n_layers=2, lorentzian=True, u_turn=False, max_iters=80, seed=0,
                  worldline_lsq=-1.0, pin_intermediate=True):
    """Build + relax the fixed-bipartite-sequence event cobordism. Identical
    endpoints to Experiment A (the omega-rep quark inputs A,B,C @ bottom, the color
    singlet R @ top), but the intermediate result window R is additionally pinned to
    the colored 3bar diquark (the #416-twisted antisymmetric q_A ^ q_B), derived in
    C++ from the two pinned quark inputs. Returns (TransportCobordism,
    FixedBipartiteSequenceTopology). `pin_intermediate=False` reproduces A on the
    same subclass (the control)."""
    topo = _make_topo_B(n_layers, lorentzian, u_turn, worldline_lsq, pin_intermediate)
    # seed: populate the windows so the omega-rep input can be read off them. The
    # diquark pin is off for this throwaway build (its A,B states are degenerate
    # placeholders); each TransportCobordism rebuilds, so the main build below
    # re-derives the 3bar from the genuine omega-rep inputs.
    topo.set_pin_intermediate(False)
    cob.TransportCobordism([list(A._SINGLET)] * 4, max_iters=0, seed=seed,
                           topology=topo)
    topo.set_pin_intermediate(pin_intermediate)
    states = A.input_states(topo)  # [A, B, C, R]; the diquark is derived from A,B
    m = cob.TransportCobordism(states, max_iters=max_iters, seed=seed, topology=topo)
    return m, topo


def imposed_diquark(topo):
    """The colored 3bar diquark color that was pinned (the normalized antisymmetric
    q_A ^ q_B), read back from the C++ builder exactly as imposed."""
    states = A.input_states(topo)
    return np.array(topo.diquark_color_for([list(s) for s in states]))


def diquark_rU(m, topo):
    """The pinned diquark's OWN realizability residual r_U: residualForPeriods over
    the diquark holes (window R at the middle slice) ALONE, with its 3bar (twisted)
    targets, at the relaxed metric. NB this single-window measure is DEGENERATE: a
    lone 3-hole window has enough edge DOF to carry any 3 target periods exactly, so
    it is ~0 (1e-25) for A's quark windows, A's singlet, and B's diquark alike --
    non-discriminating. The meaningful hosting measure is `connected_bulk_rU` (the
    JOINT carry of the three quark inputs into the colored diquark). Reported for
    completeness / the ticket's literal phrasing."""
    es1 = cob.EigenstateSynthesis(m.cobordism, 1)
    holes = [list(h) for h in topo.diquark_holes()]
    signs = list(topo.diquark_signs())
    color = imposed_diquark(topo)
    targets = [signs[k] * color[k] for k in range(len(holes))]
    return es1.residualForPeriods(holes, targets)


def connected_bulk_rU(m, topo):
    """THE hosting measure (hosted vs floored): the JOINT realizability residual of
    carrying the three quark inputs A,B,C @ bottom INTO the colored 3bar diquark @
    the middle slice, in one connected-bulk carry (12 holes -- the SAME count as A's
    3-quarks -> singlet whole path, so the two are apples-to-apples). HOSTED if it
    lands near/below A's connected-bulk 0.3-0.5; FLOORED if it is a much higher,
    free-quark-like residual. Unlike the single-window `diquark_rU`, this is
    non-degenerate (the joint multi-window carry is over-determined)."""
    es1 = cob.EigenstateSynthesis(m.cobordism, 1)
    states = A.input_states(topo)
    color = imposed_diquark(topo)
    holes, targets = [], []
    for w in range(3):  # the three quark inputs at the bottom slice
        hs = topo.window_holes_at_layer(w, 0)
        sg = topo.window_signs_at_layer(w, 0)
        for k in range(3):
            holes.append(list(hs[k]))
            targets.append(sg[k] * states[w][k])
    mid = topo.n_layers() // 2  # the colored diquark at the middle slice (3bar twist)
    hs = topo.window_holes_at_layer(3, mid)
    sg = topo.window_signs_at_layer(3, mid)
    for k in range(3):
        holes.append(list(hs[k]))
        targets.append(-sg[k] * color[k])  # the orientation-reversing #416 twist
    return es1.residualForPeriods(holes, targets)


def measure_B(m, topo):
    """The full Experiment-B read-out. Reuses A.measure (gradS2, r_state, slices,
    intermediate, singlets, Gauss-law charge, betti, validity, null edges -- all the
    inherited accessors work on the subclass) and adds the diquark-specific physics:
    the imposed 3bar color, its sigma (colored content), and its hosted-vs-floored
    r_U."""
    o = A.measure(m, topo)  # the shared event read-out (A's, on the B geometry)
    color = imposed_diquark(topo)
    o["diquark_color"] = color
    o["diquark_sigma"] = A.color_sigma(color)        # colored content (=> 0 if singlet)
    o["diquark_singlet"] = A.singlet_overlap(color)  # overlap with [1, w, w^2]
    o["diquark_rU"] = diquark_rU(m, topo)            # single-window (degenerate ~0)
    o["hosting_rU"] = connected_bulk_rU(m, topo)     # THE hosting measure (vs A's 0.3-0.5)
    return o


def compare_A_vs_B(n_layers=2, max_iters=80, seed=0):
    """Question 3: overlap A's EMERGENT intermediate (window R at the middle slice,
    off A's relaxed geometry) against B's IMPOSED diquark (window R at the middle
    slice, off B's relaxed geometry) in a COMMON frame -- both read by slice_color,
    which signs by the induced-orientation covector (endSignCovector, #412), so the
    two are gauge-fixed identically. Returns (overlap, A_inter, B_inter). High
    overlap (>= 0.9) => the geometry WANTS the bipartite path; low => A's emergent
    path diverges from the imposed one."""
    mA, tA = A.build_event(n_layers=n_layers, lorentzian=True, u_turn=False,
                           max_iters=max_iters, seed=seed)
    mB, tB = build_event_B(n_layers=n_layers, lorentzian=True, u_turn=False,
                           max_iters=max_iters, seed=seed)
    midA = tA.n_layers() // 2
    midB = tB.n_layers() // 2
    a_inter = np.array(A.slice_color(mA, tA, 3, midA))   # emergent (A)
    b_inter = np.array(A.slice_color(mB, tB, 3, midB))   # imposed diquark (B)
    overlap = A._proj(a_inter, b_inter)
    return overlap, a_inter, b_inter


def main():
    print("=== Experiment B: fixed bipartite sequence (pinned intermediates) (#438) ===\n")
    print("--- proton sector (untwisted, minimal depth nL=2), CONVERGED ---")
    mp, tp = build_event_B(n_layers=2, lorentzian=True, u_turn=False, max_iters=120)
    op = measure_B(mp, tp)
    print(f"  converged            = {op['converged']}  (iters to plateau)")
    print(f"  ||grad S||^2         = {op['gradS2']:.3f}   "
          f"(per-depth carriable floor 100 at nL=2; A was 71)")
    print(f"  r_state (whole path) = {op['r_state']:.3e}   "
          f"(the full pinned-path realizability residual)")
    print(f"  imposed diquark 3bar : sigma={op['diquark_sigma']:.3f}  "
          f"singlet={op['diquark_singlet']:.3f}   (strong 3bar: sigma > A's ~0.10)")
    print(f"  hosting r_U (HOSTED?) = {op['hosting_rU']:.3e}   "
          f"(3 quarks -> colored diquark joint carry; hosted ~ A's 0.3-0.5)")
    print(f"  diquark r_U (1-window)= {op['diquark_rU']:.3e}   "
          f"(DEGENERATE single-window measure ~0; see hosting_rU)")
    print(f"  betti / b1           = {op['betti']} / {op['b1']}  "
          f"(dualComplexValid = {op['dual_valid']}, null edges = {op['n_null']})")
    print(f"  top (final) singlet  = {op['top_singlet']:.3f}   (the proton => 1)")
    print(f"  emergent charge Q_e  = {abs(op['Q_e']):.3e}   "
          f"(closed-surface flux Q_f = {abs(op['Q_f']):.3e} => 0, protection)\n")

    print("--- A-vs-B comparison (emergent vs imposed intermediate, common frame) ---")
    overlap, a_inter, b_inter = compare_A_vs_B(n_layers=2, max_iters=120)
    print(f"  A emergent  sigma={A.color_sigma(a_inter):.3f}  "
          f"singlet={A.singlet_overlap(a_inter):.3f}")
    print(f"  B imposed   sigma={A.color_sigma(b_inter):.3f}  "
          f"singlet={A.singlet_overlap(b_inter):.3f}")
    print(f"  overlap |<A_inter, B_inter>| = {overlap:.3f}   "
          f"(>= 0.9 => geometry WANTS the bipartite path; low => A diverges)\n")

    print("--- ||grad S||^2 vs temporal depth (extensive in the volume; read at convergence) ---")
    for nl in (2, 3, 4):
        m, t = build_event_B(n_layers=nl, lorentzian=True, u_turn=False, max_iters=120)
        o = measure_B(m, t)
        print(f"  nL={nl}: ||grad S||^2 = {o['gradS2']:7.2f}   "
              f"hosting r_U = {o['hosting_rU']:.3f}   whole r_state = {o['r_state']:7.2f}   "
              f"top_singlet = {o['top_singlet']:.3f}")


if __name__ == "__main__":
    main()
