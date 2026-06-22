"""Is the Experiment-B event a proton? Final-t, proton-localized observables (#449).

The fixed-bipartite-sequence event (#438/#445, `FixedBipartiteSequenceTopology`)
builds a single baryon over temporal slices: three color-indefinite quark inputs
A,B,C at the BOTTOM slice, a colored 3bar diquark pinned in the bulk, the proton
color singlet at the TOP slice. This module asserts the top-slice object IS a proton
-- but it reads the observables CORRECTLY, which on a temporal cobordism means:

  * **final-t only** -- read the proton off the TOP slice / its conserved worldtube,
    NOT the whole temporal worldtube. The tripartite-junction readers (`radius_rms`,
    `shell_deficit` in `proton_observables`) integrate EVERY slice -- the frozen
    input quarks at t=0, the colored diquark in the bulk, the proton at the top --
    so applied to the event they read the whole event, not the proton.
  * **proton, not proton+antiproton** -- a single (untwisted) build is ONE baryon
    (`window_count == 4`: A,B,C,R; no antiproton windows). The antiproton is a
    SEPARATE `u_turn=True` build; the CPT ratio is measured ACROSS the two builds,
    never by summing a conjugate into one reading.

A geometry fact that shapes what is measurable: the event is relaxed with the
ENDPOINTS pinned, so the bottom (t=0) AND top (final-t) slices are FIXED Dirichlet
boundaries -- their spatial edges stay at the uniform seed l^2 = 1. The relaxed
geometry (the only non-trivial metric) is the BULK (the middle slices + the timelike
worldlines). So the proton's metric rest mass/radius are NOT available at its frozen
final-t slice; the proton is asserted by its final-t, proton-localized QUANTUM
NUMBERS (color singlet, sigma -> 0, baryon number, the conserved emergent charge,
the CPT ratio -1). A bulk-worldtube radius is reported ONLY as a labelled diagnostic.

Importable: the test reuses `build_proton`, `build_antiproton`, `final_t_color`,
`proton_charge`, `cpt_charge_ratio`, `bulk_worldtube_radius`, `is_single_baryon`,
`measure_proton`.
"""

import os
import sys

import numpy as np

import tessera

sys.path.insert(0, os.path.dirname(__file__))
import emergent_intermediates as A  # noqa: E402  (slice_color, singlet/sigma, charge)
import fixed_bipartite_sequence as B  # noqa: E402  (the Experiment-B event builder)

cob = tessera.cobordism


def build_proton(n_layers=2, max_iters=120, seed=0):
    """The proton sector of the Experiment-B event (untwisted): three quark inputs
    -> colored 3bar diquark -> proton color singlet at the top slice."""
    return B.build_event_B(n_layers=n_layers, lorentzian=True, u_turn=False,
                           max_iters=max_iters, seed=seed)


def build_antiproton(n_layers=2, max_iters=120, seed=0):
    """The anti-proton sector: the SEPARATE U-turn (#416) build -- a distinct
    cobordism, never the same one as the proton (so no single reading sees both)."""
    return B.build_event_B(n_layers=n_layers, lorentzian=True, u_turn=True,
                           max_iters=max_iters, seed=seed)


def is_single_baryon(topo):
    """The build is ONE baryon, not a proton+antiproton pair: exactly the four
    windows A,B,C,R (an antiproton would need its own windows). Returns the window
    count (== 4 for a single baryon)."""
    return topo.window_count()


def final_t_color(m, topo):
    """The color content of the proton at FINAL-t: window R at the TOP slice ONLY
    (not the bulk diquark, not the input quarks). Returns (singlet_overlap, sigma):
    singlet -> 1 and sigma -> 0 is the confined color-neutral proton."""
    top = topo.n_layers()
    col = A.slice_color(m, topo, 3, top)  # window R (=3) at the top slice
    return A.singlet_overlap(col), A.color_sigma(col)


def _quark_worldtube_vertices(topo, layer):
    """The vertices of the three quark windows A,B,C at a given temporal `layer`."""
    return sorted(set(v for w in range(3)
                      for h in topo.window_holes_at_layer(w, layer) for v in h))


def proton_charge(m, topo, layers=(0,)):
    """The proton's emergent electric charge Q_e = oint_S E (the temporal-sector
    Gauss-law holonomy, #411) over a closed surface enclosing THIS build's three
    quark worldtubes at the given `layers` (default the bottom reference slice, the
    merged epic convention -- the quark inputs are defined there). NB the ABSOLUTE
    magnitude is surface-dependent (it is the charge enclosed by S, not a slice-
    invariant: L0 ~ 0.12, top ~ 3.4, all-tube ~ 1.6) -- the normalization to the
    elementary +1 is NOT fixed here (out of scope). The physical, normalization-free
    statement is the proton:antiproton RATIO (`cpt_charge_ratio`), which is -1 for
    EVERY surface. Returns (Q_e, Q_f); Q_f = oint_S F is the full flux (0, protected)."""
    es1 = cob.EigenstateSynthesis(m.cobordism, 1)
    psi = np.array(es1.carriedRepresentative([list(h) for h in m.input_holes],
                                             list(m.input_hole_targets)))
    es2 = cob.EigenstateSynthesis(m.cobordism, 2)
    F = list(es2.curvatureFromConnection(list(psi)))
    enclosed = sorted(set(v for L in layers
                          for v in _quark_worldtube_vertices(topo, L)))
    return es2.gaussLawCharge(F, enclosed, True), es2.gaussLawCharge(F, enclosed, False)


def cpt_charge_ratio(mp, tp, ma, ta):
    """The proton:antiproton charge ratio Q_p / Q_p_bar, measured ACROSS the two
    SEPARATE builds (never both in one reading) and ROBUST to the Gauss surface: it
    is -1 (CPT) whether the surface encloses the bottom slice, the top slice, or the
    whole worldtube, and Q_p + Q_p_bar -> 0 for each. Returns (ratio, total,
    max_surface_deviation) -- the max |ratio - (-1)| over the three surfaces, the
    evidence the ratio is a genuine surface-robust CPT statement (not a single
    surface's artifact)."""
    top = tp.n_layers()
    surfaces = [(0,), (top,), tuple(range(top + 1))]  # bottom, top, whole tube
    ratios, total = [], 0.0
    for layers in surfaces:
        qp, _ = proton_charge(mp, tp, layers)
        qa, _ = proton_charge(ma, ta, layers)
        ratios.append((qp / qa).real if abs(qa) > 1e-12 else float("nan"))
        if layers == (0,):
            total = abs(qp + qa)
    dev = max(abs(r - (-1.0)) for r in ratios)
    return ratios[0], total, dev


def _layer_of(vid, stride):
    return vid // stride if stride else 0


def final_t_slice_radius(m, topo):
    """The radius read at the FINAL-t slice ONLY: rms of the spacelike edges with
    BOTH endpoints on the top slice. This slice is a FIXED Dirichlet boundary (the
    endpoints are pinned), so it is frozen at the uniform seed l^2 = 1 and returns
    ~1.0 -- it carries NO relaxed metric information. Reported to make the
    frozen-boundary fact explicit, NOT as a physical proton radius."""
    stride = topo.stride()
    top = topo.n_layers()
    sp = [e.getSquaredLength().real for e in m.cobordism.getEdgeList().toVector()
          if _layer_of(e.getSource().getId(), stride) == top
          and _layer_of(e.getTarget().getId(), stride) == top
          and e.getSquaredLength().real > 0]
    return (float(np.mean(sp)) ** 0.5 if sp else 0.0, len(sp))


def bulk_worldtube_radius(m, topo):
    """A DIAGNOSTIC (not a proton rest-frame observable): the rms spacelike edge of
    the RELAXED bulk -- the interior (non-boundary) slices -- where the only
    non-trivial metric lives. Excludes the frozen t=0 and final-t boundary slices.
    This is the worldtube of the whole event (all three quarks + the diquark), so it
    is NOT the proton's radius; it is reported to show the relaxed scale and to
    contrast with the frozen final-t slice."""
    stride = topo.stride()
    top = topo.n_layers()
    sp = []
    for e in m.cobordism.getEdgeList().toVector():
        la = _layer_of(e.getSource().getId(), stride)
        lb = _layer_of(e.getTarget().getId(), stride)
        on_boundary = (la == 0 and lb == 0) or (la == top and lb == top)
        if not on_boundary and e.getSquaredLength().real > 0:
            sp.append(e.getSquaredLength().real)
    return (float(np.mean(sp)) ** 0.5 if sp else 0.0, len(sp))


def measure_proton(n_layers=2, max_iters=120, seed=0):
    """Build the proton and (separately) the anti-proton sectors of the Experiment-B
    event, and read the final-t, proton-localized observables. Returns a dict."""
    mp, tp = build_proton(n_layers, max_iters, seed)
    ma, ta = build_antiproton(n_layers, max_iters, seed)

    singlet, sigma = final_t_color(mp, tp)
    qp, qf = proton_charge(mp, tp)                              # canonical L0 reference
    ratio, total, dev = cpt_charge_ratio(mp, tp, ma, ta)
    a_singlet, _a_sigma = final_t_color(ma, ta)
    return {
        "n_layers": n_layers,
        "window_count": is_single_baryon(tp),       # 4 => one baryon, no antiproton
        "singlet": singlet,                          # proton color singlet (=> 1)
        "sigma": sigma,                              # color charge (confined => 0)
        "Q_e": abs(qp),                              # Gauss-law holonomy (L0 reference)
        "Q_f": abs(qf),                              # full flux (protected => 0)
        "cpt_ratio": ratio,                          # Q_p / Q_p_bar (=> -1)
        "cpt_total": total,                          # |Q_p + Q_p_bar| (=> 0)
        "cpt_surface_dev": dev,                      # max |ratio+1| over 3 surfaces (=> 0)
        "anti_singlet": a_singlet,                   # anti-proton singlet (=> 1)
        "final_t_radius": final_t_slice_radius(mp, tp),    # frozen boundary (=> 1.0)
        "bulk_radius": bulk_worldtube_radius(mp, tp),      # relaxed scale (diagnostic)
    }


def main():
    print("=== Is the Experiment-B event a proton? (final-t, proton-localized) (#449) ===\n")
    o = measure_proton(n_layers=2, max_iters=120)

    print("-- structure: ONE baryon, not a proton+antiproton pair --")
    print(f"  window_count          = {o['window_count']}   (A,B,C,R = 4; antiproton is a SEPARATE build)\n")

    print("-- color at FINAL-t (window R at the top slice only) --")
    print(f"  color singlet overlap = {o['singlet']:.4f}   (the proton => 1)")
    print(f"  color charge sigma    = {o['sigma']:.3e}   (confinement => 0)\n")

    print("-- electric charge (Gauss-law holonomy; proton-only build) --")
    print(f"  Q_e (L0 reference)    = {o['Q_e']:.4f}   (magnitude is surface-dependent; +1 norm out of scope)")
    print(f"  full flux |Q_f|       = {o['Q_f']:.2e}   (protected => 0)")
    print(f"  CPT ratio Q_p/Q_pbar  = {o['cpt_ratio']:+.4f}   (proton:antiproton => -1)")
    print(f"  ratio surface-robust  = {o['cpt_surface_dev']:.2e}   (max |ratio+1| over bottom/top/whole-tube => 0)")
    print(f"  total |Q_p + Q_pbar|  = {o['cpt_total']:.2e}   (=> 0)")
    print(f"  anti-proton singlet   = {o['anti_singlet']:.4f}\n")

    print("-- radius: the final-t slice is a FROZEN Dirichlet boundary --")
    rt, nt = o["final_t_radius"]; rb, nb = o["bulk_radius"]
    print(f"  final-t slice radius  = {rt:.4f}  ({nt} edges)  <- frozen seed l^2=1, NO relaxed info")
    print(f"  bulk worldtube radius = {rb:.4f}  ({nb} edges)  <- DIAGNOSTIC (whole-event tube, not the proton)\n")

    proton = (o["window_count"] == 4 and o["singlet"] >= 0.95 and o["sigma"] <= 0.05
              and abs(o["cpt_ratio"] + 1.0) <= 0.05 and o["cpt_total"] <= 1e-3)
    print(f"VERDICT: {'PROTON' if proton else 'NOT a clean proton'} "
          f"(color singlet, confined, single baryon, CPT-conjugate charge to its antiproton)")


if __name__ == "__main__":
    main()
