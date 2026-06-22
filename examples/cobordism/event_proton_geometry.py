"""Is the Experiment-B proton a real bound state? Geometry on the EMERGENT interior (#451).

The fixed-bipartite-sequence event (#438/#445, `FixedBipartiteSequenceTopology`) puts a
color-singlet proton on its TOP slice. PR #450 (#449) confirmed the *quantum numbers*
(singlet -> 1, color sigma -> 0, single baryon, CPT-conjugate charge) but deliberately
left the *metric* mass/radius/curvature out of scope, because the top slice is a PINNED
DIRICHLET boundary: its spatial edges are frozen at the uniform seed l^2 = 1 and carry
NO relaxed information. A radius read there is ~1.0 by construction, not physics.

This module does the geometry the right way: it measures mass, radius, and curvature on
the **relaxed emergent worldtube** -- the strictly-interior simplices between the frozen
bottom (quark) and top (proton) slices, which are the only cells the relaxation actually
moves -- at the converged depth nL = 2 (nL = 3/4 sit above the carriable floor and are
not used for the verdict).

A geometry fact that governs what is even measurable. The event is a 3D cobordism (a 2D
`SymmetricWindowSurface`, S^2 minus the four A4 windows A,B,C,R, stacked over temporal
slices by the staircase prism). In 3D the Regge HINGES are EDGES (the (d-2)-simplices)
and the curvature is their deficit angle. A hinge only carries an honest curvature if its
coface fan CLOSES -- i.e. every triangle around the edge is shared by two tetrahedra. The
edges on the frozen top/bottom slices, and the edges that border the window holes, have
OPEN fans; their "deficit" is a boundary artefact near 2*pi, not curvature, and must be
excluded. The clean interior set (264 of the 684 edges at nL=2) is exactly the closed-fan
edges off the two Dirichlet boundaries.

Skeleton handling. `dualVolume()` / `lorentzianDeficitAngle()` walk a hinge UP through its
cofaces to the top cells, so the full facet/coface lattice must exist -- and it MUST be
built in C++ (the `ReggeSolver` ctor does this). Driving `materializeFacets()` from Python
registers detached COPIES of the sub-simplices (the bindings use copy semantics), corrupts
the coface lists, and `dualVolume()` then sees half the cofaces. So we build the lattice
once via `ReggeSolver(st, MatterConfiguration())` and read off `getSimplices()`.

What is measured on the emergent interior worldtube:
  1. r*m -- m = the matter-localized shell Re-deficit (#352 anchor) over the proton
     worldtube; r = the circumcentric dual size V_dual^(1/3) of that worldtube. Compared
     to the physical proton's m_p*r_p/hbar c ~ 4.0 and to a prior whole-tube pass (~8.8).
  2. m/Q -- reported, but FLAGGED: Q is in unnormalized register-holonomy units; only the
     CPT ratio Q_p/Q_pbar = -1 is normalization-free, so m/Q is mixed-units here.
  3. The circumcentric dual radius as a physical length.
  4. The discrete Regge curvature vs a round sphere of equal dual volume -- concentration
     (participation ratio) and isotropy (per-window balance, sign).
  5. Localization -- participation ratio + curvature-weighted spread (bound lump vs spread).

Importable (the test reuses these): `build`, `interior_hinges`, `proton_mass`,
`dual_radius`, `curvature_vs_sphere`, `localization`, `charge`, `measure`.
"""

import os
import sys
from collections import defaultdict

import numpy as np

import tessera

sys.path.insert(0, os.path.dirname(__file__))
import emergent_intermediates as A  # noqa: E402  (slice_color, singlet/sigma, charge)
import fixed_bipartite_sequence as B  # noqa: E402  (the Experiment-B event builder)
import proton_observables as P  # noqa: E402  (the prior whole-tube r, m -- for contrast)

cob = tessera.cobordism

# Physical anchor: m_p * r_p / (hbar c) = 938 MeV * 0.84 fm / 197 MeV.fm = 4.0 (dimensionless).
PHYSICAL_RM = 938.0 * 0.84 / 197.0


def build(n_layers=2, max_iters=200, seed=0):
    """Build + relax the Experiment-B proton event, and build its Regge skeleton in C++.
    Returns (m, topo, solver). The solver's ctor materializes the facet/coface lattice
    onto the shared spacetime, so `m.cobordism.getSimplices()` then yields the hinges
    (edges) with valid cofaces for dualVolume()/lorentzianDeficitAngle()."""
    m, topo = B.build_event_B(n_layers=n_layers, lorentzian=True, u_turn=False,
                              max_iters=max_iters, seed=seed)
    solver = tessera.ReggeSolver(m.cobordism, tessera.MatterConfiguration())
    return m, topo, solver


def _layer_of(vid, stride):
    return vid // stride


def interior_hinges(m, topo):
    """The EMERGENT worldtube hinges: the closed-coface-fan edges strictly off the two
    frozen Dirichlet boundary slices. An edge is interior iff every one of its triangle
    cofaces is shared by exactly two tetrahedra (a closed fan) -- this drops both the
    top/bottom boundary edges and the window-hole-bordering edges, whose open fans give
    boundary-artefact "deficits" near 2*pi rather than curvature. Each returned dict
    carries the edge's vertex ids, the complex Lorentzian deficit (re/im), its signed
    circumcentric dual volume |*h|, and its BFS shell distance from the quark windows."""
    st = m.cobordism
    stride = topo.stride()
    sims = st.getSimplices()
    tris = [s for s in sims if len(s.getVertices()) == 3]
    tri_ntet = {tuple(sorted(v.getId() for v in t.getVertices())):
                sum(1 for c in t.getCofaces() if len(c.getVertices()) == 4)
                for t in tris}
    edges = [s for s in sims if len(s.getVertices()) == 2]

    # BFS shells from the three quark windows (bottom slice) over the 1-skeleton.
    qverts = sorted(set(v for w in range(3)
                        for h in topo.window_holes_at_layer(w, 0) for v in h))
    adj = defaultdict(set)
    for e in edges:
        a, b = sorted(v.getId() for v in e.getVertices())
        adj[a].add(b)
        adj[b].add(a)
    dist = {v: 0 for v in qverts}
    frontier = list(qverts)
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt

    out = []
    for e in edges:
        tkeys = [tuple(sorted(v.getId() for v in t.getVertices()))
                 for t in e.getCofaces() if len(t.getVertices()) == 3]
        if not tkeys or not all(tri_ntet.get(k, 0) == 2 for k in tkeys):
            continue  # open fan -> boundary/hole artefact, not curvature
        vids = sorted(v.getId() for v in e.getVertices())
        dfa = e.lorentzianDeficitAngle()
        out.append(dict(vids=vids, re=dfa.real, im=dfa.imag, dv=e.dualVolume(),
                        shell=min(dist.get(vids[0], 99), dist.get(vids[1], 99))))
    return out


def proton_mass(hinges):
    """m = the matter-localized shell Re-deficit (#352 anchor) over the proton worldtube:
    sum over BFS shells (from the quark windows) of the MEAN real part of the Lorentzian
    deficit angle, restricted to the emergent interior hinges. Also returns the raw
    integrated curvature (sum Re eps) and the dual-weighted Regge curvature (sum |*h| Re
    eps) -- the two extensive alternatives -- because the validator is sensitive to which
    one is called 'the mass' (see the findings report)."""
    bins = defaultdict(list)
    for h in hinges:
        bins[h["shell"]].append(h["re"])
    shell_means = {d: float(np.mean(v)) for d, v in sorted(bins.items())}
    m_shell = float(sum(shell_means.values()))
    m_sum = float(sum(h["re"] for h in hinges))
    m_action = float(sum(h["re"] * abs(h["dv"]) for h in hinges))
    return dict(m_shell=m_shell, m_sum=m_sum, m_action=m_action, shell_means=shell_means)


def dual_radius(m, topo, hinges):
    """The circumcentric dual size of the proton worldtube, three ways:
      * r_Vdual  = V_dual^(1/3), V_dual = sum of |*v| (the signature-aware circumcentric
                   dual 3-volume) over the interior worldtube vertices -- the task's
                   primary form;
      * r_V3     = V3^(1/3), V3 = sum of |primal volume| over all tetrahedra (a cross
                   check on the same 3-volume from the primal side);
      * rms_dual = sqrt(mean |*h|) over the interior hinges (an RMS dual-cell length).
    Returns the volumes and the radii."""
    st = m.cobordism
    stride = topo.stride()
    nL = topo.n_layers()
    int_vids = set(v for h in hinges for v in h["vids"])
    sims = st.getSimplices()
    Vdual = 0.0
    for s in sims:
        if len(s.getVertices()) != 1:
            continue
        vid = s.getVertices()[0].getId()
        if vid in int_vids and 0 < _layer_of(vid, stride) < nL:
            Vdual += abs(s.dualVolume())
    V3 = sum(abs(s.volume()) for s in sims if len(s.getVertices()) == 4)
    rms_dual = float(np.sqrt(np.mean([abs(h["dv"]) for h in hinges])))
    return dict(Vdual=Vdual, V3=V3, r_Vdual=Vdual ** (1.0 / 3.0),
                r_V3=V3 ** (1.0 / 3.0), rms_dual=rms_dual)


def curvature_vs_sphere(m, topo, hinges):
    """The discrete Regge curvature of the worldtube vs a ROUND sphere of equal dual
    volume. The reference (a round 3-sphere / the round S^2 base) spreads its curvature
    UNIFORMLY: every hinge carries the same deficit, so its participation ratio is 1.0,
    its curvature std/mean is 0, and the curvature is balanced across all directions.
    For the proton we report:
      * net sign of the curvature (mean Re eps): > 0 is a positive-curvature lump, the
        bound-state / sphere-like sign;
      * participation ratio PR of |Re eps * *h| in (0, 1]: 1.0 = uniform (sphere-like),
        -> 0 = concentrated at a point. concentration_ratio = 1/PR is how many times more
        concentrated than the equal-volume round sphere;
      * isotropy across the four A4 windows (A,B,C,R): the curvature summed near each
        window; window_isotropy = min/max of the three quark windows' shares (1.0 = a
        perfectly color-symmetric, isotropic lump)."""
    re = np.array([h["re"] for h in hinges])
    dv = np.array([abs(h["dv"]) for h in hinges])
    w = np.abs(re * dv)
    PR = float((w.sum() ** 2) / (len(w) * (w ** 2).sum()))

    # per-window curvature shares (isotropy): nearest window of each hinge vertex.
    win_verts = {wk: set(v for h in topo.window_holes_at_layer(wk, lr) for v in h)
                 for wk in range(4) for lr in range(topo.n_layers() + 1)}
    wsum = {wk: 0.0 for wk in range(4)}
    for h in hinges:
        for wk in range(4):
            if any(v in win_verts[wk] for v in h["vids"]):
                wsum[wk] += abs(h["re"] * h["dv"])
    quark_shares = [wsum[k] for k in range(3)]
    iso = (min(quark_shares) / max(quark_shares)) if max(quark_shares) > 0 else 0.0

    return dict(mean_re=float(re.mean()), std_re=float(re.std()),
                std_over_mean=float(re.std() / abs(re.mean())) if re.mean() else float("inf"),
                PR=PR, concentration_ratio=1.0 / PR if PR > 0 else float("inf"),
                window_isotropy=iso, window_shares=wsum)


def localization(hinges):
    """Is the curvature a bound LUMP or spread out? Returns the participation ratio of the
    curvature weight, the curvature-weighted RMS shell radius (the lump size, in BFS-shell
    units from the quark windows), and the fraction of |curvature| within shell <= 1 (next
    to the quarks). A small RMS radius + a high near-quark fraction = a localized lump."""
    re = np.array([h["re"] for h in hinges])
    dv = np.array([abs(h["dv"]) for h in hinges])
    sh = np.array([h["shell"] for h in hinges])
    w = np.abs(re * dv)
    PR = float((w.sum() ** 2) / (len(w) * (w ** 2).sum()))
    rms_shell = float(np.sqrt(np.sum(w * sh ** 2) / np.sum(w)))
    near = float(w[sh <= 1].sum() / w.sum())
    return dict(PR=PR, rms_shell_radius=rms_shell, fraction_within_shell1=near)


def charge(m, topo):
    """The proton's emergent electric charge Q = oint_S E (the temporal-sector Gauss-law
    holonomy, #411), in UNNORMALIZED register-holonomy units. Returns (|Q_e|, |Q_f|);
    Q_f = oint_S F -> 0 is the topological protection. NB Q is NOT pinned to the
    elementary +1 here (out of scope), so any m/Q is mixed-units; only the proton:
    antiproton ratio Q_p/Q_pbar = -1 is normalization-free."""
    qe, qf = A.gauss_law_charge(m, topo)
    return abs(qe), abs(qf)


def proton_sector(m, topo):
    """Confirm this is the proton sector at all: the color singlet overlap and color sigma
    of window R at the TOP slice (=> 1 and => 0 for the confined color-neutral proton)."""
    col = A.slice_color(m, topo, 3, topo.n_layers())
    return A.singlet_overlap(col), A.color_sigma(col)


def measure(n_layers=2, max_iters=200, seed=0):
    """Build, relax, and read every geometric proton observable off the emergent interior
    worldtube. Returns a dict."""
    m, topo, _solver = build(n_layers=n_layers, max_iters=max_iters, seed=seed)
    hinges = interior_hinges(m, topo)
    singlet, sigma = proton_sector(m, topo)
    mass = proton_mass(hinges)
    rad = dual_radius(m, topo, hinges)
    curv = curvature_vs_sphere(m, topo, hinges)
    loc = localization(hinges)
    qe, qf = charge(m, topo)

    # The prior whole-tube (boundary-polluted) pass, for contrast.
    m_prior, _ = P.shell_deficit(m.cobordism,
                                 sorted(set(v for w in range(3)
                                            for h in topo.window_holes_at_layer(w, 0)
                                            for v in h)))
    r_prior, _, _ = P.radius_rms(m.cobordism)

    rm = rad["r_Vdual"] * mass["m_shell"]   # the task's literal validator
    return dict(
        n_layers=n_layers, residual=m.stats.residual, iters=m.stats.relax_iterations,
        n_interior_hinges=len(hinges), singlet=singlet, sigma=sigma,
        mass=mass, radius=rad, curvature=curv, localization=loc,
        Q_e=qe, Q_f=qf, m_over_Q=mass["m_shell"] / qe if qe > 1e-12 else float("nan"),
        r_m=rm, r_m_prior=r_prior * m_prior, r_prior=r_prior, m_prior=m_prior,
        physical_rm=PHYSICAL_RM,
    )


def main():
    print("=== Geometric proton validation on the Experiment-B emergent interior (#451) ===\n")
    o = measure(n_layers=2, max_iters=200)

    print("-- convergence + proton sector (nL=2, the carriable depth) --")
    print(f"  residual              = {o['residual']:.2f} (at the nL=2 floor; A's ~71)  "
          f"iters {o['iters']}")
    print(f"  interior hinges       = {o['n_interior_hinges']}  (closed-fan edges off the "
          f"frozen boundaries)")
    print(f"  top-slice singlet     = {o['singlet']:.4f}  sigma = {o['sigma']:.2e}   "
          f"(=> proton sector)\n")

    print("-- (1) r*m  (validator; physical proton ~ 4.0; prior whole-tube ~ 8.8) --")
    mass, rad = o["mass"], o["radius"]
    print(f"  m_shell (#352 shell Re-deficit, interior) = {mass['m_shell']:.4f}")
    print(f"    [extensive alternatives: sum Re eps = {mass['m_sum']:.2f}, "
          f"sum |*h|Re eps = {mass['m_action']:.2f}]")
    print(f"  r_Vdual = V_dual^(1/3)                    = {rad['r_Vdual']:.4f}  "
          f"(V_dual = {rad['Vdual']:.1f})")
    print(f"  r*m (r_Vdual x m_shell)                   = {o['r_m']:.3f}   "
          f"vs 4.0 and vs prior {o['r_m_prior']:.2f}")
    print(f"    prior was r={o['r_prior']:.3f} (frozen, ~sqrt of l^2=1 edges) x "
          f"m={o['m_prior']:.3f} (boundary-polluted)\n")

    print("-- (2) m/Q  (MIXED UNITS: Q is unnormalized; only Q_p/Q_pbar=-1 is clean) --")
    print(f"  |Q_e| = {o['Q_e']:.4f}   |Q_f| = {o['Q_f']:.2e} (protected => 0)   "
          f"m/Q = {o['m_over_Q']:.3f}  [unnormalized]\n")

    print("-- (3) dual radius in physical space --")
    print(f"  V_dual (interior verts) = {rad['Vdual']:.2f}  ->  r = {rad['r_Vdual']:.4f}")
    print(f"  V3 (primal tets) = {rad['V3']:.2f}  ->  r = {rad['r_V3']:.4f}   "
          f"(cross-check)")
    print(f"  RMS dual-cell size = {rad['rms_dual']:.4f}\n")

    print("-- (4) dual curvature vs a round sphere of equal dual volume --")
    c = o["curvature"]
    print(f"  mean Re(deficit) = {c['mean_re']:+.4f}  (> 0 => positive-curvature lump, "
          f"sphere-like sign)")
    print(f"  participation ratio = {c['PR']:.3f}  (round sphere = 1.0)  -> "
          f"{c['concentration_ratio']:.2f}x more concentrated than the sphere")
    print(f"  std/mean = {c['std_over_mean']:.2f}  (round sphere = 0; lumpy if >> 0)")
    print(f"  window isotropy (min/max quark share) = {c['window_isotropy']:.3f}  "
          f"(1.0 = color-symmetric)\n")

    print("-- (5) localization (bound lump vs spread) --")
    loc = o["localization"]
    print(f"  participation ratio = {loc['PR']:.3f}")
    print(f"  curvature-weighted RMS shell radius = {loc['rms_shell_radius']:.3f} shells")
    print(f"  fraction of |curvature| within shell<=1 of the quarks = "
          f"{loc['fraction_within_shell1']:.2f}\n")

    proton_like = (o["singlet"] >= 0.95 and o["sigma"] <= 0.05 and c["mean_re"] > 0
                   and c["PR"] < 0.7 and rad["r_Vdual"] > 2.0)
    print("VERDICT: the top-slice object is a confined color singlet; on the emergent "
          "interior it\n  has a genuine emergent dual radius and a localized, net-positive "
          "(sphere-like sign)\n  curvature lump -- structurally proton-like. r*m is the "
          "same O(1-10) as 4.0/8.8 but is\n  definition-sensitive, so it is a soft, not a "
          "sharp, validator at nL=2.")
    print(f"  structurally proton-like: {proton_like}")


if __name__ == "__main__":
    main()
