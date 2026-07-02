# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Mass and radius read the right way off a converged emergent 4D interior.

The #451 geometric-proton methodology ported from its 3D event to the
genuinely 4D emergent builds (#566) — the shared reader core behind the
``MassRadius`` observable and the ``examples/cobordism/emergent_mass_radius.py``
battery script (which imports from here; the machinery lives in exactly one
place). Everything here is a *post-hoc observable reader* — nothing shapes
the lattice.

The load-bearing methodology rules, ported from #451:

  * **Hinges.** On a d = 4 complex the Regge hinges are the (d-2) = 2-simplices —
    TRIANGLES (#451 was 3D, where hinges were edges). Curvature is the complex
    Lorentzian deficit angle at each triangle.
  * **Closed fans only.** A hinge carries an honest curvature ONLY if its coface
    fan is closed — every tetrahedron around the triangle shared by exactly two
    4-cells. Open-fan hinges (the ∂W input boundary, the register-hole walls) give
    boundary artefacts near 2π, not curvature: they are EXCLUDED, and the census
    (interior vs boundary counts) is reported with every reading. The fan test is
    combinatorial over the CURRENT top cells, so orphan sub-simplices stranded by
    Pachner moves never pollute the selection.
  * **Skeleton in C++.** ``dualVolume()`` / ``lorentzianDeficitAngle()`` walk a
    hinge up through its cofaces, so the facet/coface lattice must exist and be
    built in C++: the ``ReggeSolver(st, MatterConfiguration())`` ctor materializes
    tops → tetrahedra → triangle hinges, and ``ChainComplex.fromSpacetime(st)``
    (a C++ BFS seeded from the current tops) completes the lattice down to edges
    and vertices for the dual volumes. Never drive ``materializeFacets`` from
    Python — the #451 lesson: a Python-driven materialization corrupted the coface
    lists and ``dualVolume()`` saw half its cofaces.
  * **Signature-aware readings.** Dual volumes are the circumcentric
    ``Simplex.dualVolume`` (signed); the deficit is the complex
    ``Simplex.lorentzianDeficitAngle``. Masses use Re ε; the imaginary part is
    real physics (boost content), so its magnitude is always reported alongside —
    never silently dropped.
  * **Dimension-correct radius.** The dual radius is r = V_dual^(1/4) on a
    4-complex (V_dual = the summed circumcentric dual 4-volume of the strictly
    interior vertices). #451 used V_dual^(1/3) because its complex was 3D; the
    root tracks the top dimension, nothing else. The primal cross-check is
    r = V_primal^(1/4) over the top 4-cells.
  * **r·m is definition-sensitive.** The #451 lesson: across reasonable mass
    definitions (intensive shell-mean vs the two extensive sums) r·m spans orders
    of magnitude, so the FULL table with its spread is stated up front and any
    agreement with the physical m_p·r_p/ħc ≈ 4.0 is an order-of-magnitude claim
    only — never one number presented as THE mass.

What is measured (per converged spacetime):
  1. interior closed-fan hinge selection + census (interior / boundary / total);
  2. mass, intensive AND extensive: m_shell = the #352/#451 shell-mean Re-deficit,
     m_sum = Σ Re ε, m_action = Σ |★h|·Re ε;
  3. radius: r_dual = V_dual^(1/4) with the r_primal = V_primal^(1/4) cross-check;
  4. localization: participation ratio of |Re ε·★h| against the equal-volume
     uniform (round-sphere) reference, the BFS shell profile of curvature relative
     to the register holes' vertices, and the mean deficit sign;
  5. the r·m table across ALL mass × radius combinations, spread first.
"""
import itertools
import math
from collections import Counter, defaultdict

import numpy as np

import tessera

cob = tessera.cobordism

# Physical anchor: m_p·r_p/ħc = 938 MeV · 0.84 fm / 197 MeV·fm ≈ 4.0 (dimensionless).
PHYSICAL_RM = 938.0 * 0.84 / 197.0

# |Im ε| above this counts as genuinely complex (boost content at the hinge).
IM_TOL = 1e-12


def build_skeleton(st):
    """Materialize the full facet/coface lattice of ``st`` in C++ and return the
    keep-alive handles ``(solver, chain_complex)``.

    ``ReggeSolver(st, MatterConfiguration())`` is the blessed Regge path: its ctor
    builds tops → tetrahedra → triangle hinges, which is everything the deficit
    angles need. ``ChainComplex.fromSpacetime(st)`` then completes the lattice
    down to edges and vertices — a C++ BFS seeded from the CURRENT top cells only
    (orphan-safe) — which the vertex ``dualVolume()`` recursion needs for V_dual.
    Both are C++ constructors; no materialization is ever driven from Python
    (the #451 corruption lesson).
    """
    solver = tessera.ReggeSolver(st, tessera.MatterConfiguration())
    chain_complex = cob.ChainComplex.fromSpacetime(st)
    return solver, chain_complex


def _top_vertex_ids(st):
    """Sorted vertex-id tuples of the current top cells (4-simplices). Fails
    loudly if the complex is not genuinely 4D — this reader is 4D-specific (the
    hinge dimension and the radius root both track d = 4)."""
    tops = [tuple(sorted(v.getId() for v in t.getVertices()))
            for t in st.getTopSimplices()]
    if not tops:
        raise ValueError("spacetime has no top cells")
    sizes = {len(t) for t in tops}
    if sizes != {5}:
        raise ValueError(
            f"top cells have vertex counts {sorted(sizes)}; this reader is for "
            "4-complexes (5-vertex top cells) — on a d-complex the hinges are the "
            "(d-2)-simplices and the radius root is 1/d, so use the reader that "
            "matches your dimension")
    return sorted(tops)


def _bfs_shells(tops, seed_vertex_ids):
    """BFS shell distance from ``seed_vertex_ids`` over the 1-skeleton of the
    CURRENT top cells (combinatorial, orphan-immune). Returns {vertex_id: shell};
    unreachable vertices are absent. Empty seeds give an empty map (every hinge
    then reports shell None and the shell profile is skipped)."""
    adjacency = defaultdict(set)
    for top in tops:
        for a, b in itertools.combinations(top, 2):
            adjacency[a].add(b)
            adjacency[b].add(a)
    dist = {v: 0 for v in sorted(set(seed_vertex_ids)) if v in adjacency}
    frontier = sorted(dist)
    while frontier:
        nxt = []
        for u in frontier:
            for v in sorted(adjacency[u]):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def interior_hinges(st, holes=()):
    """Select the interior (closed-coface-fan) triangle hinges of the 4-complex
    and read their curvature. Returns ``(hinges, census)``.

    A triangle hinge is INTERIOR iff every tetrahedron of its coface fan is shared
    by exactly two top 4-cells; hinges with any once-shared (boundary) tetrahedron
    are boundary artefacts — their "deficit" is a near-2π opening-angle remnant,
    not curvature — and are excluded. Both the fan and the tetrahedron counts are
    derived combinatorially from the CURRENT top cells (never from raw coface
    lists), so orphan simplices stranded by Pachner moves cannot leak in; the
    readings themselves are then taken off the canonical registered triangle
    Simplex objects (skeleton required — call ``build_skeleton`` first).

    Each hinge dict carries: ``vids`` (the 3 vertex ids), ``re``/``im`` (the
    complex Lorentzian deficit), ``dv`` (the signed circumcentric dual content
    |★h|), and ``shell`` (BFS distance from the register holes' vertices, None
    when ``holes`` is empty).

    ``census`` reports the interior/boundary/total hinge counts, the boundary
    tetrahedron count, and the sizes of the complex.
    """
    tops = _top_vertex_ids(st)

    # Tetrahedron -> number of top cells sharing it, and triangle -> its fan.
    tet_count = Counter()
    tri_fans = defaultdict(set)
    for top in tops:
        for tet in itertools.combinations(top, 4):
            tet_count[tet] += 1
        for tri in itertools.combinations(top, 3):
            rest = [v for v in top if v not in tri]
            for extra in rest:
                tri_fans[tri].add(tuple(sorted(tri + (extra,))))

    # Canonical registered triangle Simplex objects, keyed by vertex set.
    tri_by_key = {}
    for s in st.getSimplices():
        if len(s.getVertices()) == 3 and s.hasTopCoface():
            tri_by_key[tuple(sorted(v.getId() for v in s.getVertices()))] = s

    hole_vertex_ids = sorted(set(v for hole in holes for v in hole))
    shell_of = _bfs_shells(tops, hole_vertex_ids)

    hinges = []
    n_boundary = 0
    for tri in sorted(tri_fans):
        closed = all(tet_count[tet] == 2 for tet in tri_fans[tri])
        if not closed:
            n_boundary += 1
            continue
        simplex = tri_by_key.get(tri)
        if simplex is None:
            raise RuntimeError(
                f"interior hinge {tri} has no registered triangle Simplex — "
                "the C++ skeleton was not built; call build_skeleton(st) first")
        deficit = simplex.lorentzianDeficitAngle()
        shell = (min(shell_of[v] for v in tri if v in shell_of)
                 if any(v in shell_of for v in tri) else None)
        hinges.append(dict(vids=list(tri), re=deficit.real, im=deficit.imag,
                           dv=simplex.dualVolume(), shell=shell))

    boundary_tets = sorted(t for t, c in tet_count.items() if c == 1)
    census = dict(
        n_tops=len(tops),
        n_tets=len(tet_count),
        n_hinges_total=len(tri_fans),
        n_hinges_interior=len(hinges),
        n_hinges_boundary=n_boundary,
        n_boundary_tets=len(boundary_tets),
        boundary_tets=boundary_tets,
        n_hole_vertices=len(hole_vertex_ids),
    )
    return hinges, census


def masses(hinges):
    """The three #451 mass readings over the interior hinges — one intensive,
    two extensive (the r·m validator is sensitive to which is called "the mass",
    so all three are always carried together):

      * ``m_shell`` (intensive): the #352/#451 shell mass — the sum over BFS
        shells (from the register holes) of the MEAN Re-deficit in that shell.
        With no holes every hinge sits in one unlabeled bin and m_shell reduces
        to the plain mean Re-deficit.
      * ``m_sum`` (extensive): Σ Re ε — the raw integrated curvature.
      * ``m_action`` (extensive, dual-weighted): Σ |★h|·Re ε — the dual-volume-
        weighted Regge curvature.

    Also returns ``shell_means`` (the per-shell means) and the imaginary-part
    accounting: ``max_abs_im`` and ``n_im_nonzero`` (|Im ε| > IM_TOL) — the
    deficit is complex Lorentzian, and whether Im is zero is always reported.
    """
    if not hinges:
        nan = float("nan")
        return dict(m_shell=nan, m_sum=nan, m_action=nan, shell_means={},
                    max_abs_im=nan, n_im_nonzero=0)
    bins = defaultdict(list)
    for h in hinges:
        bins[h["shell"]].append(h["re"])
    shell_means = {shell: float(np.mean(values))
                   for shell, values in sorted(bins.items(),
                                               key=lambda kv: (kv[0] is None, kv[0]))}
    return dict(
        m_shell=float(sum(shell_means.values())),
        m_sum=float(sum(h["re"] for h in hinges)),
        m_action=float(sum(h["re"] * abs(h["dv"]) for h in hinges)),
        shell_means=shell_means,
        max_abs_im=float(max(abs(h["im"]) for h in hinges)),
        n_im_nonzero=sum(1 for h in hinges if abs(h["im"]) > IM_TOL),
    )


def radii(st, census):
    """The emergent size of the interior, dual and primal:

      * ``r_dual`` = V_dual^(1/4): V_dual = Σ |★v| over the strictly INTERIOR
        vertices (vertices on no boundary tetrahedron) — the signature-aware
        circumcentric dual 4-volume. The fourth root is the dimension-correct
        volume→length map on a 4-complex (#451 took the cube root of a dual
        3-volume on its 3D event; same rule, different d).
      * ``r_primal`` = V_primal^(1/4): V_primal = Σ |V₄| over ALL top 4-cells —
        the primal-side cross-check of the same 4-volume (dual/primal agreement
        is the skeleton-sanity signal; on a closed uniform complex they are
        equal to machine precision).
    """
    boundary_vertex_ids = set(v for tet in census["boundary_tets"] for v in tet)
    v_dual = 0.0
    n_interior_vertices = 0
    vertex_simplices = [s for s in st.getSimplices() if len(s.getVertices()) == 1]
    for s in sorted(vertex_simplices, key=lambda s: s.getVertices()[0].getId()):
        if not s.hasTopCoface():
            continue  # orphan 0-simplex stranded by a move
        vid = s.getVertices()[0].getId()
        if vid in boundary_vertex_ids:
            continue
        v_dual += abs(s.dualVolume())
        n_interior_vertices += 1
    v_primal = sum(abs(t.volume()) for t in st.getTopSimplices())
    return dict(
        Vdual=float(v_dual),
        Vprimal=float(v_primal),
        n_interior_vertices=n_interior_vertices,
        r_dual=float(v_dual) ** 0.25 if v_dual > 0 else float("nan"),
        r_primal=float(v_primal) ** 0.25 if v_primal > 0 else float("nan"),
    )


def localization(hinges):
    """Is the curvature a localized lump or spread out?

      * ``PR``: participation ratio of the weight w = |Re ε·★h| over the interior
        hinges, in (0, 1]. The equal-volume UNIFORM reference (a round sphere
        spreads its curvature evenly over every hinge) has PR = 1.0;
        ``concentration`` = 1/PR is how many times more concentrated the lump is.
      * ``mean_re`` / ``std_re`` / ``std_over_mean``: the deficit statistics;
        mean_re > 0 is the positive-curvature (bound-state, sphere-like) sign.
      * ``shell_profile``: per BFS shell (from the register holes' vertices) the
        hinge count, mean Re ε, and share of the total weight; plus
        ``rms_shell_radius`` (the weight-weighted RMS shell distance — the lump
        size in shells) and ``frac_within_shell1`` (the weight fraction within
        one shell of the holes). Empty when no holes were given.
    """
    if not hinges:
        nan = float("nan")
        return dict(PR=nan, concentration=nan, mean_re=nan, std_re=nan,
                    std_over_mean=nan, shell_profile={}, rms_shell_radius=nan,
                    frac_within_shell1=nan)
    re = np.array([h["re"] for h in hinges])
    weight = np.abs(re * np.array([abs(h["dv"]) for h in hinges]))
    total_weight = float(weight.sum())
    pr = (float(total_weight ** 2 / (len(weight) * float((weight ** 2).sum())))
          if total_weight > 0 else float("nan"))

    shells = [h["shell"] for h in hinges]
    profile = {}
    rms_shell = float("nan")
    frac_near = float("nan")
    if all(s is not None for s in shells) and total_weight > 0:
        shell_arr = np.array(shells, dtype=float)
        for shell in sorted(set(shells)):
            mask = shell_arr == shell
            profile[shell] = dict(
                n=int(mask.sum()),
                mean_re=float(re[mask].mean()),
                weight_share=float(weight[mask].sum() / total_weight),
            )
        rms_shell = float(np.sqrt(float((weight * shell_arr ** 2).sum())
                                  / total_weight))
        frac_near = float(weight[shell_arr <= 1].sum() / total_weight)

    mean_re = float(re.mean())
    return dict(
        PR=pr,
        concentration=(1.0 / pr) if pr and pr > 0 else float("nan"),
        mean_re=mean_re,
        std_re=float(re.std()),
        std_over_mean=(float(re.std() / abs(mean_re)) if mean_re != 0.0
                       else float("inf")),
        shell_profile=profile,
        rms_shell_radius=rms_shell,
        frac_within_shell1=frac_near,
    )


def rm_table(mass, rad):
    """Every r·m combination — 3 mass definitions × 2 radius definitions — with
    the definitional spread stated FIRST (#451's lesson: r·m is too definition-
    sensitive to quote as one number; the claim is order-of-magnitude only)."""
    combos = {}
    for m_name in ("m_shell", "m_sum", "m_action"):
        for r_name in ("r_dual", "r_primal"):
            combos[f"{r_name} x {m_name}"] = rad[r_name] * mass[m_name]
    finite = [v for v in combos.values() if math.isfinite(v)]
    spread = ((min(finite), max(finite)) if finite
              else (float("nan"), float("nan")))
    return dict(spread_min=spread[0], spread_max=spread[1], combos=combos,
                physical=PHYSICAL_RM)


def measure(st, holes=(), label=""):
    """Read every geometric observable off one converged 4D spacetime: build the
    C++ skeleton, select the interior hinges, and return the readings dict.
    ``holes`` are the register holes' vertex-id tuples (e.g. ``Proton.quark_holes()``
    / ``ProtonIngredients.emergent_holes()``) — the BFS shell seeds."""
    build_skeleton(st)  # registers the lattice onto st itself; idempotent
    hinges, census = interior_hinges(st, holes)
    mass = masses(hinges)
    rad = radii(st, census)
    loc = localization(hinges)
    table = rm_table(mass, rad)
    return dict(label=label, census=census, mass=mass, radius=rad,
                localization=loc, rm=table, n_holes=len(holes), hinges=hinges)
