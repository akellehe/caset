# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The blessed read context for register observables (#583, part of #559).

``Register`` composes — never refactors — the C++ readers behind every
emergent-proton observable:

* **Skeleton in C++ only.** The ``ReggeSolver(st, MatterConfiguration())``
  constructor materializes the facet/coface lattice (tops → tetrahedra →
  triangle hinges); nothing is ever materialized from Python (the #451
  corruption lesson: a Python-driven materialization corrupted coface lists
  and ``dualVolume()`` saw half its cofaces).
* **Hole selection validated at one entry point.** ``register_holes(st,
  count)`` — a deficit RAISES with a clear message; a surplus WARNS naming the
  dropped holes (never a silent slice); both the used and the total census are
  recorded (``holes_used`` / ``holes_total``) alongside ``b3``, with the
  ``holes_vs_b3_divergent`` flag (the campaign taught us holes and b₃ can
  disagree — e.g. holes=3/b₃=2).
* **One orientation convention.** The induced-orientation signs ``ε_h = ±1``
  come from ``ChainComplex.endSignCovector`` — the label-free orientation
  under which every closed form's signed periods obey ``Σ_h ε_h p_h = 0``,
  determined up to one global sign (the propagation root, ``-1 ∈ U(1)`` on
  the register — gauge, not physics).
* **One cached ``EigenstateSynthesis``** at the register degree, plus
  lazily-built structures every observable may share: the top-cell list, the
  C++ dual adjacency (``Spacetime.getDualAdjacency``), and a generic
  ``derived(name, build)`` memo so heavier per-complex structures (e.g. the
  dual-edge cell frames of the transport readouts) are built once per
  Register when their observables land.

The GAUGE and RELABEL gates act on the Register, not on the observables:
``gauged(theta)`` rotates the register target by the surviving global U(1)
phase (which contains the Z₃ cyclic recolor of the singlet and the
orientation flip), and ``relabeled(seed)`` rebuilds the whole complex under a
random vertex-id permutation with the cell enumeration order shuffled too,
re-deriving the holes on the relabeled complex and matching the original
register's images by permuted vertex SET (vertex-order agnostic — nothing is
ever sorted to impose a convention).
"""
import cmath
import copy
import math
import warnings

import numpy as np

import tessera as _T

_cob = _T.cobordism

#: The primitive cube root of unity, `ω = exp(2πi/3)`.
OMEGA = cmath.exp(2j * math.pi / 3)

#: The color singlet `[1, ω, ω²]` — the carried register state (Σ = 0).
SINGLET = (1.0 + 0.0j, OMEGA, OMEGA * OMEGA)

#: Default GAUGE-gate angle: an incommensurate fraction of 2π, so the rotated
#: target never lands on a symmetry of the singlet by accident.
GAUGE_THETA = 2.0 * math.pi * 0.371

#: Default RELABEL-gate permutation seed (the pair-loop readout's choice).
GATE_SEED = 3


def build_complex(cells, edges, vertex_times=None, dimensions=None):
    """A live complex from explicit top cells + per-edge complex squared
    lengths, with the skeleton materialized in C++ (the ``ReggeSolver`` ctor).

    ``cells`` keep their intrinsic vertex order (never sorted — the stored
    order carries the orientation); ``edges`` maps ``(min_id, max_id)`` to the
    complex squared length; ``vertex_times`` (optional) maps vertex id to its
    recorded time, applied before the lengths. ``dimensions`` defaults to the
    first cell's dimension. Raises ``KeyError`` if an edge of the built
    complex has no recorded length — a partial metric is never silently
    defaulted.
    """
    cells = [list(c) for c in cells]
    if not cells:
        raise ValueError("build_complex needs at least one top cell")
    if dimensions is None:
        dimensions = len(cells[0]) - 1
    st = _T.Spacetime.fromCells(int(dimensions), cells, 1.0, 0.0)
    if vertex_times:
        vertex_list = st.getVertexList()
        for vid, t in dict(vertex_times).items():
            vertex_list.get(int(vid)).setTime(float(t))
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        e.setSquaredLength(edges[(a, b) if a < b else (b, a)])
    _T.ReggeSolver(st, _T.MatterConfiguration())
    return st


def register_holes(st, count=3, degree=3):
    """The structure's ``count`` register holes (removed top cells) in
    ``MultiCobordism.emergent_holes`` order — the single validated selection
    entry point (the #485 fixture convention with PR #577's semantics).

    A deficit raises; a surplus is an explicit, warned truncation NAMING the
    dropped holes, never a silent slice. Returns ``(holes, dropped)``.
    """
    holes = [tuple(h) for h in _cob.MultiCobordism.emergent_holes(st, degree)]
    if len(holes) < count:
        raise ValueError(
            f"need >= {count} register holes at degree {degree}, found "
            f"{len(holes)}{': ' + str(holes) if holes else ''} — this specimen "
            f"cannot host the requested register")
    if len(holes) > count:
        warnings.warn(
            f"register selection: {len(holes)} emergent holes; using the first "
            f"{count} (emergent_holes order), dropping {holes[count:]} — the "
            f"confinement constraint ranges over ALL holes, so the read covers "
            f"a sub-register", stacklevel=2)
    return holes[:count], holes[count:]


def induced_orientation_signs(st, holes):
    """The induced-orientation signs ``ε_h = ±1`` of the holes' boundary
    cycles relative to the complex's own coherent (facet-sharing) orientation
    (``ChainComplex.endSignCovector`` — lex-rooted, component-aware, and a
    hard error on any facet with more than two cofaces): the label-free
    orientation under which every closed form's signed periods obey
    ``Σ_h ε_h p_h = 0``. Determined up to one global sign (the propagation
    root) = ``-1 ∈ U(1)`` on the register."""
    tops = [[v.getId() for v in s.getVertices()] for s in st.getTopSimplices()]
    return list(_cob.ChainComplex.endSignCovector(
        tops, [list(h) for h in holes]))


class Register:
    """The shared per-complex read context every ``Observable`` measures.

    Parameters
    ----------
    st : Spacetime
        The converged complex to read. Its C++ skeleton is ensured here
        (``ReggeSolver`` ctor — idempotent).
    holes : optional list of vertex-id tuples
        An explicit register selection (e.g. from a build's own census); when
        omitted, ``register_holes(st, count, degree)`` selects and validates.
        A supplied list is validated with the same count semantics.
    count : int
        How many holes the register must carry: fewer raises, more warns
        naming the dropped ones. Pass the specimen's own hole count to read
        sub-3-hole specimens (the battery then skips per-observable with
        reasons instead of crashing here).
    degree : int
        The register degree k (holes are (k+2)-vertex removed top cells; the
        cached ``EigenstateSynthesis`` reads at this k). The record field
        ``b3`` is Betti at this degree (named for the k=3 default).
    target : sequence of complex
        The register target state, one component per hole convention slot —
        the color singlet ``[1, ω, ω²]`` by default. The GAUGE gate rotates
        exactly this.
    """

    def __init__(self, st, holes=None, count=3, degree=3, target=SINGLET):
        self.st = st
        self.degree = int(degree)
        self.target = tuple(complex(t) for t in target)
        # The skeleton, C++-side only (idempotent; the #451 lesson).
        _T.ReggeSolver(st, _T.MatterConfiguration())
        emergent = [tuple(h) for h in
                    _cob.MultiCobordism.emergent_holes(st, self.degree)]
        self.holes_total = len(emergent)
        if holes is None:
            self.holes, self.dropped = register_holes(st, count, self.degree)
        else:
            supplied = [tuple(h) for h in holes]
            if len(supplied) < count:
                raise ValueError(
                    f"need >= {count} register holes, got {len(supplied)}")
            if len(supplied) > count:
                warnings.warn(
                    f"register selection: given {len(supplied)} holes; using "
                    f"the first {count}, dropping {supplied[count:]}",
                    stacklevel=2)
            self.holes, self.dropped = supplied[:count], supplied[count:]
        self._derived = {}

    # ---- lazily-built shared structures -----------------------------------

    def derived(self, name, build):
        """Memoize ``build()`` under ``name`` — one construction per Register
        for anything heavier observables share (frames, facet indices,
        metric weights, ...). ``gauged`` copies SHARE this cache (the gauge
        knob only rotates the target), so target-dependent entries must key
        their name by target."""
        if name not in self._derived:
            self._derived[name] = build()
        return self._derived[name]

    @property
    def es(self):
        """The one cached ``EigenstateSynthesis(st, degree)`` every period
        readout shares."""
        return self.derived(
            "es", lambda: _cob.EigenstateSynthesis(self.st, self.degree))

    @property
    def eps(self):
        """Induced-orientation signs of ``holes`` (``endSignCovector``)."""
        return self.derived(
            "eps", lambda: induced_orientation_signs(self.st, self.holes))

    @property
    def cells(self):
        """The top-cell ``Simplex`` list, in enumeration order."""
        return self.derived("cells", lambda: list(self.st.getTopSimplices()))

    @property
    def adjacency(self):
        """The C++ dual adjacency as ``{cell_index: [neighbor indices]}``
        over ``cells`` (``Spacetime.getDualAdjacency``), validated against
        the top-cell count."""
        def build():
            rows, cols, n = self.st.getDualAdjacency()
            if n != len(self.cells):
                raise RuntimeError(
                    f"dual adjacency count {n} != top cells {len(self.cells)}")
            adjacency = {}
            for r, c in zip(rows, cols):
                adjacency.setdefault(int(r), []).append(int(c))
            return adjacency
        return self.derived("adjacency", build)

    @property
    def betti(self):
        """The full Betti vector of the complex (GF(2) ranks)."""
        return self.derived(
            "betti", lambda: [int(b) for b in _cob.MultiCobordism.betti(self.st)])

    @property
    def b3(self):
        """Betti at the register degree (b₃ for the k=3 default)."""
        betti = self.betti
        return betti[self.degree] if len(betti) > self.degree else 0

    @property
    def holes_vs_b3_divergent(self):
        """True when the emergent-hole census and Betti at the register
        degree disagree — the campaign's holes=3/b₃=2 style finding, always
        recorded, never papered over."""
        return self.holes_total != self.b3

    @property
    def dimensions(self):
        """The top-cell dimension, or None when top cells are mixed-size
        (declarative ``requires={"dimensions": d}`` gates read this)."""
        def build():
            sizes = {len(c.getVertices()) for c in self.cells}
            return next(iter(sizes)) - 1 if len(sizes) == 1 else None
        return self.derived("dimensions", build)

    @property
    def causal_content(self):
        """True iff any edge is non-spacelike (timelike or null) — the
        ``needs_causal_content`` skip path reads this. At initialization no
        time has passed; causal structure may only emerge, so all-spacelike
        specimens honestly report False."""
        def build():
            return bool(any((not e.isSpacelike())
                            for e in self.st.getEdgeList().toVector()))
        return self.derived("causal_content", build)

    def summary(self):
        """The register block of every battery record: the hole census, the
        Betti census, and the divergence flag (JSON-able)."""
        return {
            "degree": self.degree,
            "dimensions": self.dimensions,
            "n_top_cells": len(self.cells),
            "holes_used": len(self.holes),
            "holes_total": self.holes_total,
            "dropped_holes": [list(h) for h in self.dropped],
            "b3": self.b3,
            "betti": self.betti,
            "holes_vs_b3_divergent": self.holes_vs_b3_divergent,
            "causal_content": self.causal_content,
        }

    # ---- the gate transforms ----------------------------------------------

    def gauged(self, theta=GAUGE_THETA):
        """The GAUGE-gate variant: the same complex and register with the
        target rotated by the global U(1) phase ``e^{iθ}`` (the register's one
        surviving gauge freedom — it contains the Z₃ cyclic recolor of the
        singlet and the orientation flip). Shares this Register's caches (the
        complex is untouched)."""
        rotated = copy.copy(self)
        rotated.target = tuple(cmath.exp(1j * theta) * t for t in self.target)
        return rotated

    def relabeled(self, seed=GATE_SEED):
        """The RELABEL-gate variant: rebuild the whole complex under a random
        vertex-id permutation with the cell enumeration order shuffled too
        (catching enumeration-order dependence), carry the metric (complex
        squared lengths) and vertex times across, re-derive the emergent
        holes on the relabeled complex, and match this register's images
        among them by permuted vertex SET (a missing image is an error — the
        gate must compare like with like). Returns ``(register, perm)``."""
        rng = np.random.default_rng(seed)
        cells = [[v.getId() for v in c.getVertices()] for c in self.cells]
        edges = {}
        for e in self.st.getEdgeList().toVector():
            a, b = e.getSource().getId(), e.getTarget().getId()
            edges[(a, b) if a < b else (b, a)] = e.getSquaredLength()
        times = {}
        for cell in cells:
            for v in cell:
                times.setdefault(v, None)
        vertex_list = self.st.getVertexList()
        for v in times:
            times[v] = float(vertex_list.get(v).getTime())

        all_vertices = sorted({v for c in cells for v in c})
        shuffled = list(all_vertices)
        rng.shuffle(shuffled)
        perm = dict(zip(all_vertices, shuffled))
        permuted_cells = [[perm[v] for v in c] for c in cells]
        order = rng.permutation(len(permuted_cells))
        st2 = build_complex(
            [permuted_cells[i] for i in order],
            {tuple(sorted((perm[a], perm[b]))): l2
             for (a, b), l2 in edges.items()},
            vertex_times={perm[v]: t for v, t in times.items()},
            dimensions=self.dimensions)

        rederived = [tuple(h) for h in
                     _cob.MultiCobordism.emergent_holes(st2, self.degree)]
        found = {frozenset(h): h for h in rederived}
        holes2 = []
        for h in self.holes:
            image = frozenset(perm[v] for v in h)
            if image not in found:
                raise ValueError(
                    f"emergent_holes on the relabeled complex is missing the "
                    f"image of hole {h}")
            holes2.append(found[image])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            register = Register(st2, holes=holes2, count=len(holes2),
                                degree=self.degree, target=self.target)
        return register, perm
