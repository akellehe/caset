# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Animate unforced emergence and the certificates the whitepaper names.

ONE documented command drives the unmodified joint Regge-Hodge stationarity
objective on a neutral complex and, after every engine unit, draws what the
paper's certificates actually read off the accepted geometry.

What is driven, and what is only read
-------------------------------------
The dynamics is `MultiCobordism` in `SimulationMode.EMERGENCE` /
`EmergenceSubmode.STRICT` with `JointStationarityObjective`.
Nothing this file computes enters that objective: the firewall is structural
(a static `objectiveOf` over declared scalars), and every panel below is a
read-only measurement over the accepted geometry through the library's own
observable classes. No target is pinned, no register is forced, no residual
against a prescribed carrier is scored.

The paper's ontology, and what this driver refuses to draw
----------------------------------------------------------
A quark is sought as a PERSISTENT MODULAR SPECTRAL CLUSTER carrying an
anchored rank-three band -- never as a hole. Betti numbers are drawn, because
the paper keeps them as independent topological observables, but they are
never a quark count and never a success condition. There is no color-register
count, no `{1, omega, omega^2}` target and no singlet residual anywhere in
this file, because the construction those belong to is not the one the paper
specifies.

The panels, and the class that feeds each
-----------------------------------------
1.  objective trace                 -- the node's own objective terms
2.  drawing layout of the complex   -- persisted 1-skeleton (layout only)
3.  persistent modular clusters     -- `PersistentModularity`
4.  fiber rank / gap / localization -- `SpectralFiber`, `SpectralFiberTracker`
5.  anchor profile                  -- `ColorAnchor` (score, max term,
                                       participation ratio, phase dispersion)
6.  transports and holonomy         -- `FiberConnection`
7.  exchange and rotation           -- `ExchangeHolonomy`
8.  crossing readouts               -- `CrossingReadouts` (sign of Re pi_perp
                                       per crossing, crossing mass, the
                                       one-third baryon sum, charge power)
9.  <J^2> and Var(J^2)              -- `CovarianceState` Wick reads
10. Betti numbers                   -- `Spacetime` (topological observable)
11. verdict and named reasons       -- `ParticleClusters.classifyBaryon`

Refusals are first class. On accessible hosts most channels are absent, and an
absent panel states WHAT is absent and WHY rather than drawing a blank or a
zero. A zero is a measurement; an absence is not, and the two are never
conflated.

Running it
----------
Cap parallelism; this box may be shared::

    OMP_NUM_THREADS=8 .venv-build/bin/python \\
      examples/cobordism/emergence_animation.py run --size 6 --steps 6 \\
      --out emergence.gif

``--out`` accepts ``.gif`` (pillow) or ``.mp4`` (ffmpeg); ``.png`` renders the
final frame alone. ``--json`` additionally writes the per-frame measurements,
so a panel can be checked against a number.

``--edge-disposition`` chooses the seed's causal character: ``random`` (the
default, magnitude one with the real/imaginary split drawn per edge),
``spacelike`` (``l^2 = +1``), ``timelike`` (``l^2 = -1``), or ``foliated`` (a
PRESCRIBED light cone -- timelike between hop layers of M0, spacelike within
one). Only ``foliated`` prescribes a causal order; it is labelled as such
wherever it is reported and is never presented as emergent.
"""

import argparse
import cmath
import itertools
import json
import math
import os
import random
import sys

import tessera as T

cob = T.cobordism
obs = T.observables
qu = T.quantum
MC = cob.MultiCobordism


# =====================================================================
# declared parameters -- fixed before any datum is examined
# =====================================================================

#: Stellar Pachner adds applied to the seed 4-ball.
DECLARED_SIZE = 6
#: The declared name of the incoming boundary region. Named once here and
#: passed as this constant, never spelled at a call site: `regionHandle`
#: raises by name on an undeclared region, so a mis-spelling that reached
#: the engine would refuse rather than silently scope to the whole complex.
M0_REGION = "m0"
#: Engine units to drive. One unit is one stage-1 update plus one stage-2
#: relaxation -- the engine's deterministic unit (#579).
DECLARED_STEPS = 6
#: Candidate moves offered to stage 1 per unit.
DECLARED_CANDIDATE_MOVES = 6
#: Stage-1 combinatorial updates per engine unit.
DECLARED_STAGE1_ITERS = 1
#: Stage-2 relaxation iterations per unit.
DECLARED_STAGE2_ITERS = 12
#: How many moves deep stage 1 searches for an objective-lowering SEQUENCE.
#:
#: When a batch of single moves finds no improvement the search deepens
#: iteratively -- 2-move sequences, then 3, up to this many -- committing an
#: F-lowering sequence as a whole. One means single moves only.
#:
#: The deepening covers EVERY move kind the stage-1 draw offers: the four
#: Pachner moves plus the cone-outs and cone-ins, not the surgical moves
#: alone. The depth is over the whole draw, not over a subset of it.
DECLARED_SURGICAL_DEPTH = 1
#: Absolute objective tolerance. Two roles, both absolute and never relative.
#:
#: Stage 2 backs its line search off until a trial lowers the exact selected
#: objective by at least this much; and the drive stops once a whole engine
#: unit fails to improve the objective by it. A run that stops that way says
#: so -- the terminator is recorded, never inferred from a short trace.
DECLARED_TOLERANCE = 1e-12
#: Objective register degrees.
DECLARED_REGISTER_DEGREES = (1,)
#: Laplacian degrees the Hodge entropy term is scored at.
#:
#: Declared independently of the register degrees, which answer the unrelated
#: question of where a register is constructed. All four carry distinct
#: information here: the discrete weighted operator does not inherit the
#: continuum Hodge duality that would make k and 4-k the same condition counted
#: twice on a closed 4-manifold.
#:
#: Scoring more degrees makes the objective see more of the SPECTRUM, not more
#: of the TOPOLOGY. Exact zero modes are omitted from the entropy and from its
#: derivative, so each degree is blind to a change in its own kernel dimension.
DECLARED_HODGE_DEGREES = (0, 1, 2, 3)
#: Post-hoc analysis degrees. Separate from the objective's domain.
DECLARED_ANALYSIS_DEGREES = (1,)
#: Modularity resolution the clusters are read at.
DECLARED_RESOLUTION = 1.0
#: Node seed and host seed.
DECLARED_SEED = 7
DECLARED_HOST_SEED = 3
#: Degrees the Betti numbers are reported at.
DECLARED_BETTI_DEGREES = (0, 1, 2)


class EdgeDisposition:
    """The causal character the seed's edges are given, as a closed vocabulary.

    `Edge` stores the length `l`, not the squared length, so a disposition is
    written through `l` and read as `l^2`. A purely imaginary length squares
    to a real negative, which IS the timelike condition, so the whole
    vocabulary is expressible as a choice of `arg l`.

    Named constants rather than bare strings: each value is written where the
    host is built and compared where the seed is assigned, and a typo in
    either place would not fail to compile -- it would silently select a
    different causal structure.
    """

    #: Magnitude one, real/imaginary split uniformly at random per edge:
    #: `l = e^{i a}`, so `l^2 = e^{2 i a}` sweeps the unit circle. The only
    #: setting that prescribes no causal structure, and the default.
    RANDOM = "random"
    #: `l = 1`, so `l^2 = +1` on every edge.
    SPACELIKE = "spacelike"
    #: `l = i`, so `l^2 = -1` on every edge.
    TIMELIKE = "timelike"
    #: Timelike BETWEEN hop layers of M0, spacelike WITHIN a layer. A
    #: PRESCRIBED foliation: it imposes a causal order rather than letting one
    #: emerge, and must be reported as such wherever it is used.
    FOLIATED = "foliated"

    #: Every accepted value, in help order.
    ALL = (RANDOM, SPACELIKE, TIMELIKE, FOLIATED)


#: The seed disposition when the caller names none.
DECLARED_EDGE_DISPOSITION = EdgeDisposition.RANDOM


class Terminator:
    """Why a drive stopped, as a closed vocabulary.

    A short trace is ambiguous on its face: it looks the same whether the run
    exhausted its units or converged. The terminator says which, so a reader
    never has to infer it from the frame count.

    Named constants rather than bare strings: the value is written where the
    loop exits and compared where a document is read, and a typo in either
    place would produce a run that reports a terminator nothing matches.
    """

    #: The unit budget ran out. The run has not converged; it stopped.
    STEPS = "steps-exhausted"
    #: A whole engine unit failed to improve the objective by the declared
    #: absolute tolerance. The run stopped early and says so.
    TOLERANCE = "tolerance-reached"

    #: Every value a drive may report.
    ALL = (STEPS, TOLERANCE)


class DriveResult:
    """A drive's frames together with WHY it stopped.

    The terminator belongs to the loop, not to any one frame: a frame is a
    measurement of the complex at a unit, while the terminator is a fact
    about the run that produced them. Returning them together keeps a frame
    from carrying a field that is meaningless for every frame but the last.
    """

    __slots__ = ("frames", "terminator")

    def __init__(self, frames, terminator):
        if terminator not in Terminator.ALL:
            raise ValueError(
                "unknown terminator %r: expected one of %s"
                % (terminator, ", ".join(Terminator.ALL)))
        self.frames = frames
        self.terminator = terminator

    def __len__(self):
        return len(self.frames)


class CausalClass:
    """How one edge's `l^2` reads causally, as a closed vocabulary.

    Under `spacelike` and `timelike` every `l^2` sits exactly on the real
    axis, but under `random` -- the default -- `l^2 = e^{2 i a}` sweeps the
    whole unit circle, so most edges are NEITHER. A scheme that reported only
    two classes would have to bucket a genuinely complex `l^2` into one of
    them and claim a definiteness the geometry does not have.
    """

    #: `l^2` real and positive.
    SPACELIKE = "spacelike"
    #: `l^2` real and negative.
    TIMELIKE = "timelike"
    #: `l^2` real and zero -- on the light cone.
    NULL = "null"
    #: `l^2` off the real axis. Not a causal type at all: the edge has no
    #: definite character to report, and saying so is the honest reading.
    INDEFINITE = "indefinite"

    ALL = (SPACELIKE, TIMELIKE, NULL, INDEFINITE)


#: `|Im l^2|` at or below this, relative to `|l^2|`, reads as ON the real
#: axis. Above it the edge is `INDEFINITE` and is drawn as such.
DECLARED_CAUSAL_REAL_TOLERANCE = 1e-9
#: `|l^2|` at or below this reads as null rather than as a tiny spacelike or
#: timelike value.
DECLARED_CAUSAL_NULL_TOLERANCE = 1e-12

#: One colour per causal class, and the legend reads from this map, so the
#: drawn colour and its label can never disagree.
DECLARED_CAUSAL_COLOURS = {
    CausalClass.SPACELIKE: "#1f4e79",
    CausalClass.TIMELIKE: "#a33227",
    CausalClass.NULL: "#c9a227",
    CausalClass.INDEFINITE: "#7a5aa8",
}

#: Stabilization of the drawing layout. Classical MDS is defined only up to
#: rotation, reflection and scale and is globally sensitive, so a small change
#: to the complex reshuffles the whole cloud. Scale is already fixed by the
#: RMS normalization the layout read performs; these three remove the
#: orientation ambiguity, ease the positions, and ease the view.
DECLARED_LAYOUT_EASE = 0.3
DECLARED_LAYOUT_VIEW_EASE = 0.25
DECLARED_LAYOUT_PAD = 0.18

#: The two dual-curvature channels. Diverging maps, because both channels are
#: SIGNED and centred at zero -- a sequential map would make a saddle
#: indistinguishable from a peak.
DECLARED_HEAT_CMAP_SPATIAL = "coolwarm"
DECLARED_HEAT_CMAP_TEMPORAL = "PuOr"
#: Clip the heat range at this percentile of |curvature| so one extreme cell
#: cannot flatten the rest of the field to a single colour.
DECLARED_HEAT_CLIP_PERCENTILE = 95


def causal_class(squared_length):
    """The `CausalClass` of one edge, read from its `l^2`.

    Definiteness is judged RELATIVE to the magnitude, so the classification
    does not depend on the overall scale the lengths happen to carry.
    """
    value = complex(squared_length)
    magnitude = abs(value)
    if magnitude <= DECLARED_CAUSAL_NULL_TOLERANCE:
        return CausalClass.NULL
    if abs(value.imag) > DECLARED_CAUSAL_REAL_TOLERANCE * magnitude:
        return CausalClass.INDEFINITE
    return (CausalClass.SPACELIKE if value.real > 0.0
            else CausalClass.TIMELIKE)


# =====================================================================
# the neutral host -- a cobordism, because the readouts need a boundary
# =====================================================================

def boundary_vertices(spacetime):
    """The vertices of the incoming boundary M0, or [] if the complex is closed.

    A boundary facet is a (d-1)-simplex with exactly one coface. This is the
    same rule the crossing panel applies, kept in one place so the host and
    the readout cannot disagree about what M0 is.

    The facets are reached through the TOP cells rather than through
    `getSimplices()`: the lower skeleton is not materialized as registered
    simplices until something builds it, so `getSimplices()` on a fresh
    complex returns the top cells alone and would report every complex as
    closed.

    TWO passes, and the order matters. `getFacets()` REGISTERS the coface
    relation as a side effect, so a facet's coface count is only complete
    once every top cell has been visited. Counting in the same pass that
    materializes would see each facet before its second cell had registered
    and report the whole complex as boundary.
    """
    facets = {}
    for top in spacetime.getTopSimplices():
        for facet in top.getFacets():
            facets[facet.__hash__()] = facet
    boundary = set()
    for facet in facets.values():
        if len(facet.getCofaces()) == 1:
            for vertex in facet.getVertices():
                boundary.add(int(vertex.getId()))
    return sorted(boundary)


def _edge_endpoints(edge):
    """An edge's two vertex ids, through the bound accessors.

    `Edge` exposes its endpoints as vertices, not as a key pair, so an id is
    read through `getSource`/`getTarget` rather than by indexing.
    """
    return int(edge.getSource().getId()), int(edge.getTarget().getId())


def _hop_layers(spacetime, sources):
    """Hop distance from `sources` over the 1-skeleton, per vertex id.

    This is the SAME layering `CrossingReadouts::temporalFunction` derives
    from M0, recomputed here only to assign the causal character consistently
    with it. Assigning by any other partition would put a causal edge inside
    a layer, which the temporal-function certificate names and refuses.
    """
    layer = {vertex: 0 for vertex in sources}
    frontier = list(sources)
    adjacency = {}
    for edge in spacetime.getEdgeList().toVector():
        a, b = _edge_endpoints(edge)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    depth = 0
    while frontier:
        depth += 1
        nxt = []
        for vertex in frontier:
            for neighbour in adjacency.get(vertex, ()):
                if neighbour not in layer:
                    layer[neighbour] = depth
                    nxt.append(neighbour)
        frontier = nxt
    return layer


def _seed_lengths(spacetime, disposition, seed):
    """Write the seed length on every edge, per the chosen disposition.

    Every setting carries magnitude one, so the dispositions differ ONLY in
    `arg l` -- in causal character, never in scale. Reproducible from `seed`
    through a private generator, so the global random state is untouched.
    """
    if disposition not in EdgeDisposition.ALL:
        raise ValueError(
            "unknown edge disposition %r: expected one of %s"
            % (disposition, ", ".join(EdgeDisposition.ALL)))
    edges = spacetime.getEdgeList().toVector()
    if disposition == EdgeDisposition.SPACELIKE:
        for edge in edges:
            edge.setLength(complex(1.0, 0.0))            # l^2 = +1
        return
    if disposition == EdgeDisposition.TIMELIKE:
        for edge in edges:
            edge.setLength(complex(0.0, 1.0))            # l^2 = -1
        return
    if disposition == EdgeDisposition.RANDOM:
        generator = random.Random(seed)
        for edge in edges:
            angle = generator.uniform(0.0, 2.0 * math.pi)
            edge.setLength(cmath.exp(1j * angle))        # |l| = 1
        return
    # FOLIATED: the causal character follows the hop layering of M0, which is
    # the same layering the temporal function derives. Edges spanning layers
    # are timelike and edges inside one spacelike -- a prescribed light cone.
    layer = _hop_layers(spacetime, boundary_vertices(spacetime))
    for edge in edges:
        a, b = _edge_endpoints(edge)
        spans_layers = layer.get(a, 0) != layer.get(b, 0)
        edge.setLength(complex(0.0, 1.0) if spans_layers
                       else complex(1.0, 0.0))


def build_cobordism_host(n_refine=DECLARED_SIZE, seed=DECLARED_HOST_SEED,
                         disposition=DECLARED_EDGE_DISPOSITION):
    """A single simplex, refined -- the canonical seed.

    The paper's crossing readouts live on a cobordism: `tau` is the Lorentzian
    distance FROM the incoming boundary M0, and the surfaces are its level
    sets. A CLOSED complex has no such surface, so those readouts cannot run
    on one at any size. A single 4-simplex is a 4-BALL, so it HAS a boundary
    -- `S^3 = M0` -- structurally, rather than by carving one out of a closed
    manifold. The whitepaper prescribes no host topology; it specifies
    cobordisms with `∂W = M0 ⊔ M1`, and this is the smallest complex that is
    one.

    The seed's causal character is `disposition`, an `EdgeDisposition`. Every
    setting carries magnitude one, so they differ only in `arg l` -- in causal
    character, never in scale. The refinement runs on a uniform real unit
    length REGARDLESS of the disposition, so the topology a given `seed`
    produces is the same under all four and the settings are comparable as a
    controlled variable; the disposition is written afterwards, over the
    finished complex.

    NEUTRAL otherwise: no holes, no pinned carrier, no boundary blocks, no
    target register. Whatever the run comes to carry is read afterwards.
    `foliated` is the exception and is not neutral: it prescribes a causal
    order rather than letting one emerge, and is labelled as such.
    """
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT,
                     1.0, 1.0, T.PREFERRED, T.SolidSimplex(4))
    st.build()
    for edge in st.getEdgeList().toVector():
        edge.setLength(complex(1.0, 0.0))
    applied = 0
    for step in range(seed, seed + n_refine * 4):
        move = T.AddMove(st, step, False, T.PachnerMode.PreGeometric, False)
        if move.propose() and move.apply():
            applied += 1
        if applied >= n_refine:
            break
    _seed_lengths(st, disposition, seed)
    return st


# =====================================================================
# small helpers -- unknown is None, never zero
# =====================================================================

def _finite(value):
    """A float, or None when the value is not a finite measurement."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _reasons(read, attribute="failedCertificates"):
    """The NAMED reasons a read refused, as a list of strings."""
    named = getattr(read, attribute, None)
    if not named:
        return []
    return [str(reason) for reason in named]


class Absent:
    """A channel that has no measurement, carrying WHY.

    Not a zero and not an empty list: a statement that the measurement does
    not exist, with the reason the paper's certificate names.
    """

    __slots__ = ("reason",)

    def __init__(self, reason):
        self.reason = str(reason)

    def __repr__(self):                                   # pragma: no cover
        return "Absent(%r)" % self.reason

    def to_json(self):
        return {"absent": True, "reason": self.reason}


def _json_safe(value):
    """Recursively convert a measurement block to JSON-safe values."""
    if isinstance(value, Absent):
        return value.to_json()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# =====================================================================
# the read -- every panel's measurement, all read-only
# =====================================================================

class EmergenceFrame:
    """Every measurement drawn on one frame, assembled over one geometry.

    Read-only: nothing here touches the geometry or the objective. Each
    channel is either a measurement or an `Absent` carrying the named reason,
    so a panel can always say what it is showing.
    """

    def __init__(self, node, spacetime, step, config):
        self.step = step
        self.config = config
        self.spacetime = spacetime
        self.objective = self._read_objective(node)
        self.layout = self._read_layout(spacetime)
        # Drawing-only, like `layout`: neither appears in `to_json`, so the
        # record is unchanged by anything the figure needs.
        self.dual = self._read_dual_curvature(spacetime)
        self.clusters = self._read_clusters(spacetime, config)
        self.bands = self._read_bands(spacetime, config)
        self.states = self._build_states()
        self.anchors = self._read_anchors()
        self.transports = self._read_transports(spacetime)
        self.statistics = self._read_statistics(spacetime)
        self.crossings = self._read_crossings(spacetime)
        self.spin = self._read_spin()
        self.betti = self._read_betti(spacetime, config)
        self.verdict = self._read_verdict()

    # ---- 1. the objective -------------------------------------------

    @staticmethod
    def _read_objective(node):
        terms = node.objective_terms()
        block = {"total": _finite(node.objective())}
        for name in MC.objective_term_names():
            block[name] = _finite(getattr(terms, name, None))
        # Which DEGREE the Hodge share came from, not only the total. The
        # unweighted norm is carried alongside the weighted share so the raw
        # spread across degrees stays visible rather than folded into the
        # weighting.
        block["hodge_by_degree"] = [
            {"degree": contribution.degree,
             "weight": _finite(contribution.weight),
             "gradient_norm_squared":
                 _finite(contribution.gradient_norm_squared),
             "contribution": _finite(contribution.contribution)}
            for contribution in node.hodge_degree_contributions]
        return block

    # ---- 2. the drawing layout --------------------------------------

    @staticmethod
    def _read_layout(spacetime):
        """2-D coordinates per vertex for DRAWING ONLY.

        Classical multidimensional scaling on graph shortest paths under
        |l^2|^(1/2). This is a picture, not a spacetime coordinate system:
        it carries no causal content and no position here means anything
        physical.
        """
        try:
            import numpy as np
            from scipy.sparse.csgraph import shortest_path
        except ImportError:
            return Absent("numpy/scipy unavailable: no drawing layout")
        edges = spacetime.getEdgeList().toVector()
        vertices = sorted({int(v.getId())
                           for edge in edges
                           for v in (edge.getSource(), edge.getTarget())})
        if len(vertices) < 2:
            return Absent("fewer than two vertices: nothing to lay out")
        index = {v: i for i, v in enumerate(vertices)}
        n = len(vertices)
        weights = np.full((n, n), np.inf)
        np.fill_diagonal(weights, 0.0)
        pairs = []
        squared = []
        for edge in edges:
            a = index.get(int(edge.getSource().getId()))
            b = index.get(int(edge.getTarget().getId()))
            if a is None or b is None:
                continue
            length = complex(edge.getLength())
            w = math.sqrt(max(abs(length ** 2), 1e-6))
            weights[a, b] = weights[b, a] = min(weights[a, b], w)
            pairs.append((a, b))
            # Carried per drawn edge so the causal colouring reads the same
            # `l^2` the geometry holds, rather than re-deriving it from a
            # disposition setting that only describes how the SEED was built.
            squared.append(length ** 2)
        distances = shortest_path(weights, method="D", directed=False)
        finite = np.isfinite(distances)
        if not finite.any():
            return Absent("no finite graph distance: complex is disconnected")
        distances[~finite] = distances[finite].max() * 1.5
        squared_distances = distances ** 2
        centering = np.eye(n) - np.ones((n, n)) / n
        gram = -0.5 * centering @ squared_distances @ centering
        values, vectors = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1][:2]
        coords = vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))
        coords = coords - coords.mean(0)
        rms = math.sqrt((coords ** 2).sum(1).mean()) or 1.0
        return {"coords": {vertices[i]: tuple(coords[i] / rms)
                           for i in range(n)},
                "edges": pairs,
                "edge_squared_lengths": squared,
                "edge_causal_classes": [causal_class(z) for z in squared],
                "vertices": vertices}

    # ---- 2b. dual curvature, for the two heat panels ----------------

    @staticmethod
    def _read_dual_curvature(spacetime):
        """Per-top-cell Regge curvature, both channels, for DRAWING ONLY.

        Curvature in Regge calculus lives on the hinges -- the `(d-2)`
        simplices, which in 4D are the triangles. Each is weighted by its own
        dual measure and summed onto the top cells that contain it, giving one
        signed value per dual node.

        The Lorentzian deficit is COMPLEX and the two parts are different
        physics, so they are kept apart rather than collapsed to a magnitude:
        `Re eps * |star|` is the rotation angle-defect carried by timelike
        hinges, and `Im eps * |star|` is the boost / light-cone content
        carried by spacelike hinges -- those whose normal plane is timelike.
        Both keep their sign, so a saddle stays distinguishable from a peak.

        Like the drawing layout, this is computed for the figure and is NOT
        part of the record: it appears in no `to_json` block.
        """
        try:
            import numpy as np  # noqa: F401  (parity with the layout read)
        except ImportError:
            return Absent("numpy unavailable: no dual curvature")
        hinge = {}
        for simplex in spacetime.getSimplices():
            vertices = simplex.getVertices()
            if len(vertices) != 3:
                continue
            key = tuple(sorted(int(v.getId()) for v in vertices))
            try:
                deficit = complex(simplex.deficitAngle())
                weight = abs(complex(simplex.dualVolume()))
            except RuntimeError:
                # A boundary or degenerate hinge carries no deficit. Only that
                # geometric failure is absorbed -- a contract failure
                # (TypeError, ValueError) must propagate rather than render as
                # zero curvature, which would be indistinguishable from a flat
                # hinge.
                continue
            hinge[key] = (deficit.real * weight, deficit.imag * weight)
        cells = []
        for cell in spacetime.getTopSimplices():
            ids = sorted(int(v.getId()) for v in cell.getVertices())
            faces = [tuple(t) for t in itertools.combinations(ids, 3)]
            cells.append({
                "vertices": ids,
                "spatial": sum(hinge.get(f, (0.0, 0.0))[0] for f in faces),
                "temporal": sum(hinge.get(f, (0.0, 0.0))[1] for f in faces),
            })
        if not cells:
            return Absent("no top cells: nothing to draw a dual over")
        rows, cols, _count = spacetime.getDualAdjacency()
        return {"cells": cells,
                "adjacency": list(zip(list(rows), list(cols))),
                "hinges_with_curvature": len(hinge)}

    # ---- 3. persistent modular clusters -----------------------------

    def _read_clusters(self, spacetime, config):
        modularity = obs.PersistentModularity.fromSpacetime(spacetime)
        settings = obs.PersistentModularityConfig()
        settings.resolutions = [config["resolution"]]
        settings.baseSeed = config["seed"]
        report = modularity.scanResolutions(settings)
        if not report.slices:
            self.components = []
            return Absent("modularity returned no resolution slice")
        self.slice = report.slices[0]
        self.components = list(self.slice.components)
        if not self.components:
            return Absent("no persistent cluster at the analysis resolution")
        sizes = [len(list(c.support)) for c in self.components]
        return {"count": len(self.components),
                "sizes": sizes,
                "resolution": config["resolution"],
                "modularity": _finite(getattr(self.slice, "modularity", None))}

    # ---- 4. fibers: rank, gap, localization -------------------------

    def _read_bands(self, spacetime, config):
        self.candidates = []
        if not self.components:
            return Absent("no cluster to carry a band")
        settings = obs.SpectralFiberConfig()
        settings.degrees = list(config["degrees"])
        tracker = obs.SpectralFiberTracker(spacetime, settings)
        rows = []
        for component in self.components:
            for degree in config["degrees"]:
                try:
                    read = tracker.enumerateBands(component.support, degree)
                except Exception as error:                # noqa: BLE001
                    rows.append({"accepted": False,
                                 "reason": "band enumeration failed: %s"
                                           % error})
                    continue
                chosen = None
                for fiber in read.fibers:
                    if fiber.accepted():
                        chosen = fiber
                        break
                self.candidates.append(chosen)
                if chosen is None:
                    named = []
                    for fiber in read.fibers:
                        named.extend(_reasons(fiber.certificate(),
                                              "failedCertificates"))
                    rows.append({"accepted": False,
                                 "offered": len(read.fibers),
                                 "reason": ", ".join(sorted(set(named)))
                                           or "no band met the certificate"})
                    continue
                certificate = chosen.certificate()
                rows.append({
                    "accepted": True,
                    "degree": int(chosen.degree()),
                    "rank": int(chosen.rank()),
                    "lowerGap": _finite(certificate.lowerGap),
                    "upperGap": _finite(certificate.upperGap),
                    "localization": _finite(certificate.localization),
                    "localizationExcess": _finite(
                        certificate.localizationExcess),
                    "gramDefect": _finite(certificate.gramDefect),
                })
        if not rows:
            return Absent("no band read was attempted")
        return {"rows": rows,
                "accepted": sum(1 for r in rows if r.get("accepted"))}

    def _build_states(self):
        """The pure Slater covariance of each accepted band projector.

        A band whose restricted metric fails positivity supplies no quasi-free
        covariance; that band's slot stays None rather than becoming a zero.
        """
        states = []
        for fiber in self.candidates:
            if fiber is None:
                states.append(None)
                continue
            try:
                states.append(
                    qu.CovarianceState.fromBandProjector(fiber.projector()))
            except Exception:                             # noqa: BLE001
                states.append(None)
        return states

    # ---- 5. the anchor profile --------------------------------------

    def _read_anchors(self):
        """The quark reads, which carry the triangle-anchor profile.

        `classifyQuark` is the library's own evaluation of the paper's quark
        conditions, and its read reports the anchor score, maximal term,
        participation ratio and determinant-phase dispersion the paper asks
        for -- together with every certificate that failed, NAMED.
        """
        self.quarks = []
        accepted = [f for f in self.candidates if f is not None]
        if not accepted:
            return Absent("no accepted band: nothing to anchor")
        classifier = obs.ParticleClusters()
        rows = []
        for index, fiber in enumerate(self.candidates):
            if fiber is None:
                continue
            evidence = obs.QuarkCandidateEvidence()
            evidence.colorBand = fiber
            if index < len(self.components):
                evidence.component = self.components[index].id
            state = self.states[index] if index < len(self.states) else None
            if state is not None:
                evidence.parityRead = state.wickParity()
                evidence.occupationRead = state.wickTotalNumber()
            # This overlay reads ONE cobordism frame, so the frame lifetime is
            # one and its adjacent-frame overlap is vacuously one. Both are
            # MEASURED facts about this read.
            evidence.frameLifetime = 1.0
            evidence.frameMinOverlap = 1.0
            try:
                read = classifier.classifyQuark(evidence)
            except Exception as error:                    # noqa: BLE001
                rows.append({"reason": "quark read failed: %s" % error})
                continue
            self.quarks.append(read)
            rows.append({
                "classification": str(read.classification),
                "colorRank": int(read.colorRank),
                "score": _finite(read.triangleAnchorScore),
                "maxTerm": _finite(read.triangleAnchorMaxTerm),
                "participationRatio": _finite(
                    read.triangleAnchorParticipation),
                "phaseDispersion": _finite(read.anchorPhaseDispersion),
                "phaseCoherence": _finite(read.anchorPhaseCoherence),
                "reasons": _reasons(read),
            })
        if not rows:
            return Absent("no band produced a quark read")
        return {"rows": rows,
                "certified": sum(1 for r in rows
                                 if r.get("classification") == "quark")}

    # ---- 6. transports, leakage, holonomy ---------------------------

    def _read_transports(self, spacetime):
        accepted = [f for f in self.candidates if f is not None]
        if len(accepted) < 2:
            return Absent("fewer than two accepted bands: no ordered pair to "
                          "transport between")
        connection = obs.FiberConnection()
        rows = []
        for i, to_fiber in enumerate(accepted):
            for j, from_fiber in enumerate(accepted):
                if i == j:
                    continue
                if to_fiber.degree() != from_fiber.degree():
                    continue
                if to_fiber.rank() != from_fiber.rank():
                    continue
                try:
                    read = connection.transportOnSpacetime(
                        spacetime, to_fiber, from_fiber)
                except Exception:                         # noqa: BLE001
                    continue
                reason = str(getattr(read, "rejectionReason", "") or "")
                rows.append({
                    "accepted": bool(read.accepted),
                    "leakage": _finite(getattr(read, "leakage", None)),
                    "regime": str(getattr(read, "regime", "")),
                    "reason": reason,
                })
        if not rows:
            return Absent("no same-degree same-rank pair: transport is "
                          "defined only between matching bands")
        return {"rows": rows,
                "accepted": sum(1 for r in rows if r["accepted"]),
                "total": len(rows)}

    # ---- 7. exchange and rotation characters ------------------------

    def _read_statistics(self, spacetime):
        accepted = [f for f in self.candidates if f is not None]
        if len(accepted) < 2:
            return Absent("the Berry-cancelled exchange needs two odd "
                          "clusters to interchange")
        return Absent("no exchange cobordism was constructed on this host: "
                      "the reference loop requires a non-exchanging motion "
                      "of the same geometric footprint")

    # ---- 8. the world-tube crossing readouts ------------------------

    @staticmethod
    def _m0_vertices(spacetime):
        """The incoming boundary's vertices, or an empty list if closed.

        Delegates to the module-level rule the host is built against, so the
        host and the readout cannot disagree about what M0 is. A closed
        complex has none, and then there is no M0 -- which the paper's
        readouts require, so the channel refuses rather than inventing one.
        """
        try:
            return boundary_vertices(spacetime)
        except Exception:                                 # noqa: BLE001
            return []

    def _read_crossings(self, spacetime):
        accepted = [f for f in self.candidates if f is not None]
        if not accepted:
            return Absent("no accepted band: no world tube to cross a level")
        boundary = self._m0_vertices(spacetime)
        if not boundary:
            return Absent(
                "closed host: no incoming boundary M0, so tau has no "
                "reference surface. The crossing readouts are defined on a "
                "cobordism with boundary M0 + M1")
        try:
            temporal = obs.CrossingReadouts.temporalFunction(
                spacetime, boundary)
        except Exception as error:                        # noqa: BLE001
            return Absent("temporal function unavailable: %s" % error)
        if not temporal.certified:
            return Absent("Re tau is not a certified temporal function (%s); "
                          "no level set is admissible"
                          % (", ".join(_reasons(temporal))
                             or "no reason named"))
        tubes = []
        for index, fiber in enumerate(accepted):
            tube = obs.WorldTubeInput()
            tube.tubeId = "band-%d" % index
            tube.band = fiber
            tube.orientation = +1
            tube.certifiedQuarkTube = False
            tubes.append(tube)
        levels = [float(x) for x in temporal.layer] if temporal.layer else []
        level = max(levels) / 2.0 if levels else 0.0
        block = {"level": level, "tubes": len(tubes)}
        try:
            mass = obs.CrossingReadouts.crossingMass(tubes, temporal,
                                                    level, 0.0)
            block["crossingMass"] = _finite(mass.crossingMass)
            block["admissible"] = int(mass.admissibleCrossings)
            block["refused"] = int(mass.refusedCrossings)
            block["calibrated"] = bool(mass.calibrated)
            block["units"] = str(mass.units)
        except Exception as error:                        # noqa: BLE001
            block["crossingMass"] = Absent("crossing mass failed: %s" % error)
        try:
            baryon = obs.CrossingReadouts.baryonNumber(tubes, temporal,
                                                      level, 0.0)
            block["baryonNumber"] = _finite(baryon.baryonNumber)
            block["quarkTubes"] = int(baryon.quarkTubes)
            # Named defects, not a count: a tube whose crossing sign
            # disagrees with its determinant-line winding is reported, never
            # silently resolved.
            block["signDefects"] = [str(d) for d in baryon.signDefects]
        except Exception as error:                        # noqa: BLE001
            block["baryonNumber"] = Absent("baryon sum failed: %s" % error)
        if not block.get("quarkTubes"):
            block["baryonNote"] = ("no tube carries a quark certificate, so "
                                   "the one-third sum has no term")
        try:
            profile = obs.CrossingReadouts.chargePowerProfile(
                tubes, temporal, level)
            block["chargePower"] = {
                "eigenvalues": [_finite(x) for x in profile.eigenvalues],
                "power": [_finite(x) for x in profile.power],
                "normalized": bool(profile.normalized),
                "monopole": _finite(profile.monopole),
                "reasons": _reasons(profile),
            }
        except Exception as error:                        # noqa: BLE001
            block["chargePower"] = Absent("charge power failed: %s" % error)
        signs = []
        for tube in tubes:
            try:
                crossing = obs.CrossingReadouts.crossing(tube, temporal,
                                                         level)
            except Exception:                             # noqa: BLE001
                continue
            signs.append({
                "tubeId": str(crossing.tubeId),
                "sign": int(crossing.sign),
                "admissible": bool(crossing.admissible),
                "perpendicular": complex(crossing.perpendicular),
                "reasons": _reasons(crossing),
            })
        block["crossings"] = signs
        return block

    # ---- 9. <J^2> and Var(J^2) --------------------------------------

    def _read_spin(self):
        accepted = [f for f in self.candidates if f is not None]
        if not accepted:
            return Absent("no accepted band: no covariance to Wick-contract")
        live = [s for s in self.states if s is not None]
        if not live:
            return Absent("no band projector yielded a quasi-free covariance; "
                          "a band whose restricted metric fails positivity "
                          "supplies no state")
        certified = [q for q in self.quarks
                     if str(q.classification) == "quark"]
        if len(certified) < 3:
            return Absent("the total-space J^2 read needs three certified "
                          "quark clusters; %d covariance state(s) exist and "
                          "%d cluster(s) are certified quarks"
                          % (len(live), len(certified)))
        return Absent("three certified quarks exist but no total-space spin "
                      "operator was assembled on this host")

    # ---- 10. Betti numbers ------------------------------------------

    @staticmethod
    def _read_betti(spacetime, config):
        """Independent topological observables.

        The paper keeps Betti numbers as observables in their own right. They
        are NOT a quark count and NOT a success condition: the quark is sought
        as a persistent modular spectral cluster, not as a hole.
        """
        try:
            values = list(MC.betti(spacetime))
        except Exception as error:                        # noqa: BLE001
            return Absent("Betti numbers unavailable: %s" % error)
        numbers = {degree: (int(values[degree]) if degree < len(values)
                            else None)
                   for degree in config["betti_degrees"]}
        if all(v is None for v in numbers.values()):
            return Absent("the complex reports no Betti number at the "
                          "declared degrees")
        return {"numbers": numbers}

    # ---- 11. the verdict --------------------------------------------

    def _read_verdict(self):
        accepted = [f for f in self.candidates if f is not None]
        certified = [q for q in self.quarks
                     if str(q.classification) == "quark"]
        if len(certified) < 3:
            return Absent(
                "the proton certificate is evaluated on three certified "
                "quark clusters; %d band(s) accepted, %d quark read(s), %d "
                "certified" % (len(accepted), len(self.quarks),
                               len(certified)))
        evidence = obs.BaryonCandidateEvidence()
        try:
            read = obs.ParticleClusters().classifyBaryon(evidence)
        except Exception as error:                        # noqa: BLE001
            return Absent("classifier refused: %s" % error)
        return {"classification": str(read.classification),
                "confidence": _finite(read.confidence),
                "reasons": _reasons(read)}

    # ---- serialization ----------------------------------------------

    def to_json(self):
        return _json_safe({
            "step": self.step,
            "objective": self.objective,
            "clusters": self.clusters,
            "bands": self.bands,
            "anchors": self.anchors,
            "transports": self.transports,
            "statistics": self.statistics,
            "crossings": self.crossings,
            "spin": self.spin,
            "betti": self.betti,
            "verdict": self.verdict,
        })


# =====================================================================
# the drive -- unforced emergence, one frame per engine unit
# =====================================================================

def drive(config, progress=False, on_frame=None):
    """Drive unforced emergence, reading a frame after every engine unit.

    `on_frame(frames, index)` is called as each unit completes, so a caller
    can display a run while it is still running. It is the ONLY difference
    between a live drive and a headless one: the loop, the engine calls and
    the frames are the same either way, so a live view cannot diverge from
    the run it claims to be showing.

    Returns a `DriveResult` carrying the frames and the terminator.
    """
    host = build_cobordism_host(config["size"], config["host_seed"],
                                config["edge_disposition"])
    node = MC(host, [], [], list(config["register_degrees"]), 1.0,
              config["seed"])
    node.set_objective(cob.JointStationarityObjective())
    # Declared here rather than inherited from the register degrees above: the
    # degrees a register is constructed at and the degrees whose entropy should
    # be stationary are different questions.
    node.set_hodge_degrees(list(config["hodge_degrees"]))
    node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                             MC.EmergenceSubmode.STRICT)
    # M0 is HELD, not targeted. Declaring the region says only WHICH cells do
    # not vary -- the paper's fixed boundary with a relaxed bulk. No pinned
    # objective is set, so the bulk objective scores the whole cobordism
    # including M0 and the run stays bit-identical to an unpinned one in
    # everything except which coordinates are free.
    node.declare_pinned_region(M0_REGION, set(boundary_vertices(host)))

    # EVERY frame reads `node.spacetime()`, never the `host` handed to the
    # constructor. Stage 1 REPLACES the node's complex when it commits a move,
    # so `host` stops being the complex the node is driving from the first
    # committed move onward. Reading it would freeze every panel at the initial
    # geometry while the objective tracked something else entirely -- the two
    # diverge silently, with no error and no empty frame to give it away.
    frames = [EmergenceFrame(node, node.spacetime(), 0, config)]
    if progress:
        _report(frames[-1])
    if on_frame is not None:
        on_frame(frames, 0)
    terminator = Terminator.STEPS
    for step in range(1, config["steps"] + 1):
        before = _objective_total(frames[-1])
        list(node.run_stage1(max_steps=config["stage1_iters"],
                             n_candidate_moves=config["candidate_moves"],
                             max_lookahead=config["surgical_depth"]))
        list(node.run_stage2(max_iters=config["stage2_iters"],
                             tolerance=config["tolerance"]))
        frames.append(EmergenceFrame(node, node.spacetime(), step, config))
        if progress:
            _report(frames[-1])
        if on_frame is not None:
            on_frame(frames, step)
        # The unit is complete and its frame is published before the exit is
        # considered, so a run that stops here has still reported the unit
        # that stopped it.
        #
        # Convergence on the FINAL unit reports `tolerance-reached` even
        # though the budget also ran out. Both are true, and this is the more
        # informative of the two: a reader learns the run had converged, and
        # can still see it used its whole budget from the unit count, which
        # the document and the stdout line both carry.
        if _converged(before, _objective_total(frames[-1]),
                      config["tolerance"]):
            terminator = Terminator.TOLERANCE
            break
    return DriveResult(frames, terminator)


def _objective_total(frame):
    """A frame's scalar objective, or None where it was not measured."""
    total = frame.objective.get("total")
    return total if isinstance(total, (int, float)) else None


def _converged(before, after, tolerance):
    """Whether one engine unit failed to improve the objective by `tolerance`.

    ABSOLUTE, never relative: the test is on the improvement itself, not on
    its ratio to the objective, so the same tolerance means the same thing at
    every scale the objective happens to take.

    An unmeasured objective at either end is not convergence. It is an
    absence, and a run may not stop on one -- stopping there would report a
    converged run on the strength of a number nobody read.
    """
    if before is None or after is None:
        return False
    if not (math.isfinite(before) and math.isfinite(after)):
        return False
    # A unit that RAISES the objective has also failed to improve it by the
    # tolerance, so this stops on that too rather than running on.
    return (before - after) < tolerance


def _report(frame):
    """One line per frame on stdout, so a long run is legible while it runs."""
    verdict = (frame.verdict.reason if isinstance(frame.verdict, Absent)
               else frame.verdict["classification"])
    clusters = ("absent" if isinstance(frame.clusters, Absent)
                else frame.clusters["count"])
    bands = ("absent" if isinstance(frame.bands, Absent)
             else frame.bands["accepted"])
    total = frame.objective.get("total")
    sys.stdout.write(
        "[step %2d] objective %s | clusters %s | accepted bands %s | %s\n"
        % (frame.step,
           "n/a" if total is None else "%.6g" % total,
           clusters, bands, verdict))
    sys.stdout.flush()


# =====================================================================
# the overlay -- an absent panel still says what is absent, and why
# =====================================================================

class StableLayout:
    """Jitter-free drawing positions: the layout read's normalized embedding,
    rigidly aligned to the previous frame and eased toward it, with an eased
    auto-fit view.

    Classical MDS is defined only up to rotation, reflection and scale, and it
    is globally sensitive, so a small change to the complex can reshuffle the
    whole cloud. Scale is already fixed by the read's RMS normalization; this
    removes the orientation ambiguity by Procrustes, eases the positions, and
    auto-fits the view, so the structure stays legible as the complex changes.

    PRESENTATION ONLY. It consumes what the layout read measured and produces
    where to draw it; it feeds nothing back into any measurement or record.
    """

    def __init__(self, ease=DECLARED_LAYOUT_EASE,
                 view_ease=DECLARED_LAYOUT_VIEW_EASE,
                 pad=DECLARED_LAYOUT_PAD):
        self._previous = None
        self._view = None
        self.ease = ease
        self.view_ease = view_ease
        self.pad = pad

    def place(self, coords):
        """Stabilized positions for one frame's raw layout coordinates."""
        import numpy as np

        current = {v: np.asarray(p, dtype=float) for v, p in coords.items()}
        if len(current) < 2 or self._previous is None:
            # The first frame defines the frame of reference; there is nothing
            # to align to and nothing to ease toward.
            self._previous = current
            return {v: tuple(p) for v, p in current.items()}
        shared = [v for v in current if v in self._previous]
        if len(shared) >= 2:
            cur = np.array([current[v] for v in shared])
            ref = np.array([self._previous[v] for v in shared])
            cur_centre, ref_centre = cur.mean(0), ref.mean(0)
            u, _s, vt = np.linalg.svd((cur - cur_centre).T
                                      @ (ref - ref_centre))
            rotation = u @ vt          # rotation/reflection only, no scale
            aligned = {v: (p - cur_centre) @ rotation + ref_centre
                       for v, p in current.items()}
        else:
            # Too little in common to define an alignment. Taking the raw
            # embedding is honest; inventing a rotation from one point is not.
            aligned = current
        eased = {}
        for vertex, target in aligned.items():
            previous = self._previous.get(vertex)
            # A vertex that has just appeared has nowhere to ease FROM, so it
            # takes its target outright rather than sliding in from a position
            # it never occupied.
            eased[vertex] = (target if previous is None
                             else previous + self.ease * (target - previous))
        self._previous = eased
        return {v: tuple(p) for v, p in eased.items()}

    def view(self, coords):
        """An eased bounding box around the current cloud.

        Never grow-only: a view that could only expand would shrink the
        structure to an unreadable dot as soon as one frame spread out.
        """
        import numpy as np

        points = np.array(list(coords.values()), dtype=float)
        low, high = points.min(0), points.max(0)
        pad = self.pad * max(high[0] - low[0], high[1] - low[1], 1e-6)
        box = [low[0] - pad, high[0] + pad, low[1] - pad, high[1] + pad]
        if self._view is None:
            self._view = box
        else:
            self._view = [self._view[i]
                          + self.view_ease * (box[i] - self._view[i])
                          for i in range(4)]
        return self._view


def stabilize(frames):
    """Every frame's stabilized positions and view, computed once in order.

    Precomputed rather than accumulated during drawing because the alignment
    is a chain: each frame is aligned to the one before it. A renderer that
    redraws a frame, or draws only the last, would otherwise get a different
    picture depending on what it had drawn before.

    Returns one entry per frame, `None` where the layout itself is absent.
    """
    state = StableLayout()
    placed = []
    for frame in frames:
        if isinstance(frame.layout, Absent):
            placed.append(None)
            continue
        coords = state.place(frame.layout["coords"])
        placed.append({"coords": coords, "view": state.view(coords)})
    return placed


def _absent_panel(axis, title, reason):
    """Draw a legible statement of absence. Never a blank, never a zero."""
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor("#bbbbbb")
        spine.set_linestyle(":")
    axis.text(0.5, 0.5, _wrap(reason, 34), transform=axis.transAxes,
              ha="center", va="center", fontsize=6.5, color="#666666",
              style="italic", wrap=True)


def _wrap(text, width):
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width)[:6])


def _panel_objective(axis, frames):
    axis.set_title("objective trace", fontsize=8)
    totals = [f.objective.get("total") for f in frames]
    steps = [f.step for f in frames]
    finite = [(s, t) for s, t in zip(steps, totals) if t is not None]
    if not finite:
        return _absent_panel(axis, "objective trace",
                             "the objective returned no finite value")
    axis.plot([s for s, _ in finite], [t for _, t in finite],
              marker="o", markersize=2.5, linewidth=1.0, color="#1f4e79")
    axis.set_xlabel("engine unit", fontsize=6)
    axis.tick_params(labelsize=6)
    axis.grid(alpha=0.25, linewidth=0.4)


def _panel_layout(axis, frame, placement=None):
    # POSITION is a drawing artefact; COLOUR is a measurement. The two are
    # deliberately named apart in the title, because stabilizing the layout
    # and colouring the edges together make the picture look more physical
    # than it is -- where a vertex sits still means nothing at all.
    title = "complex -- position: drawing only | colour: causal type of l^2"
    if isinstance(frame.layout, Absent):
        return _absent_panel(axis, title, frame.layout.reason)
    from matplotlib.lines import Line2D

    coords = (placement or {}).get("coords") or frame.layout["coords"]
    axis.set_title(title, fontsize=8)
    vertices = frame.layout["vertices"]
    classes = frame.layout["edge_causal_classes"]
    seen = []
    for (a, b), causal in zip(frame.layout["edges"], classes):
        va = coords[vertices[a]]
        vb = coords[vertices[b]]
        axis.plot([va[0], vb[0]], [va[1], vb[1]], linewidth=0.7,
                  color=DECLARED_CAUSAL_COLOURS[causal], zorder=1)
        if causal not in seen:
            seen.append(causal)
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    axis.scatter(xs, ys, s=6, color="#333333", zorder=2)
    counts = {c: classes.count(c) for c in seen}
    handles = [Line2D([0], [0], color=DECLARED_CAUSAL_COLOURS[c],
                      linewidth=1.4,
                      label="%s (%d)" % (_CAUSAL_LEGEND[c], counts[c]))
               for c in CausalClass.ALL if c in counts]
    if handles:
        axis.legend(handles=handles, fontsize=5, loc="upper right",
                    frameon=True, framealpha=0.85, borderpad=0.3,
                    handlelength=1.2)
    if placement and placement.get("view"):
        view = placement["view"]
        axis.set_xlim(view[0], view[1])
        axis.set_ylim(view[2], view[3])
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")


#: What each causal class means, spelled out in the legend rather than left to
#: the colour alone. `indefinite` is the one that needs saying: it is not a
#: third kind of causal character, it is the absence of a definite one.
_CAUSAL_LEGEND = {
    CausalClass.SPACELIKE: "spacelike  Re l^2 > 0",
    CausalClass.TIMELIKE: "timelike  Re l^2 < 0",
    CausalClass.NULL: "null  l^2 = 0",
    CausalClass.INDEFINITE: "indefinite  Im l^2 != 0",
}


def _panel_dual(axis, frame, placement, channel, title, cmap):
    """One dual-curvature panel: dual nodes at primal cell centroids, edges
    across shared facets, heat-coloured by one signed channel."""
    if isinstance(frame.dual, Absent):
        return _absent_panel(axis, title, frame.dual.reason)
    if isinstance(frame.layout, Absent):
        return _absent_panel(axis, title,
                             "no drawing layout: nowhere to place the dual")
    import numpy as np

    coords = (placement or {}).get("coords") or frame.layout["coords"]
    cells = frame.dual["cells"]
    positions = np.full((len(cells), 2), np.nan)
    values = np.zeros(len(cells))
    for i, cell in enumerate(cells):
        here = [coords[v] for v in cell["vertices"] if v in coords]
        if here:
            positions[i] = np.mean(np.asarray(here, dtype=float), axis=0)
        values[i] = cell[channel]
    finite = np.all(np.isfinite(positions), axis=1)
    axis.set_title("%s  (%d cells)" % (title, len(cells)), fontsize=8)
    if not finite.any():
        return _absent_panel(
            axis, title, "no dual node has a drawable position")
    for a, b in frame.dual["adjacency"]:
        if a < len(cells) and b < len(cells) and finite[a] and finite[b]:
            axis.plot([positions[a, 0], positions[b, 0]],
                      [positions[a, 1], positions[b, 1]],
                      color="0.82", linewidth=0.4, zorder=1)
    shown = values[finite]
    magnitude = np.abs(shown)
    if magnitude.max() <= DECLARED_CAUSAL_NULL_TOLERANCE:
        # A channel that is identically zero would draw as one flat colour and
        # read as "measured, uniform". Say which channel vanished and why it
        # can: no hinge of the kind that carries it.
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(0.5, 0.5,
                  _wrap("identically 0: no hinge carries %s curvature here"
                        % channel, 30),
                  transform=axis.transAxes, ha="center", va="center",
                  fontsize=6.5, color="#666666", style="italic")
        return
    limit = (float(np.percentile(magnitude, DECLARED_HEAT_CLIP_PERCENTILE))
             if finite.sum() >= 5 else float(magnitude.max()))
    if not limit > 0:
        limit = float(magnitude.max()) or 1.0
    clipped = np.clip(shown, -limit, limit)
    scatter = axis.scatter(positions[finite, 0], positions[finite, 1],
                           c=clipped, cmap=cmap, vmin=-limit, vmax=limit,
                           s=18, zorder=2, edgecolors="0.3", linewidths=0.2)
    bar = axis.figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.02)
    # The centre is the whole point of a diverging map: without the zero tick
    # labelled, a reader cannot tell a saddle from a peak.
    bar.set_ticks([-limit, 0.0, limit])
    bar.ax.tick_params(labelsize=5)
    bar.set_label("signed, 0 at centre", fontsize=5)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")


def _panel_dual_spatial(axis, frame, placement=None):
    return _panel_dual(
        axis, frame, placement, "spatial",
        "dual: spatial curvature  Re eps*|star|  (timelike hinges)",
        DECLARED_HEAT_CMAP_SPATIAL)


def _panel_dual_temporal(axis, frame, placement=None):
    return _panel_dual(
        axis, frame, placement, "temporal",
        "dual: temporal curvature  Im eps*|star|  (spacelike hinges)",
        DECLARED_HEAT_CMAP_TEMPORAL)


def _panel_clusters(axis, frame):
    title = "persistent modular clusters"
    if isinstance(frame.clusters, Absent):
        return _absent_panel(axis, title, frame.clusters.reason)
    sizes = frame.clusters["sizes"]
    axis.set_title(title, fontsize=8)
    axis.bar(range(len(sizes)), sizes, color="#608c64", width=0.7)
    axis.set_xlabel("cluster", fontsize=6)
    axis.set_ylabel("support cells", fontsize=6)
    axis.tick_params(labelsize=6)


def _panel_bands(axis, frame):
    title = "fibers: rank, gap, localization"
    if isinstance(frame.bands, Absent):
        return _absent_panel(axis, title, frame.bands.reason)
    rows = frame.bands["rows"]
    accepted = [r for r in rows if r.get("accepted")]
    if not accepted:
        reason = rows[0].get("reason") if rows else "no band read"
        return _absent_panel(axis, title,
                             "no band met its certificate: %s" % reason)
    axis.set_title(title, fontsize=8)
    ranks = [r["rank"] for r in accepted]
    gaps = [r["lowerGap"] or 0.0 for r in accepted]
    axis.bar(range(len(ranks)), ranks, color="#608c64", width=0.5,
             label="rank")
    twin = axis.twinx()
    twin.plot(range(len(gaps)), gaps, marker="s", markersize=3,
              linewidth=0.8, color="#bc8836", label="lower gap")
    twin.tick_params(labelsize=6)
    axis.set_xlabel("accepted band", fontsize=6)
    axis.tick_params(labelsize=6)


def _panel_anchors(axis, frame):
    title = "anchor profile (quark reads)"
    if isinstance(frame.anchors, Absent):
        return _absent_panel(axis, title, frame.anchors.reason)
    rows = frame.anchors["rows"]
    measured = [r for r in rows if r.get("score") is not None]
    if not measured:
        reasons = sorted({x for r in rows for x in (r.get("reasons") or [])})
        return _absent_panel(
            axis, title,
            "no quark read reported an anchor score: %s"
            % (", ".join(reasons[:3]) or "no reason named"))
    axis.set_title(title, fontsize=8)
    labels = ["score", "max term", "participation", "phase disp."]
    first = measured[0]
    values = [first.get("score"), first.get("maxTerm"),
              first.get("participationRatio"), first.get("phaseDispersion")]
    shown = [v if v is not None else 0.0 for v in values]
    axis.barh(range(len(shown)), shown, color="#bc8836", height=0.6)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=6)
    axis.set_xlim(0.0, max(1.0, max(shown) * 1.15))
    axis.tick_params(labelsize=6)
    axis.text(0.98, 0.06, "%d certified" % frame.anchors["certified"],
              transform=axis.transAxes, ha="right", fontsize=6,
              color="#666666")


def _panel_transports(axis, frame):
    title = "transports: accepted vs rejected"
    if isinstance(frame.transports, Absent):
        return _absent_panel(axis, title, frame.transports.reason)
    accepted = frame.transports["accepted"]
    total = frame.transports["total"]
    axis.set_title(title, fontsize=8)
    axis.bar([0, 1], [accepted, total - accepted],
             color=["#608c64", "#b0302a"], width=0.6)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["accepted", "rejected"], fontsize=6)
    axis.tick_params(labelsize=6)
    reasons = sorted({r["reason"] for r in frame.transports["rows"]
                      if not r["accepted"] and r.get("reason")})
    if reasons:
        axis.set_xlabel(_wrap("rejected: " + ", ".join(reasons[:2]), 42),
                        fontsize=5.5, color="#666666", style="italic")


def _panel_statistics(axis, frame):
    title = "exchange / rotation characters"
    if isinstance(frame.statistics, Absent):
        return _absent_panel(axis, title, frame.statistics.reason)
    return _absent_panel(axis, title, "no character was measured")


def _panel_crossings(axis, frame):
    title = "world-tube crossings: sgn Re pi_perp"
    if isinstance(frame.crossings, Absent):
        return _absent_panel(axis, title, frame.crossings.reason)
    crossings = frame.crossings.get("crossings") or []
    if not crossings:
        return _absent_panel(axis, title,
                             "no tube crossed the level; %d refused"
                             % frame.crossings.get("refused", 0))
    admissible = [c for c in crossings if c["admissible"]]
    if not admissible:
        reasons = sorted({r for c in crossings for r in c["reasons"]})
        return _absent_panel(axis, title,
                             "every crossing refused: %s"
                             % (", ".join(reasons) or "no reason named"))
    axis.set_title(title, fontsize=8)
    signs = [c["sign"] for c in admissible]
    axis.bar(range(len(signs)), signs,
             color=["#1f4e79" if s > 0 else "#b0302a" for s in signs],
             width=0.6)
    axis.axhline(0.0, linewidth=0.5, color="#333333")
    axis.set_ylim(-1.4, 1.4)
    axis.set_xlabel("admissible crossing", fontsize=6)
    axis.tick_params(labelsize=6)


def _panel_mass(axis, frame):
    title = "crossing mass and one-third baryon sum"
    if isinstance(frame.crossings, Absent):
        return _absent_panel(axis, title, frame.crossings.reason)
    mass = frame.crossings.get("crossingMass")
    baryon = frame.crossings.get("baryonNumber")
    if isinstance(mass, Absent):
        return _absent_panel(axis, title, mass.reason)
    if mass is None and baryon is None:
        return _absent_panel(axis, title,
                             "no admissible crossing contributes a term")
    axis.set_title(title, fontsize=8)
    note = frame.crossings.get("baryonNote", "")
    units = frame.crossings.get("units", "")
    axis.text(0.05, 0.72, "crossing mass: %s" % ("n/a" if mass is None
                                                 else "%.6g" % mass),
              transform=axis.transAxes, fontsize=7)
    axis.text(0.05, 0.52, "units: %s" % (units or "unknown"),
              transform=axis.transAxes, fontsize=6, color="#666666")
    axis.text(0.05, 0.32, "B = %s" % ("n/a" if baryon is None
                                      else "%.4g" % baryon),
              transform=axis.transAxes, fontsize=7)
    if note:
        axis.text(0.05, 0.10, _wrap(note, 40), transform=axis.transAxes,
                  fontsize=5.5, color="#666666", style="italic")
    axis.set_xticks([])
    axis.set_yticks([])


def _panel_spin(axis, frame):
    title = "<J^2> and Var(J^2)"
    if isinstance(frame.spin, Absent):
        return _absent_panel(axis, title, frame.spin.reason)
    return _absent_panel(axis, title, "no sharp-spin read was assembled")


def _panel_betti(axis, frame):
    title = "Betti numbers (independent observable)"
    if isinstance(frame.betti, Absent):
        return _absent_panel(axis, title, frame.betti.reason)
    numbers = frame.betti["numbers"]
    degrees = sorted(numbers)
    values = [numbers[d] if numbers[d] is not None else 0 for d in degrees]
    axis.set_title(title, fontsize=8)
    axis.bar([str(d) for d in degrees], values, color="#5a7ca0", width=0.6)
    axis.set_xlabel("degree", fontsize=6)
    axis.tick_params(labelsize=6)
    axis.text(0.5, 0.92, "not a quark count", transform=axis.transAxes,
              ha="center", fontsize=5.5, color="#666666", style="italic")


def _panel_verdict(axis, frame):
    title = "verdict and named reasons"
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])
    if isinstance(frame.verdict, Absent):
        axis.text(0.5, 0.72, "no verdict", transform=axis.transAxes,
                  ha="center", fontsize=9, color="#b0302a")
        axis.text(0.5, 0.36, _wrap(frame.verdict.reason, 40),
                  transform=axis.transAxes, ha="center", va="center",
                  fontsize=6, color="#666666", style="italic")
        return
    axis.text(0.5, 0.74, frame.verdict["classification"],
              transform=axis.transAxes, ha="center", fontsize=9,
              color="#1f4e79")
    reasons = frame.verdict.get("reasons") or []
    axis.text(0.5, 0.34, _wrap(", ".join(reasons) or "no reason named", 40),
              transform=axis.transAxes, ha="center", va="center",
              fontsize=6, color="#666666")


#: Panels whose painter also takes the frame's stabilized placement. Named
#: rather than detected by signature, so adding a painter that needs it is a
#: deliberate act rather than something that silently starts working.
_PLACED_PANELS = ("layout", "dual_spatial", "dual_temporal")

_PANELS = [
    ("objective", _panel_objective),
    ("layout", _panel_layout),
    ("dual_spatial", _panel_dual_spatial),
    ("dual_temporal", _panel_dual_temporal),
    ("clusters", _panel_clusters),
    ("bands", _panel_bands),
    ("anchors", _panel_anchors),
    ("transports", _panel_transports),
    ("statistics", _panel_statistics),
    ("crossings", _panel_crossings),
    ("mass", _panel_mass),
    ("spin", _panel_spin),
    ("betti", _panel_betti),
    ("verdict", _panel_verdict),
]


#: Grid the panels are laid out on. Wide enough for every panel with room to
#: spare; the spare axes are removed rather than left as empty boxes, which
#: would read as absent measurements.
DECLARED_PANEL_GRID = (4, 4)


def draw_frame(figure, frames, index, placed=None):
    """Draw one frame's panels onto a figure.

    `placed` is `stabilize(frames)`. Passing it is optional so a caller can
    draw a single frame without it, in which case the raw layout is used and
    the picture is correct but unaligned.
    """
    figure.clear()
    frame = frames[index]
    placement = placed[index] if placed else None
    rows, columns = DECLARED_PANEL_GRID
    axes = figure.subplots(rows, columns)
    flat = [ax for row in axes for ax in row]
    for axis in flat[len(_PANELS):]:
        figure.delaxes(axis)
    for (name, painter), axis in zip(_PANELS, flat):
        if name == "objective":
            painter(axis, frames[:index + 1])
        elif name in _PLACED_PANELS:
            painter(axis, frame, placement)
        else:
            painter(axis, frame)
    disposition = frame.config.get("edge_disposition",
                                   DECLARED_EDGE_DISPOSITION)
    # `foliated` prescribes a causal order rather than letting one emerge, so
    # a frame drawn under it must say so on its face and never read as
    # emergent.
    seed_note = ("seed %s -- a PRESCRIBED foliation, not emergent"
                 % disposition if disposition == EdgeDisposition.FOLIATED
                 else "seed %s" % disposition)
    figure.suptitle(
        "unforced Regge-Hodge emergence -- engine unit %d of %d -- %s "
        "(certificates read post-hoc, firewalled from the objective)"
        % (frame.step, frames[-1].step, seed_note), fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.95))


def _interactive_backends():
    """The interactive matplotlib backends, lowercased.

    Asked of matplotlib rather than hard-coded, so the set cannot drift out
    of step with the installed version. The fallback names the file-only
    backends instead, which is the smaller and far more stable list.
    """
    try:
        from matplotlib.backends import BackendFilter, backend_registry
        return {name.lower() for name in
                backend_registry.list_builtin(BackendFilter.INTERACTIVE)}
    except ImportError:                     # matplotlib < 3.9
        import matplotlib
        try:
            return {name.lower() for name in matplotlib.rcsetup.interactive_bk}
        except AttributeError:
            return set()


def drive_live(config, progress=False):
    """Drive while drawing each unit as it completes, then return the result.

    The compute runs on a worker thread and the figure is drawn on the main
    one, because a GUI toolkit may only be driven from the thread that owns
    it. That is safe here rather than merely conventional: `run_stage1` and
    `run_stage2` release the GIL, so the worker genuinely proceeds while the
    main thread draws, and the worker only ever APPENDS to the frame list
    while the main thread reads indices it has already been handed.

    The drive itself is `drive`, unchanged and un-forked. A live run and a
    headless one execute the same loop with the same engine calls; only the
    callback differs. There is no second code path to diverge.
    """
    import queue
    import threading

    import matplotlib
    import matplotlib.pyplot as plt

    # Checked on the BACKEND, not on whether a figure can be created: a
    # file-only backend like Agg makes figures perfectly well and simply
    # shows nothing, so creating one successfully proves nothing about
    # whether the caller will ever see a frame. Without this the flag would
    # silently degrade into a slower headless run.
    backend = matplotlib.get_backend()
    if backend.lower() not in _interactive_backends():
        raise RuntimeError(
            "--live needs an interactive matplotlib backend; this process "
            "has %r, which renders to files and shows no window. A run "
            "under it would compute every frame and display none of them. "
            "Set MPLBACKEND to an interactive backend (webagg needs no "
            "display), or drop --live and read the rendered --out: the "
            "drive is identical either way." % backend)
    if not plt.isinteractive():
        plt.ion()
    figure = plt.figure(figsize=(15, 9))

    ready = queue.Queue()
    published = {}
    outcome = {}

    def publish(frames, index):
        published["frames"] = frames
        ready.put(index)

    def worker():
        try:
            outcome["result"] = drive(config, progress=progress,
                                      on_frame=publish)
        except BaseException as exc:            # re-raised on the main thread
            outcome["error"] = exc
        finally:
            ready.put(None)

    thread = threading.Thread(target=worker, name="emergence-drive",
                              daemon=True)
    thread.start()
    while True:
        index = ready.get()
        if index is None:
            break
        draw_frame(figure, published["frames"], index)
        figure.canvas.draw_idle()
        # Yields to the GUI event loop; a backend without one still returns.
        plt.pause(0.001)
    thread.join()
    plt.close(figure)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def render(frames, path):
    """Render the overlay to a GIF, MP4 or a single PNG of the last frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 12))
    # Computed once, in frame order: the alignment is a chain, so a renderer
    # that redraws a frame or draws only the last must still see the same
    # positions it would have seen drawing them all in sequence.
    placed = stabilize(frames)
    lowered = path.lower()
    if lowered.endswith(".png") or len(frames) == 1:
        draw_frame(figure, frames, len(frames) - 1, placed)
        figure.savefig(path, dpi=110)
        plt.close(figure)
        return path
    import matplotlib.animation as animation

    def update(index):
        draw_frame(figure, frames, index, placed)
        return []

    movie = animation.FuncAnimation(figure, update, frames=len(frames),
                                    interval=900, blit=False, repeat=False)
    writer = "pillow" if lowered.endswith(".gif") else "ffmpeg"
    movie.save(path, writer=writer, dpi=100)
    plt.close(figure)
    return path


# =====================================================================
# CLI
# =====================================================================

def build_config(size=DECLARED_SIZE, steps=DECLARED_STEPS, seed=DECLARED_SEED,
                 host_seed=DECLARED_HOST_SEED,
                 resolution=DECLARED_RESOLUTION,
                 edge_disposition=DECLARED_EDGE_DISPOSITION,
                 stage1_iters=DECLARED_STAGE1_ITERS,
                 stage2_iters=DECLARED_STAGE2_ITERS,
                 tolerance=DECLARED_TOLERANCE,
                 surgical_depth=DECLARED_SURGICAL_DEPTH):
    if edge_disposition not in EdgeDisposition.ALL:
        raise ValueError(
            "unknown edge disposition %r: expected one of %s"
            % (edge_disposition, ", ".join(EdgeDisposition.ALL)))
    if stage1_iters < 1:
        raise ValueError("stage-one iterations must be at least 1, got %r"
                         % (stage1_iters,))
    if stage2_iters < 1:
        raise ValueError("stage-two iterations must be at least 1, got %r"
                         % (stage2_iters,))
    if surgical_depth < 1:
        raise ValueError("surgical depth must be at least 1, got %r"
                         % (surgical_depth,))
    if not (tolerance > 0.0 and math.isfinite(tolerance)):
        raise ValueError("tolerance must be a positive finite absolute "
                         "threshold, got %r" % (tolerance,))
    return {
        "size": size,
        "steps": steps,
        "seed": seed,
        "host_seed": host_seed,
        "resolution": resolution,
        "edge_disposition": edge_disposition,
        "candidate_moves": DECLARED_CANDIDATE_MOVES,
        "stage1_iters": stage1_iters,
        "tolerance": tolerance,
        "surgical_depth": surgical_depth,
        "stage2_iters": stage2_iters,
        "register_degrees": list(DECLARED_REGISTER_DEGREES),
        "hodge_degrees": list(DECLARED_HODGE_DEGREES),
        "degrees": list(DECLARED_ANALYSIS_DEGREES),
        "betti_degrees": list(DECLARED_BETTI_DEGREES),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Animate unforced emergence and the paper's certificates.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="drive emergence and render the overlay")
    run.add_argument("--size", type=int, default=DECLARED_SIZE)
    run.add_argument("--steps", type=int, default=DECLARED_STEPS)
    run.add_argument("--seed", type=int, default=DECLARED_SEED)
    run.add_argument("--host-seed", type=int, default=DECLARED_HOST_SEED)
    run.add_argument("--resolution", type=float, default=DECLARED_RESOLUTION)
    run.add_argument("--edge-disposition", choices=list(EdgeDisposition.ALL),
                     default=DECLARED_EDGE_DISPOSITION,
                     help="causal character of the seed's edges: random "
                          "(default, magnitude one with the real/imaginary "
                          "split drawn per edge), spacelike (l^2 = +1), "
                          "timelike (l^2 = -1), or foliated (a PRESCRIBED "
                          "light cone: timelike between hop layers of M0, "
                          "spacelike within one)")
    run.add_argument("--stage-one-iterations", type=int,
                     default=DECLARED_STAGE1_ITERS,
                     help="combinatorial updates stage 1 may commit per "
                          "engine unit (default %d)" % DECLARED_STAGE1_ITERS)
    run.add_argument("--stage-two-iterations", type=int,
                     default=DECLARED_STAGE2_ITERS,
                     help="relaxation iterations stage 2 runs per engine "
                          "unit (default %d)" % DECLARED_STAGE2_ITERS)
    run.add_argument("--surgical-depth", type=int,
                     default=DECLARED_SURGICAL_DEPTH,
                     help="how many moves deep stage 1 searches for an "
                          "objective-lowering SEQUENCE: when single moves "
                          "find no improvement it tries 2-move sequences, "
                          "then 3, up to this many, committing a lowering "
                          "sequence whole. The depth covers EVERY move kind "
                          "in the draw -- the four Pachner moves and the "
                          "cone-outs and cone-ins alike -- not the surgical "
                          "moves alone (default %d, single moves)"
                          % DECLARED_SURGICAL_DEPTH)
    run.add_argument("--tolerance", type=float, default=DECLARED_TOLERANCE,
                     help="ABSOLUTE objective tolerance (default %g). Stage "
                          "2 backs its line search off until a trial lowers "
                          "the objective by at least this much, and the run "
                          "EXITS once a whole engine unit fails to improve "
                          "it by this much. Never relative"
                          % DECLARED_TOLERANCE)
    run.add_argument("--live", action="store_true",
                     help="draw each frame as it is computed instead of only "
                          "at the end; still writes --out and --json. Needs "
                          "an interactive matplotlib backend and fails by "
                          "name without one")
    run.add_argument("--out", default="emergence_animation.gif",
                     help="GIF, MP4, or PNG of the final frame")
    run.add_argument("--json", default=None,
                     help="also write the per-frame measurements here")
    run.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = build_config(args.size, args.steps, args.seed, args.host_seed,
                          args.resolution, args.edge_disposition,
                          args.stage_one_iterations,
                          args.stage_two_iterations,
                          args.tolerance, args.surgical_depth)
    result = (drive_live(config, progress=not args.quiet) if args.live
              else drive(config, progress=not args.quiet))
    frames = result.frames
    if not args.quiet and result.terminator == Terminator.TOLERANCE:
        sys.stdout.write(
            "exited on tolerance: one engine unit improved the objective by "
            "less than %g, after %d of %d units\n"
            % (config["tolerance"], frames[-1].step, config["steps"]))
    if args.json:
        document = {"config": config,
                    "terminator": result.terminator,
                    "frames": [f.to_json() for f in frames]}
        with open(args.json, "w") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
        if not args.quiet:
            sys.stdout.write("wrote %s\n" % args.json)
    if args.out:
        path = render(frames, args.out)
        if not args.quiet:
            sys.stdout.write("wrote %s\n" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
