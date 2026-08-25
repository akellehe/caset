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
"""

import argparse
import cmath
import json
import math
import os
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
#: Stage-2 relaxation iterations per unit.
DECLARED_STAGE2_ITERS = 12
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


#: Even weighting of the seed length's real and imaginary parts: the unit
#: vector at Re == Im, so a length of magnitude m carries m/sqrt(2) in each.
_EVEN_WEIGHT = (1.0 + 1.0j) / math.sqrt(2.0)


def build_cobordism_host(n_refine=DECLARED_SIZE, seed=DECLARED_HOST_SEED):
    """A single simplex, refined -- the canonical seed.

    The paper's crossing readouts live on a cobordism: `tau` is the Lorentzian
    distance FROM the incoming boundary M0, and the surfaces are its level
    sets. A CLOSED complex has no such surface, so those readouts cannot run
    on one at any size. A single 4-simplex is a 4-BALL, so it HAS a boundary
    -- `S^3 = M0` -- structurally, rather than by carving one out of a closed
    manifold. The whitepaper prescribes no host topology; it specifies
    cobordisms with `∂W = M0 ⊔ M1`, and this is the smallest complex that is
    one.

    Edge lengths carry EVENLY WEIGHTED real and imaginary parts. The previous
    host initialized every length purely real and positive, so the seed had no
    imaginary part and no causal content at all -- a programme that is
    Lorentzian in every path was starting from a complex that was not. Even
    weighting seeds the general-complex regime instead of one of its two real
    faces, and imposes no state: it is a metric seed, not a carrier.

    NEUTRAL otherwise: no holes, no pinned carrier, no boundary blocks, no
    target register. Whatever the run comes to carry is read afterwards.
    """
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT,
                     1.0, 1.0, T.PREFERRED, T.SolidSimplex(4))
    st.build()
    for edge in st.getEdgeList().toVector():
        edge.setLength(_EVEN_WEIGHT)
    applied = 0
    for step in range(seed, seed + n_refine * 4):
        move = T.AddMove(st, step, False, T.PachnerMode.PreGeometric, False)
        if move.propose() and move.apply():
            applied += 1
        if applied >= n_refine:
            break
    for index, edge in enumerate(st.getEdgeList().toVector()):
        edge.setLength((1.0 + 0.01 * (index % 6)) * _EVEN_WEIGHT)
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
        for edge in edges:
            a = index.get(int(edge.getSource().getId()))
            b = index.get(int(edge.getTarget().getId()))
            if a is None or b is None:
                continue
            length = complex(edge.getLength())
            w = math.sqrt(max(abs(length ** 2), 1e-6))
            weights[a, b] = weights[b, a] = min(weights[a, b], w)
            pairs.append((a, b))
        distances = shortest_path(weights, method="D", directed=False)
        finite = np.isfinite(distances)
        if not finite.any():
            return Absent("no finite graph distance: complex is disconnected")
        distances[~finite] = distances[finite].max() * 1.5
        squared = distances ** 2
        centering = np.eye(n) - np.ones((n, n)) / n
        gram = -0.5 * centering @ squared @ centering
        values, vectors = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1][:2]
        coords = vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))
        coords = coords - coords.mean(0)
        rms = math.sqrt((coords ** 2).sum(1).mean()) or 1.0
        return {"coords": {vertices[i]: tuple(coords[i] / rms)
                           for i in range(n)},
                "edges": pairs,
                "vertices": vertices}

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

def drive(config, progress=False):
    """Drive unforced emergence, reading a frame after every engine unit."""
    host = build_cobordism_host(config["size"], config["host_seed"])
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

    frames = [EmergenceFrame(node, host, 0, config)]
    if progress:
        _report(frames[-1])
    for step in range(1, config["steps"] + 1):
        list(node.run_stage1(max_steps=1,
                             n_candidate_moves=config["candidate_moves"]))
        list(node.run_stage2(max_iters=config["stage2_iters"]))
        frames.append(EmergenceFrame(node, host, step, config))
        if progress:
            _report(frames[-1])
    return frames


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


def _panel_layout(axis, frame):
    title = "complex (drawing layout only, no causal content)"
    if isinstance(frame.layout, Absent):
        return _absent_panel(axis, title, frame.layout.reason)
    coords = frame.layout["coords"]
    axis.set_title(title, fontsize=8)
    for a, b in frame.layout["edges"]:
        va = coords[frame.layout["vertices"][a]]
        vb = coords[frame.layout["vertices"][b]]
        axis.plot([va[0], vb[0]], [va[1], vb[1]], linewidth=0.4,
                  color="#888888", zorder=1)
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    axis.scatter(xs, ys, s=6, color="#1f4e79", zorder=2)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")


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


_PANELS = [
    ("objective", _panel_objective),
    ("layout", _panel_layout),
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


def draw_frame(figure, frames, index):
    """Draw one frame's twelve panels onto a figure."""
    figure.clear()
    frame = frames[index]
    axes = figure.subplots(3, 4)
    flat = [ax for row in axes for ax in row]
    for (name, painter), axis in zip(_PANELS, flat):
        if name == "objective":
            painter(axis, frames[:index + 1])
        else:
            painter(axis, frame)
    figure.suptitle(
        "unforced Regge-Hodge emergence -- engine unit %d of %d "
        "(certificates read post-hoc, firewalled from the objective)"
        % (frame.step, frames[-1].step), fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.95))


def render(frames, path):
    """Render the overlay to a GIF, MP4 or a single PNG of the last frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15, 9))
    lowered = path.lower()
    if lowered.endswith(".png") or len(frames) == 1:
        draw_frame(figure, frames, len(frames) - 1)
        figure.savefig(path, dpi=110)
        plt.close(figure)
        return path
    import matplotlib.animation as animation

    def update(index):
        draw_frame(figure, frames, index)
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
                 resolution=DECLARED_RESOLUTION):
    return {
        "size": size,
        "steps": steps,
        "seed": seed,
        "host_seed": host_seed,
        "resolution": resolution,
        "candidate_moves": DECLARED_CANDIDATE_MOVES,
        "stage2_iters": DECLARED_STAGE2_ITERS,
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
    run.add_argument("--out", default="emergence_animation.gif",
                     help="GIF, MP4, or PNG of the final frame")
    run.add_argument("--json", default=None,
                     help="also write the per-frame measurements here")
    run.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = build_config(args.size, args.steps, args.seed, args.host_seed,
                          args.resolution)
    frames = drive(config, progress=not args.quiet)
    if args.json:
        document = {"config": config,
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
