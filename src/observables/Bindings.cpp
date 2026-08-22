// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/options.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "spacetime/topologies/Topology.h"
#include "spacetime/topologies/Cylinder.h"
#include "spacetime/topologies/Sphere.h"
#include "spacetime/topologies/Toroid.h"
#include "simulations/CDT.h"
#include "spacetime/PachnerMove.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"
#include "spacetime/pachner/ShiftMove.h"
#include "simulations/ReggeSolver.h"
#include "matter/MatterConfiguration.h"
#include "mesh/SimplexFilter.h"
#include "observables/ModularityOptimizer.h"
#include "observables/PersistentModularity.h"
#include "observables/SpectralFiber.h"
#include "observables/SparseGraph.h"
#include "cobordism/AnalyticCache.h"
#include "observables/VolumeProfile.h"
#include "observables/WilsonLoop.h"
#include "observables/Spectral.h"
#include "observables/Record.h"
#include "observables/RegisterContext.h"
#include "observables/RegisterObservable.h"
#include "observables/InteriorHinges.h"
#include "observables/LiveComplex.h"
#include "observables/SingletResidual.h"
#include "observables/BlockResiduals.h"
#include "observables/EmergentMass.h"
#include "observables/EmergentRadius.h"
#include "observables/PairLoopFlavor.h"
#include "observables/ObservableGates.h"
#include "observables/DualVolumeSigns.h"
#include "cobordism/Proton.h"
#include "spacetime/Spacetime.h"
#include "ForceLayout.h"
#include "mesh/VertexList.h"
#include "mesh/EdgeList.h"
#include "spacetime/Signature.h"
#include "mesh/Vertex.h"
#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "spacetime/Metric.h"
#include "Renderer.h"

#include <vector>
#include <algorithm>


namespace py = pybind11;
using namespace tessera;
using namespace tessera::observables;

namespace {

// A JSON-able observable Record -> a native Python object (dict/list/scalar).
py::object recordToPython(const Record &r) {
  switch (r.type()) {
    case Record::Type::Null:
      return py::none();
    case Record::Type::Bool:
      return py::bool_(r.asBool());
    case Record::Type::Int:
      return py::int_(static_cast<long long>(r.asInt()));
    case Record::Type::Double:
      return py::float_(r.asDouble());
    case Record::Type::String:
      return py::str(r.asString());
    case Record::Type::List: {
      py::list out;
      for (const auto &e : r.asList()) out.append(recordToPython(e));
      return out;
    }
    case Record::Type::Map: {
      py::dict out;
      for (const auto &kv : r.asMap()) {
        out[py::str(kv.first)] = recordToPython(kv.second);
      }
      return out;
    }
  }
  return py::none();
}

// A native Python object -> a Record (for report_delta testing). bool is checked
// before int (Python bool is an int subtype).
Record pythonToRecord(const py::handle &o) {
  if (o.is_none()) return Record();
  if (py::isinstance<py::bool_>(o)) return Record(o.cast<bool>());
  if (py::isinstance<py::int_>(o)) {
    return Record(static_cast<std::int64_t>(o.cast<long long>()));
  }
  if (py::isinstance<py::float_>(o)) return Record(o.cast<double>());
  if (py::isinstance<py::str>(o)) return Record(o.cast<std::string>());
  if (py::isinstance<py::dict>(o)) {
    Record::Map m;
    for (const auto &item : o.cast<py::dict>()) {
      m[item.first.cast<std::string>()] = pythonToRecord(item.second);
    }
    return Record(std::move(m));
  }
  if (py::isinstance<py::list>(o) || py::isinstance<py::tuple>(o)) {
    Record::List l;
    for (const auto &e : o) l.push_back(pythonToRecord(e));
    return Record(std::move(l));
  }
  throw std::runtime_error(
      "report_delta: record leaves must be dict/list/str/float/int/bool/None");
}

// Emit the RegisterContext's surplus-selection warning as a Python UserWarning
// (the header's documented binding behavior).
void emitSelectionWarning(const RegisterContext &ctx) {
  if (!ctx.selectionWarning().empty()) {
    auto warnings = py::module_::import("warnings");
    warnings.attr("warn")(ctx.selectionWarning());
  }
}

}  // namespace

// Registers all tessera::observables classes into the `m` submodule
// (i.e. `tessera.observables`). Called from src/bindings.cpp's
// PYBIND11_MODULE entry point.
void register_observables(py::module_ m) {
  // ========================================
  // SparseGraph (for modularity / spectral dimension)
  // ========================================
  py::class_<SparseGraph>(m, "SparseGraph",
      R"doc(Undirected sparse graph in CSR form.

Built from the COO output of ``Spacetime.getDualAdjacency`` (or
similar).  Used by the modularity sweep to compute spectral
dimension on the dual graph.)doc")
      .def_static("fromCOO", &SparseGraph::fromCOO,
                  py::arg("rows"), py::arg("cols"), py::arg("n"),
                  "Construct from COO arrays + node count.")
      .def("nNodes", &SparseGraph::nNodes,
           "Number of nodes.")
      .def("nEdges", &SparseGraph::nEdges,
           "Number of undirected edges.")
      .def("degree", &SparseGraph::degree, py::arg("i"),
           "Degree of node ``i`` (number of incident undirected edges).")
      .def("isBipartite", &SparseGraph::isBipartite,
           "True iff the graph is 2-colorable (no odd cycle).")
      .def("modularity", &SparseGraph::modularity, py::arg("labels"),
           R"doc(Newman-Girvan modularity Q for a node partition.

Q = sum_c [L_c/m - (D_c/2m)^2] over communities c, where L_c is the
intra-community edge count, D_c the summed degree, and m the edge count.
``labels`` has one community id per node (length nNodes()); distinct
values are distinct communities and need not be dense. Returns 0 for an
empty / edgeless graph; raises ValueError if len(labels) != nNodes().)doc")
      .def("diagonalHeatKernel",
           [](const SparseGraph &self,
              const std::vector<std::uint32_t> &starts,
              const std::vector<double> &times,
              int krylovDim) {
             auto flat = self.diagonalHeatKernel(starts, times, krylovDim);
             // Reshape to list-of-lists for Python convenience.
             std::vector<std::vector<double>> out(starts.size(),
                                                   std::vector<double>(times.size()));
             for (std::size_t s = 0; s < starts.size(); ++s) {
               for (std::size_t j = 0; j < times.size(); ++j) {
                 out[s][j] = flat[s * times.size() + j];
               }
             }
             return out;
           },
           py::arg("starts"), py::arg("times"), py::arg("krylovDim") = 30,
           R"doc(Diagonal of the heat kernel ``e^{-t L_sym}`` for each
(start, t) pair.  Returns a list of lists with shape
(len(starts), len(times)).)doc")
      .def("spectralDimension",
           [](const SparseGraph &self, int nWalks, double maxSigma,
              std::uint64_t seed, double tailFraction, int nTimes,
              double tMin, int krylovDim) {
             std::mt19937 rng(seed);
             return self.spectralDimension(nWalks, maxSigma, &rng,
                                           tailFraction, nTimes, tMin,
                                           krylovDim);
           },
           py::arg("nWalks"), py::arg("maxSigma"), py::arg("seed") = 0,
           py::arg("tailFraction") = 0.2, py::arg("nTimes") = 40,
           py::arg("tMin") = 0.5, py::arg("krylovDim") = 30,
           R"doc(Estimate spectral dimension at small / large diffusion times.

Returns ``(D_S_small, D_S_large)``.  Mirrors the Python implementation
in ``examples/modularity.py:Graph.spectral_dimension``.)doc")
      // ── Per-sigma return probability + spectral-dimension curve ──────────
      // Inherited from SpectralGraph (SparseGraph::applyLaplacian installs
      // the symmetric-normalised Laplacian L_sym).  Exposed here so the
      // examples can read the full D_S(sigma) curve straight off the dual
      // graph instead of re-implementing dual-graph diffusion + finite
      // differences in NumPy/SciPy.  Mirrors the EmergentGraph bindings.
      .def("returnProbability",
           &::tessera::graph::SpectralGraph::returnProbability,
           py::arg("sigmas"), py::arg("krylovDim") = 30,
           py::arg("m") = 0, py::arg("seed") = 0,
           R"doc(P(sigma) = (1/|V|) Tr exp(-sigma L_sym) via Krylov-Lanczos
diagonal estimation, evaluated at each diffusion time in ``sigmas``.

``m`` is the Hutchinson-style subsample of start vertices: 0 (the default)
uses ``min(nNodes(), 3000)``; pass ``m = nNodes()`` for the exact trace.
``seed`` controls the subset RNG for reproducibility.

This is the continuous-diffusion counterpart of the discrete random-walk
return probability the modularity / spectral-dimension examples used to
build by hand; feed the result to ``spectralDimensionCurve`` (or
``spectralDimensionSmoothed``) to extract D_S(sigma).)doc")
      .def_static("spectralDimensionCurve",
                  &::tessera::graph::SpectralGraph::spectralDimension,
                  py::arg("sigmas"), py::arg("P"),
                  R"doc(D_S(sigma) = -2 d log P / d log sigma via centered finite
differences (one-sided at the endpoints); NaN where P <= 0 or non-finite.

The full per-sigma curve, aligned with ``sigmas``.  Distinct from the
``spectralDimension(nWalks, maxSigma, ...)`` instance method above, which
random-walk samples and returns only the (small, large) summary pair.)doc")
      .def_static("spectralDimensionSmoothed",
                  &::tessera::graph::SpectralGraph::spectralDimensionSmoothed,
                  py::arg("sigmas"), py::arg("P"),
                  py::arg("windowSize") = 5, py::arg("polyOrder") = 2,
                  R"doc(Savitzky-Golay-smoothed D_S(sigma): a local polynomial of
order ``polyOrder`` is fit over a centered ``windowSize`` window in
(log sigma, log P) and its slope read off at each point.  ``windowSize``
must be odd and >= ``polyOrder + 1``.)doc");
  // ========================================
  // PersistentModularity (#765): label-free persistent component discovery
  // ========================================
  py::class_<ComponentId>(m, "ComponentId",
      R"doc(Stable label-free component identity: a canonical hash derived
from oriented incidence structure and parent lineage (never raw vertex
numbers) plus the multilevel-aggregation level.  Used for persistence
matching and deterministic tie-breaking, never as a physical observable.
Structurally identical (automorphic) components share a hash.)doc")
      .def("canonicalHash", &ComponentId::canonicalHash,
           "The canonical structural hash (32 lowercase hex chars).")
      .def("level", &ComponentId::level,
           "Multilevel-aggregation depth at which the component formed.")
      .def("__eq__", [](const ComponentId &a, const ComponentId &b) {
             return a == b;
           }, py::is_operator())
      .def("__lt__", [](const ComponentId &a, const ComponentId &b) {
             return a < b;
           }, py::is_operator())
      .def("__hash__", [](const ComponentId &a) {
             return py::hash(py::make_tuple(a.canonicalHash(), a.level()));
           })
      .def("__repr__", [](const ComponentId &a) {
             return "ComponentId(" + a.canonicalHash() + ", level=" +
                    std::to_string(a.level()) + ")";
           });

  py::class_<PersistentModularityConfig>(m, "PersistentModularityConfig",
      "Configuration for the label-free multiscale component discovery.")
      .def(py::init<>())
      .def_readwrite("resolutions", &PersistentModularityConfig::resolutions,
                     "Resolution parameters gamma, in scan order.")
      .def_readwrite("baseSeed", &PersistentModularityConfig::baseSeed,
                     "Base of the fixed restart seed sequence "
                     "(restart t uses splitmix64(baseSeed + t)).")
      .def_readwrite("restarts", &PersistentModularityConfig::restarts,
                     "Deterministic restarts per resolution; best exact "
                     "score kept, spread reported.")
      .def_readwrite("maxSweepsPerLevel",
                     &PersistentModularityConfig::maxSweepsPerLevel,
                     "Hard cap on local-move sweeps per aggregation level.")
      .def_readwrite("overlapThreshold",
                     &PersistentModularityConfig::overlapThreshold,
                     "Minimum support overlap for a persistence track to "
                     "continue across adjacent resolutions.");

  py::class_<ComponentRead>(m, "ComponentRead",
      "One discovered component: canonical id, level-0 cell support, cached "
      "sufficient statistics, and the exact per-component scores.")
      .def_readonly("id", &ComponentRead::id)
      .def_readonly("support", &ComponentRead::support,
                    "Level-0 member cell ids (ascending; a set — the order "
                    "carries no convention).")
      .def_readonly("internalWeight", &ComponentRead::internalWeight,
                    "Sigma_in: internal weight counting both directions.")
      .def_readonly("strength", &ComponentRead::strength,
                    "S_C: summed member strength.")
      .def_readonly("conductance", &ComponentRead::conductance,
                    "cut(C)/min(vol C, vol V\\C); 0 when the denominator "
                    "vanishes.")
      .def_readonly("modularityContribution",
                    &ComponentRead::modularityContribution,
                    "This community's exact additive Q_gamma term.");

  py::class_<RestartRead>(m, "RestartRead",
      "One deterministic restart: seed and exact best score.")
      .def_readonly("seed", &RestartRead::seed)
      .def_readonly("q", &RestartRead::q)
      .def_readonly("communities", &RestartRead::communities);

  py::class_<ResolutionSlice>(m, "ResolutionSlice",
      R"doc(Discovery result at one resolution gamma.  ``q`` is the exact
Q_gamma of the winning partition (cold recompute) — the best score across
deterministic restarts, a heuristic proposal, never the NP-hard global
optimum.  ``qIncremental`` is the accepted-delta-Q ledger and must agree
with ``q`` to double round-off.)doc")
      .def_readonly("gamma", &ResolutionSlice::gamma)
      .def_readonly("q", &ResolutionSlice::q)
      .def_readonly("qIncremental", &ResolutionSlice::qIncremental)
      .def_readonly("levels", &ResolutionSlice::levels)
      .def_readonly("components", &ResolutionSlice::components,
                    "Final-level components, ordered by canonical hash.")
      .def_readonly("hierarchy", &ResolutionSlice::hierarchy,
                    "hierarchy[k] = communities at aggregation level k+1.")
      .def_readonly("restarts", &ResolutionSlice::restarts)
      .def_readonly("restartSpread", &ResolutionSlice::restartSpread,
                    "max - min of the restart scores (honest heuristic "
                    "uncertainty).");

  py::class_<ComponentMatch>(m, "ComponentMatch",
      R"doc(Matched component pair across adjacent resolutions or cobordism
time.  ``projectorOverlap`` is the documented spectral-projector hook: None
(unknown) until a later ticket supplies projectors and a hook is installed;
unknown is never encoded as zero.)doc")
      .def_readonly("fromId", &ComponentMatch::from)
      .def_readonly("toId", &ComponentMatch::to)
      .def_readonly("fromIndex", &ComponentMatch::fromIndex)
      .def_readonly("toIndex", &ComponentMatch::toIndex)
      .def_readonly("supportOverlap", &ComponentMatch::supportOverlap,
                    "Jaccard overlap of level-0 cell supports.")
      .def_readonly("projectorOverlap", &ComponentMatch::projectorOverlap);

  py::class_<PersistenceTrack>(m, "PersistenceTrack",
      R"doc(A component followed across the resolution scan by maximum
support overlap.  Lifetime/overlap/conductance are proposal diagnostics
only: they neither accept nor veto a fiber.  ``weightAwareStatus`` is the
downstream weight-aware gap/localization/persistence status — None until
the later weight-aware certificate tickets populate it (unknown is never
encoded as zero).)doc")
      .def_readonly("members", &PersistenceTrack::members)
      .def_readonly("memberIndices", &PersistenceTrack::memberIndices)
      .def_readonly("firstSlice", &PersistenceTrack::firstSlice)
      .def_readonly("lastSlice", &PersistenceTrack::lastSlice)
      .def_readonly("gammaFirst", &PersistenceTrack::gammaFirst)
      .def_readonly("gammaLast", &PersistenceTrack::gammaLast)
      .def_readonly("minAdjacentOverlap",
                    &PersistenceTrack::minAdjacentOverlap)
      .def_readonly("meanConductance", &PersistenceTrack::meanConductance)
      .def_property_readonly("weightAwareStatus",
           [](const PersistenceTrack &t) {
             return recordToPython(t.weightAwareStatus);
           });

  py::class_<ScanReport>(m, "ScanReport",
      "The full resolution-scan report: slices, adjacent-slice matches, and "
      "persistence tracks.")
      .def_readonly("slices", &ScanReport::slices)
      .def_readonly("matches", &ScanReport::matches)
      .def_readonly("tracks", &ScanReport::tracks);

  py::class_<InvalidationRead>(m, "InvalidationRead",
      "Components and tracks invalidated by a local change; positions "
      "(slice, level index, index in level) disambiguate automorphic twins "
      "that share a hash.")
      .def_readonly("components", &InvalidationRead::components)
      .def_readonly("positions", &InvalidationRead::positions)
      .def_readonly("tracks", &InvalidationRead::tracks);

  py::class_<PersistentModularity> pm(m, "PersistentModularity",
      R"doc(Label-free discovery of modular components that persist across
resolution and cobordism time (ticket #765, design spec section 8).

Exact identities on the nonnegative weighted undirected similarity graph:
generalized modularity Q_gamma(P) = (1/2m) sum_ij (A_ij - gamma k_i k_j/2m)
[c_i = c_j], evaluated from per-community sufficient statistics, and the
exact O(deg v) cached local move gain
dQ(v: a->b) = (w_vb - w_va)/m - gamma k_v (k_v + S_b - S_a)/(2 m^2), so one
sparse sweep is near O(|E|).  Incremental accumulations are tested against
cold recomputation at double round-off.

Heuristic status: global modularity maximization is NP-hard; discovery is a
deterministic multilevel aggregation from a fixed seed sequence with the
restart spread reported honestly.  The score is blind to signed/complex
Hodge weights.  Modularity is a heuristic proposal generator only: it never
enters the emergence objective and may not veto an otherwise certified
fiber (acceptance belongs to the independent weight-aware certificates,
which later tickets supply; unknown is reported as None, never zero).

Read-only: never calls a solver, never mutates the spacetime it reads.)doc");

  py::enum_<PersistentModularity::WeightMap>(pm, "WeightMap",
      "Documented monotone map from complex edge magnitude to similarity.")
      .value("Unit", PersistentModularity::WeightMap::Unit,
             "w = 1: the combinatorial one-skeleton, exactly the legacy "
             "Newman-Girvan graph.")
      .value("ExpNegAbsLength",
             PersistentModularity::WeightMap::ExpNegAbsLength,
             "w = exp(-|l|): monotone decreasing in the complex edge "
             "magnitude (the mutual-information convention l = -log I).");

  pm.def_static("fromWeightedEdges", &PersistentModularity::fromWeightedEdges,
                py::arg("src"), py::arg("tgt"), py::arg("weight"),
                py::arg("isolatedCells") = std::vector<std::uint64_t>{},
                R"doc(Build from an explicit nonnegative weighted edge list
(cells are arbitrary 64-bit ids; parallel edges consolidate by weight
summation; self-loops and zero weights are ignored).  Raises ValueError on
negative weights or mismatched lengths.)doc")
      .def_static("fromSpacetime",
                  [](const std::shared_ptr<Spacetime> &st,
                     PersistentModularity::WeightMap map) {
                    return PersistentModularity::fromSpacetime(*st, map);
                  },
                  py::arg("spacetime"),
                  py::arg("map") =
                      PersistentModularity::WeightMap::ExpNegAbsLength,
                  "Build the similarity graph from the spacetime one-skeleton "
                  "(read-only).")
      .def("nCells", &PersistentModularity::nCells)
      .def("nEdges", &PersistentModularity::nEdges)
      .def("totalWeight2", &PersistentModularity::totalWeight2,
           "Total adjacency weight 2m = sum_ij A_ij.")
      .def("cellIds", &PersistentModularity::cellIds,
           "Cell ids in internal storage order (no convention).")
      .def("modularityGamma", &PersistentModularity::modularityGamma,
           py::arg("labels"), py::arg("gamma"),
           R"doc(Exact generalized modularity Q_gamma of a fixed partition
(labels[i] labels cellIds()[i]).  The fixed-partition entry point: at
gamma = 1 on a Unit-weight graph this is exactly the Newman-Girvan score.)doc")
      .def("discover",
           [](const PersistentModularity &self, double gamma,
              const PersistentModularityConfig &cfg) {
             py::gil_scoped_release release;
             return self.discover(gamma, cfg);
           },
           py::arg("gamma"), py::arg("config"),
           "Deterministic label-free discovery at one resolution.")
      .def("scanResolutions",
           [](const PersistentModularity &self,
              const PersistentModularityConfig &cfg) {
             py::gil_scoped_release release;
             return self.scanResolutions(cfg);
           },
           py::arg("config"),
           "The configurable resolution-sequence scan with persistence "
           "tracks.")
      .def("matchComponents", &PersistentModularity::matchComponents,
           py::arg("a"), py::arg("b"),
           R"doc(Match components across resolution or cobordism time by
simplex-support overlap (Jaccard on level-0 cell ids over a common cell-id
universe).  When a projector-overlap hook is installed its value is
reported per match; matching decisions remain support-based until a later
ticket supplies the projectors.)doc")
      .def("setProjectorOverlapHook",
           [](PersistentModularity &self, py::object hook) {
             if (hook.is_none()) {
               self.setProjectorOverlapHook(nullptr);
               return;
             }
             self.setProjectorOverlapHook(
                 [hook](const ComponentId &a, const ComponentId &b) {
                   py::gil_scoped_acquire acquire;
                   return hook(a, b).cast<double>();
                 });
           },
           py::arg("hook"),
           "Install (or clear with None) the documented spectral-projector "
           "overlap hook: hook(fromId, toId) -> float in [0, 1].  This "
           "ticket only plumbs the hook; a later ticket supplies the "
           "projectors.")
      .def_static("invalidatedAncestry",
                  &PersistentModularity::invalidatedAncestry,
                  py::arg("report"), py::arg("touchedCells"),
                  R"doc(Components (at every hierarchy level of every slice)
whose support intersects the touched level-0 cells, plus the affected
tracks.  Siblings with disjoint support remain valid.  Pure bookkeeping —
no recomputation.)doc");

  // ========================================
  // SpectralFiber (#769): localized spectral bands and their certificates
  // ========================================
  py::class_<SpectralFiberConfig>(m, "SpectralFiberConfig",
      "Configuration of the spectral-band detector/tracker (ticket #769, "
      "design spec section 9).  Thresholds select which bands are "
      "CERTIFIED, never which eigenvalues exist; no threshold is a "
      "Betti-number oracle and no rank is ever requested.")
      .def(py::init<>())
      .def_readwrite("degrees", &SpectralFiberConfig::degrees,
                     "Form degrees enumerated by enumerateOnComponents.")
      .def_readwrite("groupingTolerance",
                     &SpectralFiberConfig::groupingTolerance,
                     "Relative band-grouping width (fraction of the "
                     "spectral scale).")
      .def_readwrite("minRelativeGap", &SpectralFiberConfig::minRelativeGap,
                     "Isolation floor: certified bands need both gaps >= "
                     "minRelativeGap * scale (a closing gap returns an "
                     "uncertified band).")
      .def_readwrite("gapDominance", &SpectralFiberConfig::gapDominance,
                     "Certified gaps must exceed this multiple of the "
                     "in-band spread.")
      .def_readwrite("residualTolerance",
                     &SpectralFiberConfig::residualTolerance,
                     "Cap on the relative eigen/left/projector residuals.")
      .def_readwrite("gramDefectTolerance",
                     &SpectralFiberConfig::gramDefectTolerance,
                     "Cap on ||Phi^dagger W Phi - J||.")
      .def_readwrite("conditionNumberCap",
                     &SpectralFiberConfig::conditionNumberCap,
                     "Cap on the band condition number ||P||_2.")
      .def_readwrite("denseCrossover", &SpectralFiberConfig::denseCrossover,
                     "Dimension at/above which the self-adjoint path goes "
                     "sparse.")
      .def_readwrite("requestedEigenpairs",
                     &SpectralFiberConfig::requestedEigenpairs,
                     "Lowest eigenpairs the sparse block path computes.")
      .def_readwrite("oversample", &SpectralFiberConfig::oversample,
                     "Extra Ritz vectors beyond requestedEigenpairs.")
      .def_readwrite("maxSolverIterations",
                     &SpectralFiberConfig::maxSolverIterations)
      .def_readwrite("solverTolerance", &SpectralFiberConfig::solverTolerance)
      .def_readwrite("solverSeed", &SpectralFiberConfig::solverSeed,
                     "Seed of the deterministic sparse start block.")
      .def_readwrite("trackOverlapThreshold",
                     &SpectralFiberConfig::trackOverlapThreshold,
                     "Minimum subspace overlap for a certified track "
                     "continuation.")
      .def_readwrite("crossValidateDense",
                     &SpectralFiberConfig::crossValidateDense,
                     "Cross-check solves below the crossover against the "
                     "independent DenseReference kernel and record the "
                     "deviation on the certificate.");

  py::class_<SpectralBandCertificate>(m, "SpectralBandCertificate",
      R"doc(Certification record of one whole spectral band (design spec
section 6.3): degree, rank, lower/upper gap, localization (projector-
diagonal inverse participation ratio), projector/eigen/left residuals,
weighted Gram/signature defect ||Phi^dagger W Phi - J||, band condition
number ||P||_2, Krein inertia (p, q), frequency window, self-adjointness
flag, and the graded #764 Certificate (BandWindow domain; an uncertified
band carries HeuristicDiscovery, which never holds).

A degenerate band is one object of rank >= 2; an unexplained multiplicity
is reported exactly as its rank and never labeled.  Negative signature is
a certificate, never an automatic antiparticle identification.  Unmeasured
quantities are NaN, never zero.)doc")
      .def_readonly("degree", &SpectralBandCertificate::degree)
      .def_readonly("rank", &SpectralBandCertificate::rank)
      .def_readonly("lowerGap", &SpectralBandCertificate::lowerGap)
      .def_readonly("upperGap", &SpectralBandCertificate::upperGap)
      .def_readonly("localization", &SpectralBandCertificate::localization)
      .def_readonly("projectorResidual",
                    &SpectralBandCertificate::projectorResidual)
      .def_readonly("eigenResidual", &SpectralBandCertificate::eigenResidual)
      .def_readonly("leftResidual", &SpectralBandCertificate::leftResidual)
      .def_readonly("gramDefect", &SpectralBandCertificate::gramDefect)
      .def_readonly("conditionNumber",
                    &SpectralBandCertificate::conditionNumber)
      .def_readonly("positiveSignature",
                    &SpectralBandCertificate::positiveSignature)
      .def_readonly("negativeSignature",
                    &SpectralBandCertificate::negativeSignature)
      .def_readonly("frequencyLower",
                    &SpectralBandCertificate::frequencyLower)
      .def_readonly("frequencyUpper",
                    &SpectralBandCertificate::frequencyUpper)
      .def_readonly("selfAdjoint", &SpectralBandCertificate::selfAdjoint)
      .def_readonly("accepted", &SpectralBandCertificate::accepted)
      .def_readonly("certificate", &SpectralBandCertificate::certificate)
      .def("describe", &SpectralBandCertificate::describe)
      .def("__repr__", &SpectralBandCertificate::describe);

  py::class_<FiberOverlapRead>(m, "FiberOverlapRead",
      "Principal-angle / support comparison of two fibers: cells matched "
      "by sorted vertex-id tuple (gauge- and relabeling-invariant).")
      .def_readonly("supportOverlap", &FiberOverlapRead::supportOverlap)
      .def_readonly("sharedCells", &FiberOverlapRead::sharedCells)
      .def_readonly("principalAngles", &FiberOverlapRead::principalAngles)
      .def_readonly("subspaceOverlap", &FiberOverlapRead::subspaceOverlap);

  py::class_<SpectralFiber>(m, "SpectralFiber",
      R"doc(One whole isolated spectral band of a component-restricted Hodge
operator (design spec section 6.3): right/left frames, band projector
P = Phi Psi^dagger W with Psi^dagger W Phi = I, eigenvalues, and the
SpectralBandCertificate.  The band is represented by its PROJECTOR —
individual eigenvectors are a gauge choice and never determine an identity
or a downstream observable.)doc")
      .def("degree", &SpectralFiber::degree)
      .def("rank", &SpectralFiber::rank)
      .def("accepted", &SpectralFiber::accepted)
      .def("rightFrame", &SpectralFiber::rightFrame,
           "Right frame Phi (cells x rank).")
      .def("leftFrame", &SpectralFiber::leftFrame,
           "Left frame Psi (cells x rank), Psi^dagger W Phi = I.")
      .def("projector", &SpectralFiber::projector,
           "The band projector P = Phi Psi^dagger W (cells x cells).")
      .def("weightDiagonal", &SpectralFiber::weightDiagonal,
           "Diagonal inner-product weights W restricted to the band's "
           "cells.")
      .def("eigenvalues", &SpectralFiber::eigenvalues,
           "Band eigenvalues (with multiplicity), sorted by (Re, Im).")
      .def("bandCenter", &SpectralFiber::bandCenter)
      .def("cellVertices", &SpectralFiber::cellVertices,
           "The k-cells carrying the band, as sorted vertex-id tuples in "
           "frame row order.")
      .def("certificate", &SpectralFiber::certificate,
           py::return_value_policy::copy)
      .def_static("overlap", &SpectralFiber::overlap, py::arg("a"),
                  py::arg("b"),
                  "Principal-angle / support comparison (cells matched by "
                  "vertex-id tuple).")
      .def("toRecord",
           [](const SpectralFiber &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization: the JSON-able record of the fiber "
           "(schema-versioned; complex leaves split _re/_im).")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return SpectralFiber::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate from toRecord() output; rejects an unknown "
                  "schema_version (ValueError).");

  py::class_<SpectralBandWindow>(m, "SpectralBandWindow",
      "An accepted band's frequency window as PLAIN DATA for the response "
      "consumer (#768): lower/upper frequency bounds plus the band "
      "certificate.  Carries no operator, frame, or quotient reference.")
      .def_readonly("degree", &SpectralBandWindow::degree)
      .def_readonly("rank", &SpectralBandWindow::rank)
      .def_readonly("frequencyLower", &SpectralBandWindow::frequencyLower)
      .def_readonly("frequencyUpper", &SpectralBandWindow::frequencyUpper)
      .def_readonly("certificate", &SpectralBandWindow::certificate);

  py::class_<FiberMatchRead>(m, "FiberMatchRead",
      "One matched fiber pair across frames or resolutions.  A certified "
      "continuation needs both endpoint bands accepted, equal ranks, and "
      "subspace overlap above the threshold — an endpoint whose gap closed "
      "is reported but never certified (no discontinuous identity flip).")
      .def_readonly("fromIndex", &FiberMatchRead::fromIndex)
      .def_readonly("toIndex", &FiberMatchRead::toIndex)
      .def_readonly("degree", &FiberMatchRead::degree)
      .def_readonly("overlap", &FiberMatchRead::overlap)
      .def_readonly("ranksEqual", &FiberMatchRead::ranksEqual)
      .def_readonly("certifiedContinuation",
                    &FiberMatchRead::certifiedContinuation);

  py::class_<ComponentBandRead>(m, "ComponentBandRead",
      "The band enumeration of one (component, degree) pair: the restricted "
      "operator's cells, verified regime, solver path, covered eigenvalues, "
      "every enumerated band (certified or not), and the solve certificate.")
      .def_readonly("support", &ComponentBandRead::support)
      .def_readonly("degree", &ComponentBandRead::degree)
      .def_readonly("dimension", &ComponentBandRead::dimension)
      .def_readonly("cellVertices", &ComponentBandRead::cellVertices)
      .def_readonly("regime", &ComponentBandRead::regime)
      .def_readonly("solverPath", &ComponentBandRead::solverPath)
      .def_readonly("truncated", &ComponentBandRead::truncated)
      .def_readonly("coveredEigenvalues",
                    &ComponentBandRead::coveredEigenvalues)
      .def_readonly("fibers", &ComponentBandRead::fibers)
      .def_readonly("solveCertificate", &ComponentBandRead::solveCertificate)
      .def("toRecord",
           [](const ComponentBandRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization of the whole read (fibers included).")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return ComponentBandRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version "
                  "(ValueError).");

  py::class_<SpectralFiberTracker>(m, "SpectralFiberTracker",
      R"doc(Extraction and tracking of whole isolated localized Hodge bands
on persistent components (ticket #769; design spec section 9, Algorithm B).

Identity: for a component support S the tracker assembles the weighted
Hodge operator of the full induced subcomplex on S (the same boundary maps,
canonical cell order, and diagonal inner-product weights as the
whole-complex HodgeLaplacian, consumed read-only), so support = all
vertices reproduces HodgeLaplacian.laplacian(k) entry for entry.  Regimes
are VERIFIED, never assumed: positive -> self-adjoint solves (exact dense
below the crossover, deterministic sparse block shift-invert at/above);
real signed weights -> W-self-adjointness verified, Krein inertia of
Phi^dagger W Phi recorded and normalized to diag(I_p, -I_q); complex
weights -> matched biorthogonal right/left subspaces with
Psi^dagger W Phi = I.  Bands are grouped by a relative gap rule and every
band is reported with its projector and certificate; a closing gap yields
an uncertified band, never a different identity.  The detector never
requests rank three and no eigenvalue threshold is a Betti oracle.

Read-only observable: never calls a solver on the spacetime, never mutates
it, and nothing here enters any emergence objective.)doc")
      .def(py::init([](std::shared_ptr<Spacetime> st,
                       const SpectralFiberConfig &cfg,
                       const py::object &weights) {
             const auto convention =
                 weights.is_none()
                     ? tessera::cobordism::HodgeLaplacian::
                           defaultWeightConvention()
                     : weights
                           .cast<tessera::cobordism::HodgeLaplacian::
                                     WeightConvention>();
             return SpectralFiberTracker(std::move(st), cfg, convention);
           }),
           py::arg("spacetime"), py::arg("config") = SpectralFiberConfig{},
           py::arg("weights") = py::none(),
           "Bind to the spacetime to read; weights=None follows the "
           "process-wide HodgeLaplacian.defaultWeightConvention() at call "
           "time.")
      .def("config", &SpectralFiberTracker::config,
           py::return_value_policy::copy)
      .def("weightConvention", &SpectralFiberTracker::weightConvention)
      .def("enumerateBands",
           [](const SpectralFiberTracker &self,
              const std::vector<std::uint64_t> &support, int degree) {
             py::gil_scoped_release release;
             return self.enumerateBands(support, degree);
           },
           py::arg("support"), py::arg("degree"),
           "Enumerate the bands of one component (vertex-id support) at "
           "one form degree.")
      .def("enumerateOnComponents",
           [](const SpectralFiberTracker &self,
              const std::vector<ComponentRead> &components) {
             py::gil_scoped_release release;
             return self.enumerateOnComponents(components);
           },
           py::arg("components"),
           "Enumerate every configured degree on every #765 component.")
      .def("enumerateBandsCached",
           [](const SpectralFiberTracker &self,
              tessera::cobordism::AnalyticCache &cache,
              const std::vector<std::uint64_t> &support, int degree) {
             py::gil_scoped_release release;
             return self.enumerateBandsCached(cache, support, degree);
           },
           py::arg("cache"), py::arg("support"), py::arg("degree"),
           "enumerateBands through the #764 AnalyticCache contract "
           "(touched-star invalidation; served while the component is "
           "untouched).")
      .def_static("acceptedWindows", &SpectralFiberTracker::acceptedWindows,
                  py::arg("reads"),
                  "The accepted bands' frequency windows as plain data for "
                  "the response consumer (#768).")
      .def_static("matchFibers", &SpectralFiberTracker::matchFibers,
                  py::arg("fromFibers"), py::arg("toFibers"),
                  py::arg("overlapThreshold") = 0.5,
                  "Track fibers across frames/resolutions by principal "
                  "angles and component overlap.")
      .def_readonly_static("CACHE_KIND", &SpectralFiberTracker::kCacheKind);

  // ========================================
  // ModularityOptimizer
  // ========================================
  py::class_<ModularityMeasurement>(m, "ModularityMeasurement",
      R"doc(One recorded point on the (Q, D_S) trajectory.

Mirrors examples/modularity.py:Measurement.)doc")
      .def_readonly("Q", &ModularityMeasurement::Q)
      .def_readonly("dsSmall", &ModularityMeasurement::dsSmall)
      .def_readonly("dsLarge", &ModularityMeasurement::dsLarge)
      .def_readonly("nVertices", &ModularityMeasurement::nVertices)
      .def_readonly("nEdges", &ModularityMeasurement::nEdges)
      .def_readonly("nSimplices", &ModularityMeasurement::nSimplices)
      .def_readonly("iter", &ModularityMeasurement::iter)
      .def_readonly("direction", &ModularityMeasurement::direction);

  py::class_<ModularityOptimizerConfig>(m, "ModularityOptimizerConfig")
      .def(py::init<>())
      .def_readwrite("targetDq", &ModularityOptimizerConfig::targetDq)
      .def_readwrite("maxIterations",
                     &ModularityOptimizerConfig::maxIterations)
      .def_readwrite("nDiffusionWalks",
                     &ModularityOptimizerConfig::nDiffusionWalks)
      .def_readwrite("maxSigma", &ModularityOptimizerConfig::maxSigma)
      .def_readwrite("negativeRetryMax",
                     &ModularityOptimizerConfig::negativeRetryMax)
      .def_readwrite("epsilonQMax",
                     &ModularityOptimizerConfig::epsilonQMax)
      .def_readwrite("krylovDim", &ModularityOptimizerConfig::krylovDim)
      .def_readwrite("targetNModules",
                     &ModularityOptimizerConfig::targetNModules);

  py::class_<ModularityOptimizer>(m, "ModularityOptimizer",
      R"doc(Modularity sweep on a CDT spacetime, driven by transactional
Pachner moves with Q-direction acceptance.

Each iteration:
  1. Picks a random move type from {add, remove, flip, iflip, shift}.
  2. Calls cdt.proposeXxx() — if no eligible target, tries another type.
  3. Snapshots Q on the spacetime 1-skeleton.
  4. Calls move.apply() to commit the move.
  5. Computes new Q.  If direction matches, keeps the move; else
     calls move.rollback().
  6. If Q crossed the next target_dq threshold, builds the dual graph
     and measures D_S.

See docs/source/modularity-plan.md for the design rationale.)doc")
      .def(py::init<ModularityOptimizerConfig, std::uint64_t>(),
           py::arg("config"), py::arg("seed") = 0)
      .def("sweep",
           [](ModularityOptimizer &self, CDT &cdt,
              const std::string &direction, py::object progress) {
             ModularityOptimizer::ProgressCallback cb = nullptr;
             if (!progress.is_none()) {
               cb = [&progress](int it, int maxIt, double q,
                                std::size_t n) {
                 py::gil_scoped_acquire acquire;
                 progress(it, maxIt, q, n);
               };
             }
             py::gil_scoped_release release;
             return self.sweep(cdt, direction, cb);
           },
           py::arg("cdt"), py::arg("direction"),
           py::arg("progress") = py::none(),
           "Drive `cdt` to walk Q in direction ('up' or 'down'). "
           "Returns list of ModularityMeasurement.")
      .def("getNAccepted", &ModularityOptimizer::getNAccepted,
           "Number of moves applied + kept (since the last sweep).")
      .def("getNRolledBack", &ModularityOptimizer::getNRolledBack,
           "Number of moves applied then rolled back.")
      .def("getNNoMove", &ModularityOptimizer::getNNoMove,
           "Number of iterations with no eligible move.")
      .def("getNMeasurements", &ModularityOptimizer::getNMeasurements,
           "Number of D_S measurements taken.")
      .def("discoverComponents",
           [](const ModularityOptimizer &self,
              const std::shared_ptr<Spacetime> &st,
              const PersistentModularityConfig &cfg,
              PersistentModularity::WeightMap map) {
             py::gil_scoped_release release;
             return self.discoverComponents(*st, cfg, map);
           },
           py::arg("spacetime"), py::arg("config"),
           py::arg("map") = PersistentModularity::WeightMap::ExpNegAbsLength,
           R"doc(Label-free discovery of persistent modular components on the
CURRENT spacetime one-skeleton (ticket #765).  Read-only: never mutates the
spacetime and never proposes moves.  Builds the nonnegative similarity graph
under ``map`` and runs PersistentModularity.scanResolutions(config).  A
heuristic proposal generator — blind to signed/complex Hodge weights, never
part of the emergence objective, and never a veto over a certified fiber.)doc");
  // ========================================
  // VolumeProfile
  // ========================================
  py::class_<VolumeProfile, std::shared_ptr<VolumeProfile> >(m, "VolumeProfile",
      R"doc(Observable measuring the spatial volume profile N(t).

Counts top simplices per time slice.  Can accumulate measurements over
multiple configurations for averaging.)doc")
      .def(py::init<>())
      .def("compute", &VolumeProfile::compute, py::arg("spacetime"),
           "Compute the volume profile for the current configuration.")
      .def("getProfile", &VolumeProfile::getProfile,
           "Return the most recent volume profile as a list.")
      .def("getAverageProfile", &VolumeProfile::getAverageProfile,
           "Return the time-averaged volume profile (over all measure() calls).")
      .def("getCenteredAverageProfile",
           &VolumeProfile::getCenteredAverageProfile,
           py::arg("subtractStalk") = false,
           py::arg("normalizePeak") = false,
           "Peak-centered average of all measure() calls (see centeredAverage).")
      .def_static("centeredAverage", &VolumeProfile::centeredAverage,
           py::arg("profiles"),
           py::arg("subtractStalk") = false,
           py::arg("normalizePeak") = false,
           R"doc(Peak-centered average of a set of volume profiles.

Each profile is zero-padded to the longest length and circularly rolled so
its peak sits at T//2 before the bin-wise mean is taken, preventing the de
Sitter blob from smearing when its position fluctuates along the chain
(Ambjorn, Jurkiewicz, Loll, 2005).

Args:
    profiles: Per-configuration volume profiles (lists of counts).
    subtractStalk: Subtract each padded profile's minimum before centering.
    normalizePeak: Rescale the result so its peak equals 1.)doc")
      .def("measure", &VolumeProfile::measure, py::arg("spacetime"),
           "Compute and accumulate a volume profile measurement for averaging.")
      .def("reset", &VolumeProfile::reset,
           "Reset the accumulated measurements.");
  // ========================================
  // Hodge spectral Observables (#95): scalars over cobordism::HodgeLaplacian
  // ========================================
  py::class_<SpectralGap, std::shared_ptr<SpectralGap>>(m, "SpectralGap",
      "Observable: first spectral gap lambda_1 - lambda_0 of the Hermitian-"
      "weighted Hodge Laplacian (cobordism::HodgeLaplacian, k=0). The gauge-"
      "invariant interference signature: on the triangle it collapses from 3 to "
      "0 at flux Phi=pi.")
      .def(py::init<>())
      .def("compute", &SpectralGap::compute, py::arg("spacetime"));
  py::class_<HarmonicDimension, std::shared_ptr<HarmonicDimension>>(m, "HarmonicDimension",
      "Observable: dim ker L_0 (harmonic zero-mode count) of the Hermitian-"
      "weighted Hodge Laplacian. Equals b_0 at zero flux; a nonzero U(1) flux "
      "lifts the zero-mode, dropping it below the topological ChainComplex count.")
      .def(py::init<>())
      .def("compute", &HarmonicDimension::compute, py::arg("spacetime"));
  // ========================================
  // WilsonLoop
  // ========================================
  py::enum_<WilsonMode>(m, "WilsonMode",
      "Evaluation mode for Wilson loops.")
      .value("COMBINATORIAL", WilsonMode::COMBINATORIAL,
             "Dual-graph topology only (loop length, enclosed hinges).")
      .value("DEFICIT_ANGLE", WilsonMode::DEFICIT_ANGLE,
             "Deficit-angle based: W = ((d-2)+2cos(epsilon))/d.")
      .value("CAUSAL", WilsonMode::CAUSAL,
             "CDT causal orientation changes around the loop.")
      .value("U1_CONNECTION", WilsonMode::U1_CONNECTION,
             "U(1) connection holonomy: oriented sum of Edge.phase around a "
             "1-skeleton vertex cycle, reduced mod 2*pi. The Wilson-loop view "
             "of the Stage-1 cobordism.HodgeLaplacian cycle flux.");

  py::enum_<LoopType>(m, "LoopType",
      "Which loop-shape generator to use.")
      .value("HINGE", LoopType::HINGE,
             "Elementary loop around a (d-2)-simplex.")
      .value("DUAL_LATTICE", LoopType::DUAL_LATTICE,
             "BFS-discovered loop of a target size.")
      .value("GEODESIC", LoopType::GEODESIC,
             "Shortest cycle through a start simplex.");

  py::class_<LoopPath>(m, "LoopPath",
      "A closed path through the dual graph (sequence of top-simplices).")
      .def_readonly("simplices", &LoopPath::simplices,
                    py::return_value_policy::reference)
      .def_readonly("facets", &LoopPath::facets,
                    py::return_value_policy::reference)
      .def("__len__", [](const LoopPath &lp) { return lp.simplices.size(); });

  py::class_<WilsonResult>(m, "WilsonResult",
      "Result of evaluating a Wilson loop.")
      .def_readonly("value", &WilsonResult::value,
                    "Primary scalar value.")
      .def_readonly("loopSize", &WilsonResult::loopSize,
                    "Number of simplices in the loop.")
      .def_readonly("enclosedHinges", &WilsonResult::enclosedHinges,
                    "Hinges enclosed by the loop.")
      .def_readonly("contractible", &WilsonResult::contractible,
                    "Whether the loop is contractible.")
      .def_readonly("causalWindingNumber", &WilsonResult::causalWindingNumber,
                    "Net time-orientation changes (causal mode).");

  py::class_<WilsonLoop, std::shared_ptr<WilsonLoop>>(m, "WilsonLoop",
      R"doc(Wilson loop observable on a triangulated spacetime.

A Wilson loop is the trace of a parallel-transport operator around a
closed path. On a curved triangulation without an explicit gauge field
tessera computes the Levi-Civita holonomy analogue: closed walks on the
dual graph (top-simplices as nodes, shared facets as edges), with the
loop value determined by the deficit angles of enclosed hinges.

Four evaluation modes:

* ``COMBINATORIAL``  — dual-graph topology only. ``value`` is the loop
  length; ``enclosedHinges`` counts hinges contained in every loop
  simplex; ``contractible`` is True iff ``enclosedHinges == 0``.
* ``DEFICIT_ANGLE``  — Regge-curvature holonomy. For a hinge loop
  enclosing one hinge h:
      W = ((d-2) + 2 cos(eps_h)) / d
  For multi-hinge loops the U(1) approximation is used:
      W = product_{h in enclosed} cos(eps_h).
  W = 1 corresponds to a flat loop; deviation from 1 measures local
  curvature.
* ``CAUSAL``  — CDT causal-orientation winding. ``causalWindingNumber``
  is the signed net change in foliation index around the loop; non-
  zero values mark loops that cross a CDT slice boundary.
* ``U1_CONNECTION``  — U(1) connection holonomy. The oriented sum of the
  ``Edge.phase`` carried on the primal 1-skeleton around a closed vertex
  cycle (``+phase`` along the stored source->target orientation,
  ``-phase`` reversed), reduced mod 2*pi. Evaluated via
  ``evaluateU1Connection(cycle)`` (the connection is a primal-edge, not a
  dual-graph, quantity); ``value`` carries the holonomy. This is the
  Wilson-loop view of the Stage-1 ``cobordism.HodgeLaplacian`` cycle flux.

Three loop-shape generators:

* ``hingeLoop(h)``        — cyclically ordered loop of top-simplices
                            around the (d-2)-simplex ``h``. Encloses
                            exactly one hinge (``h``), so the
                            DEFICIT_ANGLE formula above is exact.
* ``dualLatticeLoop(start, L)``  — BFS-discovered loop of approximately
                            ``L`` simplices through ``start``. Suitable
                            for population sweeps at fixed loop scale.
* ``geodesicLoop(start)``  — shortest cycle through ``start`` in the
                            dual graph (girth at that simplex).

Measurement bookkeeping: ``measure()`` and ``measureAllHinges()`` append
``WilsonResult`` entries to an internal list. ``getMeasurements()``
returns the accumulated list; ``getAverageBySize()`` aggregates by loop
length (the standard form for Creutz-ratio-style analyses);
``reset()`` clears.

See ``docs/source/wilson_loops.md`` for an end-to-end tutorial with
curvature-scan, contractibility-statistics, and causal-winding
examples.
)doc")
      .def(py::init<std::shared_ptr<Spacetime>>(), py::arg("spacetime"),
           "Construct a Wilson-loop calculator bound to a Spacetime.")
      .def("evaluate", &WilsonLoop::evaluate,
           py::arg("loop"), py::arg("mode"),
           R"doc(Evaluate the Wilson loop in the given mode.

Dispatches to ``evaluateCombinatorial``, ``evaluateDeficitAngle``, or
``evaluateCausal`` depending on ``mode``.
)doc")
      .def("evaluateCombinatorial", &WilsonLoop::evaluateCombinatorial,
           py::arg("loop"),
           R"doc(Evaluate using dual-graph topology only.

Returns a ``WilsonResult`` with ``value = loopSize``,
``enclosedHinges`` = count of hinges shared by every loop simplex, and
``contractible`` = True iff no hinge is enclosed.
)doc")
      .def("evaluateDeficitAngle", &WilsonLoop::evaluateDeficitAngle,
           py::arg("loop"),
           R"doc(Evaluate using Regge deficit angles.

For a hinge loop (exactly one enclosed hinge h):
    W = ((d - 2) + 2 cos(eps_h)) / d.
For multi-hinge loops the U(1) approximation:
    W = product_{h in enclosed} cos(eps_h).
``value`` carries W; ``enclosedHinges`` carries the count.
)doc")
      .def("evaluateCausal", &WilsonLoop::evaluateCausal,
           py::arg("loop"),
           R"doc(Evaluate using CDT causal-orientation changes.

Walks the loop and accumulates a signed winding count from the
final-time stamps of consecutive simplices. ``causalWindingNumber`` is
the net winding; ``value`` carries the same number as a double.
)doc")
      .def("evaluateU1Connection", &WilsonLoop::evaluateU1Connection,
           py::arg("cycle"),
           R"doc(U(1) connection holonomy around a closed vertex cycle.

``cycle`` is an ordered list of vertices on the primal 1-skeleton whose
consecutive pairs (with wrap-around) are joined by edges. Accumulates each
edge's ``phase`` along its stored source->target orientation (``+phase``
forward, ``-phase`` reversed) and returns the total reduced into the
principal interval ``(-pi, pi]`` in ``value``; ``loopSize`` is the number of
edges. Returns an empty result (``loopSize == 0``) for a degenerate (fewer
than two vertices) or open (a consecutive pair with no joining edge) cycle.

This is the Wilson-loop counterpart of the Stage-1 cycle flux carried by the
Hermitian-weighted ``cobordism.HodgeLaplacian`` — the same oriented phase
sum. Restricted to phases in ``{0, pi}`` the holonomy lands in ``{0, pi}``
and reproduces the Z2 flux.
)doc")
      .def("hingeLoop", &WilsonLoop::hingeLoop,
           py::arg("hinge"),
           R"doc(Loop of top-simplices around a hinge, ordered cyclically.

Encloses exactly one hinge (the input). This is the natural loop for
``DEFICIT_ANGLE`` mode and is what ``measureAllHinges`` uses internally.
)doc")
      .def("dualLatticeLoop", &WilsonLoop::dualLatticeLoop,
           py::arg("start"), py::arg("targetLength"),
           R"doc(BFS-discovered loop of approximately ``targetLength`` simplices.

Not guaranteed to be exactly ``targetLength`` — the BFS may overshoot
or return a shorter loop if local connectivity doesn't permit closing
at the target size. Suitable for population-level scans at a fixed
loop scale (analogous to specifying Wilson-loop side length in lattice
gauge theory).
)doc")
      .def("geodesicLoop", &WilsonLoop::geodesicLoop,
           py::arg("start"),
           R"doc(Shortest cycle through ``start`` in the dual graph.

The cycle length is the local girth of the dual graph at ``start``.
Useful when you want the natural shortest loop without specifying a
target size.
)doc")
      .def("measure", &WilsonLoop::measure,
           py::arg("loop"), py::arg("mode"),
           "Evaluate ``loop`` in ``mode`` and append the result to the "
           "internal measurement list.")
      .def("measureAllHinges", &WilsonLoop::measureAllHinges,
           py::arg("mode"),
           R"doc(Walk every (d-2)-simplex of the spacetime, generate its hinge
loop, and record the evaluation in ``mode``. Skips degenerate hinges
whose loop has fewer than 2 distinct simplices. Bulk shortcut for a
curvature scan.
)doc")
      .def("reset", &WilsonLoop::reset,
           "Clear all accumulated measurements.")
      .def("getMeasurements", &WilsonLoop::getMeasurements,
           "Return the full list of accumulated ``WilsonResult`` entries.")
      .def("getAverageBySize", &WilsonLoop::getAverageBySize,
           R"doc(Mean ``value`` grouped by loop size, as a ``{size: mean}`` dict.

The standard form for Creutz-ratio-style analyses: fix loop size L,
read off the population-averaged Wilson value at that scale.
)doc");

  // ==========================================================================
  // Emergent-proton readout battery (#593): pure readers over a live complex,
  // the loader/transform layer, and the GAUGE/RELABEL gates.
  // ==========================================================================

  // ---- LiveComplex: the loader / transform layer (outside the readers) ----
  py::class_<LiveComplex::Relabeled>(m, "Relabeled",
      "A relabeled rebuild: the live relabeled complex + the vertex-id "
      "permutation (original id -> relabeled id).")
      .def_readonly("spacetime", &LiveComplex::Relabeled::spacetime)
      .def_readonly("vertex_map", &LiveComplex::Relabeled::vertexMap);

  py::class_<LiveComplex>(m, "LiveComplex",
      R"doc(The loader / transform layer OUTSIDE the pure readers: LOAD a saved
combinatorial + metric description back into a live, skeleton-complete Spacetime,
and produce a relabeled copy for the RELABEL gate. Never builds a spacetime of
its own or re-runs the emergent dynamics (those live in Proton / ProtonIngredients
/ MultiCobordism) — only reads a recorded geometry back through the canonical
``Spacetime.fromCells``, completing the facet skeleton with ``materializeFacets``.)doc")
      .def_static("load", &LiveComplex::load, py::arg("cells"),
                  py::arg("squared_lengths"), py::arg("vertex_times"),
                  py::arg("dimensions"),
                  "Load a live skeleton-complete complex from cells + per-edge "
                  "complex squared lengths + vertex times (schema-1 dump "
                  "rehydration).")
      .def_static("subcomplex", &LiveComplex::subcomplex, py::arg("cells"),
                  py::arg("dimensions"),
                  "Load a uniform-metric sub-complex from already-selected "
                  "ambient cells (the block-residual carry diagnostic).")
      .def_static("relabel", &LiveComplex::relabel, py::arg("spacetime"),
                  py::arg("seed"),
                  "A relabeled rebuild under a deterministic vertex-id "
                  "permutation (the RELABEL-gate transform).");

  // ---- RegisterContext: the validated read context (pure reader) ----
  py::class_<RegisterContext, std::shared_ptr<RegisterContext>>(
      m, "RegisterContext",
      R"doc(The one validated read context every emergent-proton observable
measures: a LIVE, already-built complex, its emergent register holes, the
induced-orientation signs, and the shared per-complex caches. A pure reader — it
never builds, solves, or materializes anything.)doc")
      .def(py::init([](std::shared_ptr<Spacetime> st, int count, int degree,
                       std::vector<std::complex<double>> target) {
             auto ctx = std::make_shared<RegisterContext>(
                 std::move(st), count, degree, std::move(target));
             emitSelectionWarning(*ctx);
             return ctx;
           }),
           py::arg("spacetime"), py::arg("count") = 3, py::arg("degree") = 3,
           py::arg("target") = ::tessera::cobordism::Proton::singlet())
      .def(py::init([](std::shared_ptr<Spacetime> st,
                       std::vector<std::vector<std::uint64_t>> holes, int count,
                       int degree, std::vector<std::complex<double>> target) {
             auto ctx = std::make_shared<RegisterContext>(
                 std::move(st), holes, count, degree, std::move(target));
             emitSelectionWarning(*ctx);
             return ctx;
           }),
           py::arg("spacetime"), py::arg("holes"), py::arg("count"),
           py::arg("degree"), py::arg("target"))
      .def("spacetime", &RegisterContext::spacetime)
      .def("degree", &RegisterContext::degree)
      .def("target", &RegisterContext::target)
      .def("holes", &RegisterContext::holes)
      .def("dropped_holes", &RegisterContext::droppedHoles)
      .def("holes_used", &RegisterContext::holesUsed)
      .def("holes_total", &RegisterContext::holesTotal)
      .def("bK", &RegisterContext::bK)
      .def("betti", &RegisterContext::betti)
      .def("holes_vs_betti_divergent",
           &RegisterContext::holesVsBettiDivergent)
      .def("dimensions", &RegisterContext::dimensions)
      .def("top_cell_count", &RegisterContext::topCellCount)
      .def("causal_content", &RegisterContext::causalContent)
      .def("selection_warning", &RegisterContext::selectionWarning)
      .def("gauged", &RegisterContext::gauged, py::arg("theta"))
      .def("summary", [](const RegisterContext &ctx) {
        py::dict d;
        d["degree"] = ctx.degree();
        d["dimensions"] = ctx.dimensions();
        d["n_top_cells"] = ctx.topCellCount();
        d["holes_used"] = ctx.holesUsed();
        d["holes_total"] = ctx.holesTotal();
        py::list dropped;
        for (const auto &h : ctx.droppedHoles()) {
          dropped.append(py::cast(h));
        }
        d["dropped_holes"] = dropped;
        d["b3"] = ctx.bK();
        d["betti"] = py::cast(ctx.betti());
        d["holes_vs_b3_divergent"] = ctx.holesVsBettiDivergent();
        d["causal_content"] = ctx.causalContent();
        return d;
      });

  // ---- the observable base (pure reader) ----
  py::class_<RegisterObservable, std::shared_ptr<RegisterObservable>>(
      m, "RegisterObservable",
      "Base for the emergent-proton readouts: a pure post-hoc reader over a "
      "RegisterContext.")
      .def("record_key", &RegisterObservable::recordKey)
      .def("gate_tol", &RegisterObservable::gateTol)
      .def("min_holes", &RegisterObservable::minHoles)
      .def("required_dimensions", &RegisterObservable::requiredDimensions)
      .def("needs_provenance", &RegisterObservable::needsProvenance)
      .def("has_provenance", &RegisterObservable::hasProvenance)
      .def("needs_causal_content", &RegisterObservable::needsCausalContent)
      .def("skip_reason", &RegisterObservable::skipReason, py::arg("ctx"))
      .def(
          "record",
          [](const RegisterObservable &o, const RegisterContext &ctx) {
            return recordToPython(o.record(ctx));
          },
          py::arg("ctx"))
      .def(
          "compute",
          [](const RegisterObservable &o, const RegisterContext &ctx) {
            return o.compute(ctx);
          },
          py::arg("ctx"));

  py::class_<SingletResidual, RegisterObservable,
             std::shared_ptr<SingletResidual>>(m, "SingletResidual",
      "The #574 whole-complex singlet diagnostic (headline = singlet r_state).")
      .def(py::init<>())
      .def("conjugate_residual", &SingletResidual::conjugateResidual,
           py::arg("ctx"));

  py::class_<BlockResiduals::Block>(m, "Block",
      "One provenance block: a label, its vertex region, and its register target.")
      .def(py::init([](std::string label, std::vector<std::uint64_t> vertices,
                       std::vector<std::complex<double>> target) {
             return BlockResiduals::Block{std::move(label), std::move(vertices),
                                          std::move(target)};
           }),
           py::arg("label"), py::arg("vertices"), py::arg("target"))
      .def_readwrite("label", &BlockResiduals::Block::label)
      .def_readwrite("vertices", &BlockResiduals::Block::vertices)
      .def_readwrite("target", &BlockResiduals::Block::target);

  py::class_<BlockResiduals, RegisterObservable,
             std::shared_ptr<BlockResiduals>>(m, "BlockResiduals",
      "The #574 per-output-block carry residuals (blocks are ctor provenance).")
      .def(py::init<std::vector<BlockResiduals::Block>>(), py::arg("blocks"));

  // The mass/radius reader structs (typed accessors).
  py::class_<InteriorHinges::Masses>(m, "EmergentMasses")
      .def_readonly("m_shell", &InteriorHinges::Masses::mShell)
      .def_readonly("m_sum", &InteriorHinges::Masses::mSum)
      .def_readonly("m_action", &InteriorHinges::Masses::mAction)
      .def_readonly("max_abs_im", &InteriorHinges::Masses::maxAbsIm)
      .def_readonly("n_im_nonzero", &InteriorHinges::Masses::nImNonzero)
      .def_readonly("empty", &InteriorHinges::Masses::empty);
  py::class_<InteriorHinges::Radii>(m, "EmergentRadii")
      .def_readonly("v_dual", &InteriorHinges::Radii::vDual)
      .def_readonly("v_primal", &InteriorHinges::Radii::vPrimal)
      .def_readonly("n_interior_vertices",
                    &InteriorHinges::Radii::nInteriorVertices)
      .def_readonly("r_dual", &InteriorHinges::Radii::rDual)
      .def_readonly("r_primal", &InteriorHinges::Radii::rPrimal);
  py::class_<InteriorHinges::Localization>(m, "EmergentLocalization")
      .def_readonly("pr", &InteriorHinges::Localization::pr)
      .def_readonly("concentration", &InteriorHinges::Localization::concentration)
      .def_readonly("mean_re", &InteriorHinges::Localization::meanRe)
      .def_readonly("std_re", &InteriorHinges::Localization::stdRe)
      .def_readonly("std_over_mean",
                    &InteriorHinges::Localization::stdOverMean)
      .def_readonly("rms_shell_radius",
                    &InteriorHinges::Localization::rmsShellRadius)
      .def_readonly("frac_within_shell1",
                    &InteriorHinges::Localization::fracWithinShell1)
      .def_readonly("empty", &InteriorHinges::Localization::empty);

  py::class_<EmergentMass, RegisterObservable, std::shared_ptr<EmergentMass>>(
      m, "EmergentMass",
      "The #575 mass half on the relaxed 4D interior (headline = m_shell).")
      .def(py::init<>())
      .def("masses", &EmergentMass::masses, py::arg("ctx"))
      .def("localization", &EmergentMass::localization, py::arg("ctx"));

  py::class_<EmergentRadius, RegisterObservable,
             std::shared_ptr<EmergentRadius>>(m, "EmergentRadius",
      "The #575 radius half on the relaxed 4D interior (headline = r_dual).")
      .def(py::init<>())
      .def("radii", &EmergentRadius::radii, py::arg("ctx"));

  py::class_<PairLoopFlavor::JointRead>(m, "PairLoopJointRead")
      .def_readonly("sigma", &PairLoopFlavor::JointRead::sigma)
      .def_readonly("r_u", &PairLoopFlavor::JointRead::rU)
      .def_readonly("w", &PairLoopFlavor::JointRead::w)
      .def_readonly("q", &PairLoopFlavor::JointRead::q)
      .def_readonly("loop_w", &PairLoopFlavor::JointRead::loopW)
      .def_readonly("loop_q", &PairLoopFlavor::JointRead::loopQ)
      .def_readonly("dual_residual", &PairLoopFlavor::JointRead::dualResidual);
  py::class_<PairLoopFlavor::Verdict>(m, "PairLoopVerdict")
      .def_readonly("odd_loop", &PairLoopFlavor::Verdict::oddLoop)
      .def_readonly("dual_hole", &PairLoopFlavor::Verdict::dualHole)
      .def_readonly("rho", &PairLoopFlavor::Verdict::rho)
      .def_readonly("multiplicity_2_1", &PairLoopFlavor::Verdict::multiplicity21)
      .def_readonly("odd_is_diquark_loop",
                    &PairLoopFlavor::Verdict::oddIsDiquarkLoop);

  py::class_<PairLoopFlavor, RegisterObservable,
             std::shared_ptr<PairLoopFlavor>>(m, "PairLoopFlavor",
      "The #561/#576 pair-loop dual-basis flavor read (headline = rho).")
      .def(py::init<>())
      .def(py::init([](std::pair<int, int> diquark) {
             return std::make_shared<PairLoopFlavor>(diquark);
           }),
           py::arg("diquark_pair"))
      .def("joint_read", &PairLoopFlavor::jointRead, py::arg("ctx"))
      .def("evaluate_criteria", &PairLoopFlavor::evaluateCriteria,
           py::arg("read"))
      .def_static("odd_one_out", &PairLoopFlavor::oddOneOut, py::arg("loop_q"))
      .def_static("complement_hole", &PairLoopFlavor::complementHole,
                  py::arg("pair"));
  m.attr("PairLoopFlavor").attr("RHO_MAX") = PairLoopFlavor::RHO_MAX;

  // ---- the self-test probes ----
  py::class_<LabelLeakProbe, RegisterObservable,
             std::shared_ptr<LabelLeakProbe>>(m, "LabelLeakProbe",
      "A deliberately label-dependent probe (RELABEL must flag it).")
      .def(py::init<>());
  py::class_<GaugeLeakProbe, RegisterObservable,
             std::shared_ptr<GaugeLeakProbe>>(m, "GaugeLeakProbe",
      "A deliberately gauge-dependent probe (GAUGE must flag it).")
      .def(py::init<>());

  // ---- the GAUGE/RELABEL gate harness ----
  py::class_<ObservableGates::GateResult>(m, "GateResult")
      .def_readonly("gauge_delta", &ObservableGates::GateResult::gaugeDelta)
      .def_readonly("relabel_delta", &ObservableGates::GateResult::relabelDelta)
      .def_readonly("gate_tol", &ObservableGates::GateResult::gateTol)
      .def_readonly("gauge_ok", &ObservableGates::GateResult::gaugeOk)
      .def_readonly("relabel_ok", &ObservableGates::GateResult::relabelOk);

  py::class_<ObservableGates>(m, "ObservableGates",
      "The GAUGE/RELABEL gate harness — post-hoc validation, never a loop "
      "condition.")
      .def_static("gauge_delta", &ObservableGates::gaugeDelta,
                  py::arg("observable"), py::arg("ctx"))
      .def_static("relabel_delta", &ObservableGates::relabelDelta,
                  py::arg("observable"), py::arg("ctx"))
      .def_static("evaluate", &ObservableGates::evaluate, py::arg("observable"),
                  py::arg("ctx"))
      .def_static("self_test", &ObservableGates::selfTest, py::arg("ctx"))
      .def_static(
          "report_delta",
          [](const py::object &a, const py::object &b) {
            return Record::reportDelta(pythonToRecord(a), pythonToRecord(b));
          },
          py::arg("a"), py::arg("b"),
          "The max-abs delta over every numeric leaf of two records (the gate "
          "metric).");
  m.attr("ObservableGates").attr("GAUGE_THETA") = ObservableGates::GAUGE_THETA;
  m.attr("ObservableGates").attr("GATE_SEED") =
      py::int_(ObservableGates::GATE_SEED);

  // ========================================
  // DualVolumeSigns (#605)
  // ========================================
  py::class_<DualVolumeSigns::DimensionReport>(m, "DualVolumeDimensionReport",
      "Per-dimension counts from the diagonal DEC Hodge star sign audit.")
      .def_readonly("dimension", &DualVolumeSigns::DimensionReport::dimension)
      .def_readonly("n_simplices", &DualVolumeSigns::DimensionReport::nSimplices)
      .def_readonly("n_negative_dual_volume",
                    &DualVolumeSigns::DimensionReport::nNegativeDualVolume)
      .def_readonly("n_degenerate_volume",
                    &DualVolumeSigns::DimensionReport::nDegenerateVolume)
      .def_readonly("n_circumcenter_outside",
                    &DualVolumeSigns::DimensionReport::nCircumcenterOutside)
      .def_readonly("n_negative_circumradius",
                    &DualVolumeSigns::DimensionReport::nNegativeCircumradius)
      .def_readonly("n_negative_star",
                    &DualVolumeSigns::DimensionReport::nNegativeStar)
      .def_readonly("n_all_spacelike",
                    &DualVolumeSigns::DimensionReport::nAllSpacelike)
      .def_readonly("n_negative_star_all_spacelike",
                    &DualVolumeSigns::DimensionReport::nNegativeStarAllSpacelike)
      .def_readonly("n_mixed_signature",
                    &DualVolumeSigns::DimensionReport::nMixedSignature)
      .def_readonly(
          "n_negative_star_mixed_signature",
          &DualVolumeSigns::DimensionReport::nNegativeStarMixedSignature)
      .def_readonly("min_star_ratio",
                    &DualVolumeSigns::DimensionReport::minStarRatio)
      .def_readonly("max_star_ratio",
                    &DualVolumeSigns::DimensionReport::maxStarRatio)
      .def_readonly("mean_star_ratio",
                    &DualVolumeSigns::DimensionReport::meanStarRatio);

  py::class_<DualVolumeSigns::Report>(m, "DualVolumeReport",
      "The full diagonal DEC Hodge star sign audit, one entry per simplex "
      "dimension.")
      .def_readonly("dimensions", &DualVolumeSigns::Report::dimensions)
      .def_readonly("n_simplices", &DualVolumeSigns::Report::nSimplices)
      .def_readonly("n_negative_star", &DualVolumeSigns::Report::nNegativeStar);

  py::class_<DualVolumeSigns, std::shared_ptr<DualVolumeSigns> >(
      m, "DualVolumeSigns",
      R"doc(Read-only audit of the sign of the diagonal DEC Hodge star.

The diagonal Discrete Exterior Calculus Hodge star assigns each k-simplex the
scalar ratio |*sigma| / |sigma|, the signed circumcentric dual cell content over
the simplex's own signed content. A Maxwell-type or gauge term discretised with
DEC carries its whole metric dependence in that ratio, so a negative entry costs
positive-definiteness of the Hodge Laplacian and breaks the sign structure a
self-dual / anti-self-dual split of a 2-cochain relies on.

The audit separates the two causes of a negative ratio. A circumcenter falling
outside its simplex (a negative barycentric coordinate) is the Riemannian
well-centeredness violation and indicates badly shaped cells. A timelike
circumcenter displacement (negative signed circumradius squared) is reachable
only in Lorentzian signature and is expected rather than defective. Counts are
therefore broken out by all-spacelike versus mixed-signature cells.

Measures only: changes no geometry and enforces nothing.)doc")
      .def(py::init<double>(), py::arg("tolerance") = 1e-12)
      .def("analyze", &DualVolumeSigns::analyze, py::arg("spacetime"),
           "The full per-dimension audit.")
      .def("compute", &DualVolumeSigns::compute, py::arg("spacetime"),
           "Fraction of audited, non-degenerate simplices whose star ratio is "
           "negative. Zero means the diagonal star is positive everywhere.");
}
