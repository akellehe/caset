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
#include "observables/SparseGraph.h"
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
#include "observables/ColorFiber.h"
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

  // ==========================================================================
  // ColorFiber / ColorAnchor (#767): the exact three-edge SU(3) color kernel
  // and the calibrated weighted oriented-triangle anchor.  Pure reads over
  // caller-supplied data; nothing enters the emergence objective.
  // ==========================================================================
  py::class_<OrientedTriangle>(m, "OrientedTriangle",
      R"doc(One oriented 2-simplex descriptor for the anchoring kernel: the
three boundary edge indices in the cyclic order induced by the triangle's
orientation (rows of the caller's frame), with their incidence signs (+1 /
-1).  det A_tau is invariant under cyclic rotation of (edges, signs) and
negates under an odd permutation (the opposite orientation).)doc")
      .def(py::init([](std::array<Eigen::Index, 3> edges,
                       std::array<int, 3> signs) {
             return OrientedTriangle{edges, signs};
           }),
           py::arg("edges"),
           py::arg("signs") = std::array<int, 3>{+1, +1, +1})
      .def_readwrite("edges", &OrientedTriangle::edges)
      .def_readwrite("signs", &OrientedTriangle::signs);

  py::class_<AnchorProfile>(m, "AnchorProfile",
      R"doc(The reported anchor datum -- the PROFILE, not only the score:
the calibrated atlas score a^2 = sum_tau w_tau |det A_tau|^2, the per-
triangle terms, the maximal term, the participation ratio of the term
distribution, the determinant phases with their circular coherence /
dispersion on overlapping oriented triangles (NaN when no determinant is
nonzero -- unknown is never encoded as zero), the per-triangle Krein
signatures (n+, n0, n-) of the restricted weight blocks (reported
separately from the |W_tau|-restricted score), the frame-normalization
residual, the numerically-checked calibration margin, and the pre-declared
convex weighting that produced the score.)doc")
      .def_readonly("score", &AnchorProfile::score)
      .def_readonly("terms", &AnchorProfile::terms)
      .def_readonly("max_term", &AnchorProfile::maxTerm)
      .def_readonly("max_term_index", &AnchorProfile::maxTermIndex)
      .def_readonly("participation_ratio", &AnchorProfile::participationRatio)
      .def_readonly("det_phases", &AnchorProfile::detPhases)
      .def_readonly("phase_coherence", &AnchorProfile::phaseCoherence)
      .def_readonly("phase_dispersion", &AnchorProfile::phaseDispersion)
      .def_readonly("krein_signatures", &AnchorProfile::kreinSignatures)
      .def_readonly("positive_regime", &AnchorProfile::positiveRegime)
      .def_readonly("frame_gram_residual", &AnchorProfile::frameGramResidual)
      .def_readonly("calibration_margin", &AnchorProfile::calibrationMargin)
      .def_readonly("weighting_id", &AnchorProfile::weightingId)
      .def_readonly("weights", &AnchorProfile::weights)
      .def_readonly("certificate", &AnchorProfile::certificate,
          "The #764 tessera.cobordism.Certificate grading the calibrated "
          "score: StructureExact on the diagonal (decoupled) weight path, "
          "CertifiedNumerical on the general Hermitian-matrix path; regime "
          "PositiveSemidefinite / HermitianIndefinite per the Krein read; "
          "residual = max(frame_gram_residual, max(0, calibration_margin)) "
          "against the evaluate gram tolerance.");

  py::class_<ColorFiber::SectorWeights>(m, "SectorWeights",
      "Occupation-sector weights ||P_N psi||^2 of an 8-dimensional Fock "
      "vector over the three edge modes: vacuum (N=0), quark / fundamental "
      "triplet (N=1), anti-triplet / diquark (N=2), top-wedge color singlet "
      "(N=3).  Sector READS only -- never a particle classification.")
      .def_readonly("vacuum", &ColorFiber::SectorWeights::vacuum)
      .def_readonly("quark", &ColorFiber::SectorWeights::quark)
      .def_readonly("anti_triplet", &ColorFiber::SectorWeights::antiTriplet)
      .def_readonly("singlet", &ColorFiber::SectorWeights::singlet);

  py::class_<ColorFiber::OctetRead>(m, "OctetRead",
      "Frobenius split of a 3x3 bilinear under 3 (x) 3bar = 1 (+) 8: "
      "octet = ||M - (tr M / 3) I||_F^2, singlet = |tr M|^2 / 3; their sum "
      "is ||M||_F^2 exactly.")
      .def_readonly("octet", &ColorFiber::OctetRead::octet)
      .def_readonly("singlet", &ColorFiber::OctetRead::singlet);

  py::class_<ColorFiber>(m, "ColorFiber",
      R"doc(The exact three-edge SU(3) color kernel (#767, design spec
section 11 "Algorithm D"): the constant color-sector algebra of three
oriented edge modes, Lambda* C^3 = 1 (+) 3 (+) 3bar (+) 1, layered over the
#766 exterior-algebra primitives (sector projectors and CAR matrices are
delegated to tessera.quantum.ExteriorAlgebra, never reimplemented).

All members are static; Fock operators are dense 8x8 matrices on the
occupation basis n(b) = sum_i b_i 2^i, and the one-occupation (triplet)
sector is spanned by Fock indices (1, 2, 4).  Exact identities (tested to
double round-off): F3^dag F3 = I, |det F3| = 1; lambda_a Hermitian,
traceless, Tr(lambda_a lambda_b) = 2 delta_ab; [E_ij, E_kl] = delta_jk E_il
- delta_il E_kj on both representations; det(gC) = det(C) for g in SU(3);
||v1 ^ v2 ^ v3||^2 = det[<v_i, v_j>].  Pure constants and reads -- no
solver call, no mutation, nothing enters the emergence objective.)doc")
      .def_static("sectorProjector", &ColorFiber::sectorProjector,
                  py::arg("occupation"),
                  "The 8x8 projector onto total occupation N (0..3; zero "
                  "matrix above 3).  Delegates to "
                  "quantum.ExteriorAlgebra.sectorProjector on three modes.")
      .def_static("vacuumProjector", &ColorFiber::vacuumProjector,
                  "Lambda^0: the even vacuum singlet (N=0).")
      .def_static("tripletProjector", &ColorFiber::tripletProjector,
                  "Lambda^1: the odd fundamental color triplet 3 (N=1).")
      .def_static("antiTripletProjector", &ColorFiber::antiTripletProjector,
                  "Lambda^2: the even antisymmetric anti-triplet 3bar (N=2).")
      .def_static("singletProjector", &ColorFiber::singletProjector,
                  "Lambda^3: the odd top-wedge color singlet (N=3).")
      .def_static("creationMatrix", &ColorFiber::creationMatrix,
                  py::arg("mode"), "The 8x8 creation matrix a_i^dag.")
      .def_static("annihilationMatrix", &ColorFiber::annihilationMatrix,
                  py::arg("mode"), "The 8x8 annihilation matrix a_i.")
      .def_static("hoppingMatrix", &ColorFiber::hoppingMatrix,
                  py::arg("i"), py::arg("j"),
                  "The 8x8 bilinear E_ij = a_i^dag a_j (exact gl(3) "
                  "commutation relations on the whole Fock space).")
      .def_static("tripletBasisIndices", &ColorFiber::tripletBasisIndices,
                  "The Fock indices (1, 2, 4) identifying the N=1 sector "
                  "with C^3.")
      .def_static("restrictToTriplet", &ColorFiber::restrictToTriplet,
                  py::arg("op"),
                  "Restrict an 8x8 Fock operator to the one-occupation "
                  "sector as a 3x3 matrix; restrictToTriplet(dGamma(M)) = M "
                  "exactly.")
      .def_static("matrixUnit", &ColorFiber::matrixUnit,
                  py::arg("i"), py::arg("j"),
                  "The 3x3 matrix unit E_ij on the one-occupation sector.")
      .def_static("dGamma", &ColorFiber::dGamma, py::arg("m"),
                  "Second quantization dGamma(M) = sum_ij M_ij a_i^dag a_j "
                  "of a 3x3 one-particle matrix (8x8).")
      .def_static("gellMann", &ColorFiber::gellMann, py::arg("a"),
                  "lambda_a for a in 1..8, assembled from the matrix units "
                  "(lambda_3 = E11-E22, lambda_8 = (E11+E22-2E33)/sqrt(3)).")
      .def_static("adjointOctetProjector", &ColorFiber::adjointOctetProjector,
                  "The 9x9 orthogonal projector onto the traceless "
                  "(adjoint-octet) part of a 3x3 bilinear, acting on "
                  "column-major vec(M).")
      .def_static("tracelessPart", &ColorFiber::tracelessPart, py::arg("m"),
                  "M - (tr M / 3) I: the octet component of a bilinear.")
      .def_static("omega", &ColorFiber::omega,
                  "The primitive cube root of unity as its algebraic value "
                  "(-1 + i sqrt(3))/2 (never exp), so 1 + omega + omega^2 "
                  "cancels exactly in floating point.")
      .def_static("fourierFrame", &ColorFiber::fourierFrame,
                  "The exact unitary Fourier frame F3 with entries "
                  "omega^{jk}/sqrt(3), assembled from the algebraic table "
                  "{1, omega, omega^2} by exponent jk mod 3.")
      .def_static("fourierBasisVector", &ColorFiber::fourierBasisVector,
                  py::arg("k"),
                  "Column k of F3: the Z3 character vector "
                  "(1, omega^k, omega^{2k})/sqrt(3).")
      .def_static("omegaPhaseState", &ColorFiber::omegaPhaseState,
                  "The existing phase pattern (1, omega, omega^2)/sqrt(3), "
                  "identified as ONE color basis vector "
                  "(fourierBasisVector(1)); its cyclic orbit under pointwise "
                  "Z3 powers is the exact orthonormal triad = the columns of "
                  "F3.")
      .def_static("perimeter", &ColorFiber::perimeter, py::arg("z"),
                  "The triangle perimeter sum_i |z_i|^{1/2} of three stored "
                  "complex SQUARED lengths (the L1 geometric datum).")
      .def_static("perimeterNormalized", &ColorFiber::perimeterNormalized,
                  py::arg("z"),
                  "Rescale the squared lengths so the perimeter is one -- a "
                  "GEOMETRIC SCALE GAUGE (L1), never a state normalization.")
      .def_static("hilbertNorm", &ColorFiber::hilbertNorm, py::arg("z"),
                  "The Hilbert L2 norm ||z||_2.")
      .def_static("hilbertNormalized", &ColorFiber::hilbertNormalized,
                  py::arg("z"),
                  "z / ||z||_2 with <c|c> = 1 -- the STATE normalization, "
                  "distinct from the perimeter gauge.")
      .def_static("colorVector", &ColorFiber::colorVector, py::arg("z"),
                  "The color vector from the stored complex squared "
                  "lengths: c = z / ||z||_2.")
      .def_static("colorWedge",
                  py::overload_cast<const Eigen::Matrix3cd&>(
                      &ColorFiber::colorWedge),
                  py::arg("c"),
                  "The color-wedge (singlet) amplitude det C = eps_ijk C_i1 "
                  "C_j2 C_k3; det(gC) = det(C) for g in SU(3).")
      .def_static("colorWedgeColumns",
                  py::overload_cast<const Eigen::Vector3cd&,
                                    const Eigen::Vector3cd&,
                                    const Eigen::Vector3cd&>(
                      &ColorFiber::colorWedge),
                  py::arg("a"), py::arg("b"), py::arg("c"),
                  "colorWedge of three explicit color columns.")
      .def_static("singletGram", &ColorFiber::singletGram, py::arg("c"),
                  "det(C^dag C) = |det C|^2 = ||c1 ^ c2 ^ c3||^2: exactly "
                  "zero for duplicate color modes, exactly one for an "
                  "orthonormal triad.")
      .def_static("isSpecialUnitary", &ColorFiber::isSpecialUnitary,
                  py::arg("g"), py::arg("tol") = 1e-12,
                  "Certify g in SU(3): ||g^dag g - I||_max <= tol and "
                  "|det g - 1| <= tol.")
      .def_static("sectorWeights", &ColorFiber::sectorWeights,
                  py::arg("state"),
                  "The four occupation-sector weights of an 8-dimensional "
                  "Fock vector (their sum is ||psi||^2 exactly).")
      .def_static("octetRead", &ColorFiber::octetRead, py::arg("m"),
                  "The octet/singlet Frobenius weights of a 3x3 bilinear.")
      .def_static("verifyConstantAlgebra", &ColorFiber::verifyConstantAlgebra,
                  "Re-derive every constant-algebra identity and return the "
                  "maximum absolute residual (run at startup in debug "
                  "builds; callable in every build).")
      .def_static("constantAlgebraCertificate",
                  &ColorFiber::constantAlgebraCertificate,
                  "The #764 AlgebraicallyExact certificate of the constant "
                  "algebra (measured verifyConstantAlgebra residual against "
                  "the startup tolerance 1e-12).");

  py::class_<ColorAnchor>(m, "ColorAnchor",
      R"doc(The calibrated weighted oriented-triangle anchoring kernel for
an abstract rank-three band (#767; whitepaper "Quarks as modular
clusters"): A_tau = |W_tau|^{1/2} R_tau Phi per declared oriented triangle,
atlas score a^2 = sum_tau w_tau |det A_tau|^2 with the convex weighting
DECLARED BEFORE the data are examined (post-hoc re-weighting raises).

Exact identity and domain: with the frame |W|-orthonormal (verified per
evaluate and reported as frame_gram_residual) and |W| triangle-decoupled
(any diagonal per-edge metric -- the production DEC/Hodge case), each
|det A_tau|^2 = det(A_tau^dag A_tau) <= 1 because R_tau^dag |W_tau| R_tau
is dominated by |W|, so the score is calibrated to [0, 1] with value one
exactly at full concentration on the weighted edge span.  A single literal
triangle is the exact oracle; an extended anchored fiber is the production
case.  For a general Hermitian (coupled) weight the <= 1 bound is CHECKED
(calibration_margin), never assumed.  Signed sectors restrict with
|W_tau|^{1/2} and report each restricted block's Krein signature
separately.

Operates only on caller-supplied inputs (frame over oriented edges, edge
weight data, oriented-triangle descriptors); mutates nothing; never enters
the emergence objective; contains no transport code.)doc")
      .def(py::init<std::vector<OrientedTriangle>>(), py::arg("triangles"),
           "Declare the atlas with the UNIFORM convex weighting 1/T.")
      .def(py::init<std::vector<OrientedTriangle>, std::vector<double>>(),
           py::arg("triangles"), py::arg("weights"),
           "Declare the atlas with an EXPLICIT convex weighting (each >= 0, "
           "summing to one within 1e-12).")
      .def("triangles", &ColorAnchor::triangles,
           "The declared oriented triangles (immutable).")
      .def("weights", &ColorAnchor::weights, "The declared convex weights.")
      .def("weightingId", &ColorAnchor::weightingId,
           "'uniform' or 'declared'.")
      .def("sealed", &ColorAnchor::sealed,
           "True once any data have been evaluated (weighting sealed).")
      .def("declareWeights", &ColorAnchor::declareWeights, py::arg("weights"),
           "Replace the declared convex weighting -- allowed ONLY before "
           "the first evaluate(); afterwards post-hoc weight selection is "
           "rejected (raises).")
      .def("evaluate",
           py::overload_cast<const Eigen::MatrixXcd&, const Eigen::VectorXd&,
                             double>(&ColorAnchor::evaluate),
           py::arg("frame"), py::arg("edge_weights"),
           py::arg("gram_tolerance") = 1e-9,
           "Evaluate against a DIAGONAL (possibly signed) per-edge weight "
           "vector -- the domain where the [0,1] calibration bound is "
           "exact.  The frame must be |W|-orthonormal within "
           "gram_tolerance (use orthonormalizeFrame).")
      .def("evaluateMatrix",
           py::overload_cast<const Eigen::MatrixXcd&, const Eigen::MatrixXcd&,
                             double>(&ColorAnchor::evaluate),
           py::arg("frame"), py::arg("weight"),
           py::arg("gram_tolerance") = 1e-9,
           "Evaluate against a general Hermitian ExE weight matrix; the "
           "calibration bound is checked (calibration_margin), not "
           "assumed.")
      .def_static("anchorMatrix", &ColorAnchor::anchorMatrix,
                  py::arg("frame"), py::arg("edge_weights"), py::arg("tri"),
                  "The raw 3x3 weighted anchor matrix A_tau = |W_tau|^{1/2} "
                  "R_tau Phi of one triangle (diagonal weights; no "
                  "normalization check).")
      .def_static("orthonormalizeFrame",
                  py::overload_cast<const Eigen::MatrixXcd&,
                                    const Eigen::VectorXd&>(
                      &ColorAnchor::orthonormalizeFrame),
                  py::arg("frame"), py::arg("edge_weights"),
                  "The |W|-orthonormalized frame Phi (Phi^dag |W| "
                  "Phi)^{-1/2} for a diagonal per-edge weight vector.")
      .def_static("orthonormalizeFrameMatrix",
                  py::overload_cast<const Eigen::MatrixXcd&,
                                    const Eigen::MatrixXcd&>(
                      &ColorAnchor::orthonormalizeFrame),
                  py::arg("frame"), py::arg("weight"),
                  "Matrix-weight overload of orthonormalizeFrame (Hermitian "
                  "W; uses the eigen-modulus |W|).");
}
