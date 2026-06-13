// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/options.h>
#include <pybind11/complex.h>
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
#include "observables/SparseGraph.h"
#include "observables/VolumeProfile.h"
#include "observables/WilsonLoop.h"
#include "observables/Spectral.h"
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
      .def("isBipartite", &SparseGraph::isBipartite,
           "True iff the graph is 2-colorable (no odd cycle).")
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
           "Number of D_S measurements taken.");
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
}
