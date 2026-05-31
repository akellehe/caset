// Pybind11 bindings for the quantum subsystem. Lives outside libtessera_quantum
// (which is pybind-free) so the static library can be reused without pulling
// in the Python dependency. This translation unit is always added to
// _tessera's sources (the quantum subsystem is unconditional).
//
// Surface area is deliberately minimal per PLAN.md §1: scalars in, scalars
// out. No MPS / MPO / ITensor types cross the Python boundary.
//
// API style: every Python-visible operation is a method on a coarse-grained
// class — there are no free functions in tessera.quantum. The four user-
// facing classes are:
//
//   • SchwingerModel(config)    — DMRG ground-state pipeline.
//   • SchwingerQuench(config)   — quench + TDVP dynamics + causal-order
//                                 comparison pipeline.
//   • Majorization              — static utility for predicate-driven
//                                 poset construction and pairwise
//                                 order-agreement statistics.
//   • Causet                    — static utility for tessera.Spacetime
//                                 → causet adapters.
//
// Plus the existing data classes (QuantumConfig, GroundStateResult,
// SchmidtSpectra, …) and the MajorizationPredicate hierarchy.

#include "quantum/CausalCompare.hpp"
#include "quantum/CausetChain.hpp"
#include "quantum/ChoiState.hpp"
#include "quantum/DMRGRunner.hpp"
#include "quantum/Holography.hpp"
#include "simulations/InteractionSimulation.h"
#include "quantum/Majorization.hpp"
#include "quantum/MutualInformation.hpp"
#include "quantum/KoashiImoto.hpp"
#include "quantum/QuantumSimplex.hpp"
#include "quantum/QuantumVertex.hpp"
#include "quantum/Schmidt.hpp"
#include "quantum/TDVPRunner.hpp"
#include "spacetime/Spacetime.h"  // full type needed for py::cast<Spacetime*>()

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_quantum(py::module_ m) {
    using namespace tessera::quantum;
    // InteractionSimulation moved to tessera:: (see issue #44). Keep its
    // Python module path as tessera.quantum.InteractionSimulation for
    // backward compatibility with existing scripts; only the C++ host
    // namespace changed.
    using ::tessera::InitialChargeMode;
    using ::tessera::InteractionConfig;
    using ::tessera::InteractionSimulation;

    m.doc() = R"doc(
Schwinger model + DMRG + TDVP + causal-order analysis.

The user-facing API is exclusively class-based:

* :class:`SchwingerModel`  — DMRG ground-state pipeline (Phases 2 / 3).
* :class:`SchwingerQuench` — q-qbar quench + TDVP + causal-order
                              comparison (Phases 4 / 5).
* :class:`Majorization`    — static utility: poset construction and
                              pairwise order-agreement statistics.
* :class:`Causet`          — static utility: tessera.Spacetime → causet
                              adapter.

Plus the data classes (QuantumConfig, GroundStateResult, SchmidtSpectra,
TDVPConfig, …) and the MajorizationPredicate hierarchy
(StandardMajorization, LogConcaveMajorization, PeakRadialMajorization)
that callers configure the workflow with.

The Hamiltonian is the staggered Kogut-Susskind Schwinger model after
Jordan-Wigner mapping and Gauss's-law elimination, expressed as a spin
chain (PLAN.md §4 / Bañuls et al., JHEP 11, 158 (2013), eq. 2.6):

.. math::

    H = H_\text{hop} + H_m + H_E

    H_\text{hop} = \frac{1}{4a} \sum_{n=1}^{N-1}
                   (X_n X_{n+1} + Y_n Y_{n+1})

    H_m   = \frac{m}{2} \sum_{n=1}^{N} (-1)^n \sigma^z_n

    H_E   = \frac{g^2 a}{2} \sum_{n=1}^{N-1} L_n^2,
            \quad L_n = L_0 + \sum_{k=1}^{n}\bigl[(1-\sigma^z_k)/2 - (1-(-1)^k)/2\bigr]

References
----------
* Bañuls, Cichy, Cirac, Jansen, *JHEP* **11**, 158 (2013),
  arXiv:1305.3765 — primary reference for the Hamiltonian and benchmarks.
* Schwinger, *Phys. Rev.* **128**, 2425 (1962) — original gauge theory.
* Coleman, *Ann. Phys.* **101**, 239 (1976) — massive Schwinger model.
)doc";

    // ─── Ground-state config + result ──────────────────────────────────
    py::class_<QuantumConfig>(m, "QuantumConfig",
            R"doc(Configuration for a Schwinger-model DMRG ground-state run.

Bundles the dimensional Hamiltonian parameters and the DMRG sweep
settings into a single struct so calling code only hands one Python
object across the C++ boundary. Default-constructed instances have
N = 0 and must be filled in before passing to :class:`SchwingerModel`.

The dt and T fields are reserved for the TDVP quench pipeline and
are ignored by SchwingerModel.

Attributes
----------
N : int
    Staggered sites, 1-based indexing. Must be ≥ 2; even N is required
    if you want the global GS to live in the Sz = 0 sector.
a : float
    Lattice spacing. Must be positive. Default 1.0.
m : float
    Bare staggered-fermion mass.
g : float
    Gauge coupling. May be 0 (free-Dirac limit, gauge field decouples).
L0 : float
    Background electric field on the link to the left of site 1.
maxBondDim : int
    Cap on the MPS bond dimension during DMRG sweeps.
nSweeps : int
    Total number of DMRG sweeps.
cutoff : float
    SVD truncation threshold per local solve.
krylovDim : int
    Lanczos / Krylov dimension per local 2-site solve.
quiet : bool
    If True (default), suppress ITensor's per-sweep diagnostic prints.
conserveQns : bool
    If True (default), enforce U(1) total-Sz conservation on the SiteSet.
dt : float
    TDVP real-time step size; used by SchwingerQuench, not by SchwingerModel.
T : float
    Total TDVP evolution time; used by SchwingerQuench, not by SchwingerModel.

Examples
--------
>>> from tessera.quantum import QuantumConfig, SchwingerModel
>>> cfg = QuantumConfig()
>>> cfg.N = 20
>>> cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.0; cfg.L0 = 0.0
>>> cfg.maxBondDim = 100; cfg.nSweeps = 12
>>> result = SchwingerModel(cfg).solve()
)doc")
        .def(py::init<>())
        .def_readwrite("N",            &QuantumConfig::N)
        .def_readwrite("a",            &QuantumConfig::a)
        .def_readwrite("m",            &QuantumConfig::m)
        .def_readwrite("g",            &QuantumConfig::g)
        .def_readwrite("L0",           &QuantumConfig::L0)
        .def_readwrite("maxBondDim",   &QuantumConfig::maxBondDim)
        .def_readwrite("nSweeps",      &QuantumConfig::nSweeps)
        .def_readwrite("cutoff",       &QuantumConfig::cutoff)
        .def_readwrite("krylovDim",    &QuantumConfig::krylovDim)
        .def_readwrite("quiet",        &QuantumConfig::quiet)
        .def_readwrite("conserveQns",  &QuantumConfig::conserveQns)
        .def_readwrite("dt",           &QuantumConfig::dt)
        .def_readwrite("T",            &QuantumConfig::T)
        .def("__repr__", [](QuantumConfig const& c) {
            return "QuantumConfig(N=" + std::to_string(c.N) +
                   ", a=" + std::to_string(c.a) +
                   ", m=" + std::to_string(c.m) +
                   ", g=" + std::to_string(c.g) +
                   ", L0=" + std::to_string(c.L0) +
                   ", maxBondDim=" + std::to_string(c.maxBondDim) +
                   ", nSweeps=" + std::to_string(c.nSweeps) + ")";
        });

    py::class_<GroundStateResult>(m, "GroundStateResult",
            R"doc(Result of :meth:`SchwingerModel.solve`.

Attributes
----------
energy : float
    Full physical energy ⟨H⟩ + constant.
operatorEnergy : float
    ⟨H⟩ alone, as returned by ITensor's dmrg().
constant : float
    The c-number shift (g²a/2) Σ_n (c_n² + n/4) pulled out of L_n²
    when AutoMPO-encoding the electric term.
bondDim : int
    Largest bond dimension of the optimized MPS.
truncationErr : float
    Conservative upper bound on the SVD truncation error.
)doc")
        .def_readonly("energy",          &GroundStateResult::energy)
        .def_readonly("operatorEnergy",  &GroundStateResult::operatorEnergy)
        .def_readonly("constant",        &GroundStateResult::constant)
        .def_readonly("bondDim",         &GroundStateResult::bondDim)
        .def_readonly("truncationErr",   &GroundStateResult::truncationErr)
        .def("__repr__", [](GroundStateResult const& r) {
            return "GroundStateResult(energy=" + std::to_string(r.energy) +
                   ", bondDim=" + std::to_string(r.bondDim) +
                   ", truncationErr=" + std::to_string(r.truncationErr) + ")";
        });

    // ─── Schmidt / majorization data classes ──────────────────────────
    py::class_<Interval>(m, "Interval",
            R"doc(1-based contiguous interval [i, j] on the spin chain.)doc")
        .def(py::init<>())
        .def_readwrite("i", &Interval::i)
        .def_readwrite("j", &Interval::j)
        .def("__repr__", [](Interval const& iv) {
            return "Interval(i=" + std::to_string(iv.i) +
                   ", j=" + std::to_string(iv.j) + ")";
        });

    py::class_<SchmidtSpectra>(m, "SchmidtSpectra",
            R"doc(All contiguous-cut Schmidt spectra of an MPS.)doc")
        .def_readonly("N",         &SchmidtSpectra::N)
        .def_readonly("intervals", &SchmidtSpectra::intervals)
        .def_readonly("spectra",   &SchmidtSpectra::spectra);

    py::class_<Poset>(m, "Poset",
            R"doc(Hasse / cover representation of a finite partial order.

Nodes are integers ``0 .. getNodeCount - 1``. ``covers`` lists the cover
edges: each entry ``(a, b)`` means ``a`` strictly precedes ``b`` with no
intermediate node. The full strict order is the transitive closure of
the covers; see :func:`compareOrders` for pairwise statistics derived
from that closure.

Construct empty (``Poset()``) and resize via the ``getNodeCount``
setter, or pass an integer to pre-populate node count
(``Poset(4)``). Mutate via :meth:`addCover` (single edge) or the
``covers`` setter (whole list). The class makes no internal
consistency checks; callers are responsible for transitivity and
acyclicity.

See ``docs/source/causal_sets.md`` for the conceptual background.
)doc")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("nodeCount"),
            "Construct with the node count pre-set to ``nodeCount``.")
        .def("addCover", &Poset::addCover, py::arg("a"), py::arg("b"),
            R"doc(Add the cover edge ``a -> b`` (a strictly precedes b, no intermediate).

Both endpoints must already exist (call the int constructor or the
``getNodeCount`` setter first). No deduplication is performed —
adding the same cover twice creates two parallel edges. The standard
use is to feed covers from a transitive-reduction algorithm where
duplicates can't arise.
)doc")
        .def_property("getNodeCount",
            [](Poset const& p) { return p.getNodeCount(); },
            [](Poset& p, int n) { p.setNodeCount(n); },
            "Number of nodes. Setting grows the node set; nodes are not "
            "removed if you set a smaller value, and cover edges are "
            "preserved across resizes.")
        .def("getCoverCount", &Poset::getCoverCount,
            "Number of cover edges currently registered.")
        .def_property("covers",
            [](Poset const& p) { return p.covers(); },
            [](Poset& p, std::vector<std::pair<int, int>> const& covers) {
                p.setCovers(covers);
            },
            "Cover edges as a list of ``(a, b)`` pairs. Setting replaces "
            "the entire cover list in one pass.")
        .def("toDot", &Poset::toDot,
            "Graphviz DOT representation of the Hasse diagram. "
            "Nodes labelled by their integer id; one directed edge per "
            "cover. Suitable for ``dot -Tsvg`` rendering.")
        .def_static("fromSpacetime",
            [](py::object spacetime_obj) {
                auto const* st = spacetime_obj.cast<tessera::spacetime::Spacetime const*>();
                return tessera::Poset::fromSpacetime(*st);
            }, py::arg("spacetime"),
            R"doc(Build the causet partial order on a Spacetime's vertices.

Reads the directed-edge / timelike-edge subgraph as the strict
``precedes`` relation, then takes the transitive reduction to recover
cover edges. The result has one node per Spacetime vertex (in
ascending ID order); cover edges are between strictly comparable
vertices with no intermediate.
)doc")
        .def("__repr__", [](Poset const& p) {
            return "Poset(getNodeCount=" + std::to_string(p.getNodeCount()) +
                   ", covers=" + std::to_string(p.getCoverCount()) + " edges)";
        });

    py::class_<GroundStateMajorizationResult>(m, "GroundStateMajorizationResult",
            R"doc(Result of :meth:`SchwingerModel.solveWithMajorization`.

Attributes
----------
groundState : GroundStateResult
    Same diagnostics returned by :meth:`SchwingerModel.solve`.
spectra : SchmidtSpectra
    All contiguous-cut Schmidt spectra of the ground-state MPS.
poset : Poset
    Hasse cover edges of the strict-majorization order on
    ``spectra.spectra``.
)doc")
        .def_readonly("groundState", &GroundStateMajorizationResult::groundState)
        .def_readonly("spectra",     &GroundStateMajorizationResult::spectra)
        .def_readonly("poset",       &GroundStateMajorizationResult::poset);

    // ─── MajorizationPredicate hierarchy ────────────────────────────────
    py::class_<MajorizationPredicate>(m, "MajorizationPredicate",
            R"doc(Abstract base class for variants of the majorization predicate.

Concrete subclasses:

* :class:`StandardMajorization` -- classical Nielsen 1999 PRL 83, 436 eq. (1).
* :class:`LogConcaveMajorization` -- gated on log-concave spectra
  (Brändén 2015 §1).
* :class:`PeakRadialMajorization` -- peak-relative entrywise dominance.

Methods that take a predicate -- :meth:`Majorization.posetOf` and
:meth:`SchwingerQuench.compareCausalOrders` -- accept any concrete
subclass instance.
)doc")
        .def("majorizes",
             &MajorizationPredicate::majorizes,
             py::arg("mu"), py::arg("lambda_"),
             "True iff μ majorizes λ under this variant.")
        .def("strictlyMajorizes",
             &MajorizationPredicate::strictlyMajorizes,
             py::arg("mu"), py::arg("lambda_"),
             "True iff μ strictly majorizes λ under this variant.")
        .def_property_readonly("name",
             &MajorizationPredicate::name,
             "Short identifier of the variant.");

    py::class_<StandardMajorization, MajorizationPredicate>(
            m, "StandardMajorization",
            R"doc(Classical Nielsen-1999 majorization.

μ ≻ λ iff for every k, sum of the top-k of μ ≥ sum of the top-k of λ,
with equality at the total-mass step (Nielsen 1999 PRL 83, 436, eq. 1).
)doc")
        .def(py::init<double>(), py::arg("tol") = 1e-12)
        .def_property_readonly("tol", &StandardMajorization::tol);

    py::class_<LogConcaveMajorization, StandardMajorization>(
            m, "LogConcaveMajorization",
            R"doc(Standard majorization, restricted to log-concave spectra
(Brändén 2015 arXiv:1410.6601 §1).
)doc")
        .def(py::init<double>(), py::arg("tol") = 1e-12)
        .def_static("isLogConcave",
                    &LogConcaveMajorization::isLogConcave,
                    py::arg("v"), py::arg("tol") = 1e-12,
                    "True iff the spectrum is log-concave on its support.");

    py::class_<PeakRadialMajorization, MajorizationPredicate>(
            m, "PeakRadialMajorization",
            R"doc(Peak-radial dominance: relative-to-peak entrywise majorization.
Strictly stronger than classical majorization.
)doc")
        .def(py::init<double>(), py::arg("tol") = 1e-12)
        .def_property_readonly("tol", &PeakRadialMajorization::tol);

    // ─── TDVP quench config + snapshot ─────────────────────────────────
    py::class_<TDVPConfig>(m, "TDVPConfig",
            R"doc(Configuration for the q-qbar quench + TDVP run.

Bundles the Hamiltonian parameters, the DMRG ground-state setup, the
quench location / separation, and the real-time-evolution schedule.
``i0`` and ``d`` describe the q-qbar pair: σ⁻ acts at site ``i0``,
σ⁺ at site ``i0 + d``. With ``quenchEnforceParity = True`` (default)
``i0`` must be odd and ``d`` must be odd.

The pipeline runs through :meth:`SchwingerQuench.evolve` (or
:meth:`SchwingerQuench.compareCausalOrders` for the causal-order
extension).
)doc")
        .def(py::init<>())
        .def_readwrite("N",                       &TDVPConfig::N)
        .def_readwrite("a",                       &TDVPConfig::a)
        .def_readwrite("m",                       &TDVPConfig::m)
        .def_readwrite("g",                       &TDVPConfig::g)
        .def_readwrite("L0",                      &TDVPConfig::L0)
        .def_readwrite("dmrgMaxBondDim",          &TDVPConfig::dmrgMaxBondDim)
        .def_readwrite("dmrgNSweeps",             &TDVPConfig::dmrgNSweeps)
        .def_readwrite("dmrgKrylovDim",           &TDVPConfig::dmrgKrylovDim)
        .def_readwrite("dmrgCutoff",              &TDVPConfig::dmrgCutoff)
        .def_readwrite("i0",                      &TDVPConfig::i0)
        .def_readwrite("d",                       &TDVPConfig::d)
        .def_readwrite("quenchEnforceParity",     &TDVPConfig::quenchEnforceParity)
        .def_readwrite("dt",                      &TDVPConfig::dt)
        .def_readwrite("T",                       &TDVPConfig::T)
        .def_readwrite("maxBondDim",              &TDVPConfig::maxBondDim)
        .def_readwrite("krylovDim",               &TDVPConfig::krylovDim)
        .def_readwrite("cutoff",                  &TDVPConfig::cutoff)
        .def_readwrite("snapshotEvery",           &TDVPConfig::snapshotEvery)
        .def_readwrite("quiet",                   &TDVPConfig::quiet)
        .def_readwrite("conserveQns",             &TDVPConfig::conserveQns)
        .def_readwrite("recordSpectra",           &TDVPConfig::recordSpectra)
        .def_readwrite("recordPoset",             &TDVPConfig::recordPoset)
        .def_readwrite("recordMutualInformation", &TDVPConfig::recordMutualInformation)
        .def_readwrite("recordBondMutualInformation", &TDVPConfig::recordBondMutualInformation)
        .def_readwrite("hoppingPairs",            &TDVPConfig::hoppingPairs,
            "Optional custom hopping graph as a list of (i, j) "
            "0-based flat-lattice pairs. Empty = default 1D NN chain. "
            "Sourced from tessera.quantum.Causet.chainFrom(spacetime).");

    py::class_<TDVPSnapshot>(m, "TDVPSnapshot",
            R"doc(Per-step diagnostics recorded during a TDVP run.

Always-populated fields: time, energy, bondDim, zProfile, lProfile.
Optional fields (populated only if the corresponding TDVPConfig flag
is set): spectra, poset, mutualInformation.
)doc")
        .def_readonly("time",      &TDVPSnapshot::time)
        .def_readonly("energy",    &TDVPSnapshot::energy)
        .def_readonly("bondDim",   &TDVPSnapshot::bondDim)
        .def_readonly("zProfile",  &TDVPSnapshot::zProfile)
        .def_readonly("lProfile",  &TDVPSnapshot::lProfile)
        .def_readonly("spectra",   &TDVPSnapshot::spectra)
        .def_readonly("poset",     &TDVPSnapshot::poset)
        .def_readonly("bondMutualInformation",
                       &TDVPSnapshot::bondMutualInformation,
                       R"doc(Symmetric (N-1)×(N-1) bond-cut tripartite-information matrix in nats,
flattened row-major. Zero diagonal.
Populated iff TDVPConfig.recordBondMutualInformation = True.)doc")
        .def_readonly("mutualInformation", &TDVPSnapshot::mutualInformation,
                       "Flat row-major N×N matrix of site-site MI in nats. "
                       "Populated iff TDVPConfig.recordMutualInformation = True.")
        .def("__repr__", [](TDVPSnapshot const& s) {
            return "TDVPSnapshot(time=" + std::to_string(s.time) +
                   ", energy=" + std::to_string(s.energy) +
                   ", bondDim=" + std::to_string(s.bondDim) + ")";
        });

    py::class_<QuenchResult>(m, "QuenchResult",
            R"doc(Result of :meth:`SchwingerQuench.evolve`.

Attributes
----------
groundState : GroundStateResult
    DMRG ground-state diagnostics for the pre-quench state.
snapshots : list[TDVPSnapshot]
    Per-step diagnostics; ``snapshots[0]`` is the post-quench state.
)doc")
        .def_readonly("groundState", &QuenchResult::groundState)
        .def_readonly("snapshots",   &QuenchResult::snapshots);

    // ─── Causal-order comparison data classes ─────────────────────────
    py::class_<LabelSpacetime>(m, "LabelSpacetime",
            R"doc(One label in a (cut, time) spacetime.)doc")
        .def_readonly("cutIdx",    &LabelSpacetime::cutIdx)
        .def_readonly("tIdx",      &LabelSpacetime::tIdx)
        .def_readonly("intervalI", &LabelSpacetime::intervalI)
        .def_readonly("intervalJ", &LabelSpacetime::intervalJ)
        .def_readonly("time",      &LabelSpacetime::time);

    py::class_<CausalOrders>(m, "CausalOrders",
            R"doc(Three Hasse-cover posets on a shared (cut, time) label set.

Build via :meth:`CausalOrders.fromSnapshots` (which delegates to the
underlying SchwingerQuench pipeline; most users go through
:meth:`SchwingerQuench.compareCausalOrders` instead).
)doc")
        .def_readonly("labels", &CausalOrders::labels)
        .def_readonly("maj",    &CausalOrders::maj)
        .def_readonly("lr",     &CausalOrders::lr)
        .def_readonly("cs",     &CausalOrders::cs)
        .def_static("fromSnapshots",
            [](std::vector<TDVPSnapshot> const& snapshots,
               double vLr,
               MajorizationPredicate const* predicate) {
                return CausalOrders::fromSnapshots(snapshots, vLr, predicate);
            },
            py::arg("snapshots"), py::arg("vLr"), py::arg("predicate") = nullptr,
            R"doc(Build the three orders from a list of TDVP snapshots.)doc");

    py::class_<OrderAgreement>(m, "OrderAgreement",
            R"doc(Pairwise agreement statistics between two posets.

Counted over unordered pairs (i, j) with i < j:
* concordant — both orders relate the pair, in the same direction.
* discordant — both orders relate the pair, in opposite directions.
* only_a / only_b — exactly one order relates the pair.

Build via :meth:`Majorization.agreement(a, b, n_labels)`.
)doc")
        .def_readonly("kendallTau",         &OrderAgreement::kendallTau)
        .def_readonly("discordantFraction", &OrderAgreement::discordantFraction)
        .def_readonly("hasseEditDistance",  &OrderAgreement::hasseEditDistance)
        .def_readonly("nConcordant",        &OrderAgreement::nConcordant)
        .def_readonly("nDiscordant",        &OrderAgreement::nDiscordant)
        .def_readonly("nComparableBoth",    &OrderAgreement::nComparableBoth)
        .def_readonly("nOnlyA",             &OrderAgreement::nOnlyA)
        .def_readonly("nOnlyB",             &OrderAgreement::nOnlyB);

    m.def("compareOrders", &tessera::compareOrders,
        py::arg("a"), py::arg("b"), py::arg("nLabels"),
        R"doc(Pairwise agreement statistics between two posets on a shared label set.

Counts unordered pairs (i, j) with i < j in five disjoint buckets via
Floyd–Warshall transitive closures of `a` and `b`:

* concordant   — both orders relate the pair the same way
* discordant   — both orders relate the pair, opposite ways
* only-a       — `a` relates the pair, `b` does not
* only-b       — `b` relates the pair, `a` does not
* neither      — neither order relates the pair

Returns an :class:`OrderAgreement` with ``kendallTau``,
``discordantFraction``, ``hasseEditDistance``, and the five counts.

Complexity: O(nLabels^3) for the transitive closure, O(nLabels^2) for
the pair counts. Practical up to a few thousand labels.

See ``docs/source/causal_sets.md`` for the methodology context.
)doc");

    py::class_<CausalComparisonReport>(m, "CausalComparisonReport",
            R"doc(Pairwise agreement statistics across the three causal orders (≼_maj, ≼_LR, ≼_cs).

Build via :meth:`SchwingerQuench.compareCausalOrders`.
)doc")
        .def_readonly("majVsLr",    &CausalComparisonReport::majVsLr)
        .def_readonly("majVsCs",    &CausalComparisonReport::majVsCs)
        .def_readonly("lrVsCs",     &CausalComparisonReport::lrVsCs)
        .def_readonly("nLabels",    &CausalComparisonReport::nLabels)
        .def_readonly("nSnapshots", &CausalComparisonReport::nSnapshots)
        .def_readonly("vLr",        &CausalComparisonReport::vLr)
        .def_readonly("majKind",    &CausalComparisonReport::majKind);

    // ─── Causet adapter data classes ───────────────────────────────────
    py::class_<CausetChain>(m, "CausetChain",
            R"doc(Spacetime-derived chain layout for the Schwinger MPO.

Build via :meth:`Causet.chainFrom`.

Attributes
----------
nSites : int
    Total number of lattice sites.
times : list[int]
    Sorted ascending list of integer time slices.
antichains : list[list[int]]
    ``antichains[s]`` is the ascending-ID list of vertex IDs at
    ``times[s]``.
vertexIds : list[int]
    Flat lattice site → Spacetime vertex ID.
hoppingPairs : list[tuple[int, int]]
    Pairs ``(i, j)`` of flat lattice sites coupled by adjacent-slice
    timelike causet edges.
partialOrder : Poset
    Hasse cover Poset on the nSites label set.
)doc")
        .def_readonly("nSites",       &CausetChain::nSites)
        .def_readonly("times",        &CausetChain::times)
        .def_readonly("antichains",   &CausetChain::antichains)
        .def_readonly("vertexIds",    &CausetChain::vertexIds)
        .def_readonly("hoppingPairs", &CausetChain::hoppingPairs)
        .def_readonly("partialOrder", &CausetChain::partialOrder)
        .def("__repr__", [](CausetChain const& c) {
            return "CausetChain(nSites=" + std::to_string(c.nSites) +
                   ", times=" + std::to_string(c.times.size()) +
                   ", hops=" + std::to_string(c.hoppingPairs.size()) + ")";
        });

    // ─── Coarse-grained workflow classes ────────────────────────────────

    py::class_<SchwingerModel>(m, "SchwingerModel",
            R"doc(Schwinger-model DMRG ground-state pipeline.

Coarse-grained interface for the ground-state workflow:
bundle a :class:`QuantumConfig` and call :meth:`solve` for the bare
DMRG diagnostics or :meth:`solveWithMajorization` for the full Schmidt
+ majorization-poset bundle.

The model is stateless beyond its config — every method runs the
underlying ITensor pipeline from scratch.

Examples
--------
>>> from tessera.quantum import QuantumConfig, SchwingerModel
>>> cfg = QuantumConfig()
>>> cfg.N = 4; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.0; cfg.L0 = 0.0
>>> cfg.maxBondDim = 32; cfg.nSweeps = 8
>>> r = SchwingerModel(cfg).solve()
>>> abs(r.operatorEnergy - (-1.738676174)) < 1e-8
True
)doc")
        .def(py::init<QuantumConfig>(), py::arg("config"))
        .def_property_readonly("config",
            [](SchwingerModel const& m) { return m.config(); },
            "The bound QuantumConfig (read-only).")
        .def("solve", &SchwingerModel::solve,
            R"doc(Run DMRG to the Schwinger ground state.

Returns
-------
GroundStateResult
    Energy, bond-dim, and truncation diagnostics.

Raises
------
RuntimeError or ValueError
    If config.N < 2 or config.a <= 0.
)doc")
        .def("solveWithMajorization",
            &SchwingerModel::solveWithMajorization,
            py::arg("tol") = 1e-12,
            R"doc(Run DMRG, then extract Schmidt spectra and majorization poset.

Single-shot pipeline:
1. DMRG ground state of the Schwinger MPO.
2. Schmidt spectrum of every contiguous bipartition (excluding the
   trivial full-chain cut).
3. Majorization poset on those spectra (Hasse cover edges only).

Parameters
----------
tol : float, optional
    Slack for the majorization comparisons. Default 1e-12.

Returns
-------
GroundStateMajorizationResult
)doc");

    py::class_<SchwingerQuench>(m, "SchwingerQuench",
            R"doc(Schwinger-model q-qbar quench + TDVP pipeline.

Coarse-grained interface for the dynamics workflow: bundle a
:class:`TDVPConfig` and call :meth:`evolve` for the snapshot
trajectory or :meth:`compareCausalOrders` for the causal-order
comparison. The model is stateless beyond its config.

Examples
--------
>>> from tessera.quantum import TDVPConfig, SchwingerQuench
>>> cfg = TDVPConfig()
>>> cfg.N = 14; cfg.m = 20.0; cfg.g = 1.0
>>> cfg.i0 = 5; cfg.d = 5
>>> cfg.dt = 0.05; cfg.T = 5.0; cfg.snapshotEvery = 5
>>> r = SchwingerQuench(cfg).evolve()  # doctest: +SKIP
>>> r.snapshots[0].lProfile[:3]        # doctest: +SKIP
[-1.0, -0.0, -0.0]
)doc")
        .def(py::init<TDVPConfig>(), py::arg("config"))
        .def_property_readonly("config",
            [](SchwingerQuench const& q) { return q.config(); },
            "The bound TDVPConfig (read-only).")
        .def("evolve", &SchwingerQuench::evolve,
            R"doc(Run the full DMRG → quench → TDVP pipeline.

Returns
-------
QuenchResult
    Ground-state diagnostics + snapshot list.
)doc")
        .def("compareCausalOrders",
            [](SchwingerQuench const& q,
               double vLr,
               MajorizationPredicate const* predicate) {
                return q.compareCausalOrders(vLr, predicate);
            },
            py::arg("vLr") = 1.0,
            py::arg("predicate") = nullptr,
            R"doc(End-to-end causal-order comparison.

Runs :meth:`evolve` (forcing recordSpectra=True), builds three
partial orders on the (cut, time) label set:

* ≼_maj — strict-majorization on Schmidt spectra (across cuts AND times).
* ≼_LR  — Lieb-Robinson cone: dist(A, B) ≤ vLr · (t_B - t_A).
* ≼_cs  — causet (time-only on regular chain).

Pairwise agreement is reported as Kendall-τ, the discordant-pair
fraction, and the Hasse-graph edit distance.

Parameters
----------
vLr : float, optional
    Lieb-Robinson velocity in lattice units. Default 1.0.
predicate : MajorizationPredicate, optional
    Majorization variant for ≼_maj. None = StandardMajorization (default).

Returns
-------
CausalComparisonReport
)doc");

    py::class_<Majorization>(m, "Majorization",
            R"doc(Static utility for majorization-poset construction and pairwise
order-agreement statistics. Not instantiable; call methods on the class.

>>> from tessera.quantum import Majorization, StandardMajorization
>>> p = Majorization.posetOf([[1.0], [0.5, 0.5], [1/3]*3])
>>> sorted(p.covers)
[(0, 1), (1, 2)]
)doc")
        // Two static overloads of `posetOf` (one with predicate, one with
        // tol). Spelled out explicitly because pybind11 needs help selecting
        // between them.
        .def_static("posetOf",
            [](std::vector<std::vector<double>> const& spectra,
               MajorizationPredicate const& predicate) {
                return Majorization::posetOf(spectra, predicate);
            },
            py::arg("spectra"), py::arg("predicate"),
            R"doc(Build the majorization poset under an explicit predicate.)doc")
        .def_static("posetOf",
            [](std::vector<std::vector<double>> const& spectra, double tol) {
                return Majorization::posetOf(spectra, tol);
            },
            py::arg("spectra"), py::arg("tol") = 1e-12,
            R"doc(Build the classical {N1999} majorization poset at the given tolerance.)doc")
        .def_static("agreement",
            &Majorization::agreement,
            py::arg("a"), py::arg("b"), py::arg("nLabels"),
            R"doc(Pairwise agreement statistics between two Posets on the same
label set of size nLabels. Returns an :class:`OrderAgreement`.
)doc");

    py::class_<Causet>(m, "Causet",
            R"doc(Static utility for tessera.Spacetime → causet adapters
. Not instantiable; call methods on the class.
)doc")
        .def_static("chainFrom",
            [](py::object spacetime_obj) {
                auto const* st = spacetime_obj.cast<tessera::spacetime::Spacetime const*>();
                return Causet::chainFrom(*st);
            }, py::arg("spacetime"),
            R"doc(Extract a chain-of-antichains adapter from a Spacetime.

Walks the Spacetime's vertex list, groups vertices by integer time
slice, and packages the antichain layering, the flat-lattice ↔
Spacetime ID mapping, the adjacent-slice timelike-edge hopping pairs,
and the inherited Hasse cover :class:`Poset`.

Parameters
----------
spacetime : tessera.Spacetime
    Source spacetime.

Returns
-------
CausetChain
)doc");

    // ─── MutualInformation utility ─────────────────────────────────────
    py::class_<MutualInformation>(m, "MutualInformation",
            R"doc(Static utility for site-site mutual information on a Schwinger MPS.

Not instantiable. The full computation pipeline goes through
SchwingerQuench(cfg).evolve() with TDVPConfig.recordMutualInformation
= True; this class is exposed so callers can run the pure-math
operations (vonNeumannEntropy, edgeLength) on numpy data without
needing an MPS in hand.
)doc")
        .def_static("vonNeumannEntropy",
            [](py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast> rho,
                double tol) -> double {
                auto buf = rho.unchecked<2>();
                const int r = static_cast<int>(buf.shape(0));
                const int c = static_cast<int>(buf.shape(1));
                if (r != c) {
                    throw std::invalid_argument(
                        "MutualInformation.vonNeumannEntropy: rho must be square");
                }
                Eigen::MatrixXcd m(r, c);
                for (int i = 0; i < r; ++i) {
                    for (int j = 0; j < c; ++j) {
                        m(i, j) = buf(i, j);
                    }
                }
                return MutualInformation::vonNeumannEntropy(m, tol);
            },
            py::arg("rho"), py::arg("tol") = 1e-12,
            R"doc(Von Neumann entropy of a Hermitian density matrix, in nats.

The implementation accepts any square dimension; in the holography
pipeline we only need 2×2 (single-site marginals) and 4×4 (two-site
joint reduced density matrices).
)doc")
        .def_static("edgeLength",
            &MutualInformation::edgeLength,
            py::arg("I"), py::arg("epsilon") = 1e-10,
            R"doc(ℓ = -log(I) with infinity floor at -log(epsilon).)doc");

    // ─── InteractionSimulation: interaction-history Monte Carlo ────────
    // See docs/source/interaction-history-monte-carlo.md.
    py::class_<InteractionConfig>(m, "InteractionConfig",
        R"doc(Configuration for an interaction-history Monte Carlo run.

nSystems randomized correlated mixed-state systems on a Poisson-Delaunay
initial layer (delaunayEdges is the connectivity, supplied by the
caller); the Schwinger two-site unitary exp(-i H_XY dt) drives each
interaction; beta is the inverse temperature in e^{-beta S}.
)doc")
        .def(py::init<>())
        .def_readwrite("nSystems",           &InteractionConfig::nSystems)
        .def_readwrite("a",                  &InteractionConfig::a)
        .def_readwrite("g",                  &InteractionConfig::g)
        .def_readwrite("m",                  &InteractionConfig::m)
        .def_readwrite("dt",                 &InteractionConfig::dt)
        .def_readwrite("beta",               &InteractionConfig::beta)
        .def_readwrite("epsilonI",           &InteractionConfig::epsilonI)
        .def_readwrite("targetInteractions",
                       &InteractionConfig::targetInteractions)
        .def_readwrite("delaunayEdges",      &InteractionConfig::delaunayEdges)
        .def_readwrite("useCharges",         &InteractionConfig::useCharges)
        .def_readwrite("featureCharges",
                       &InteractionConfig::featureCharges)
        .def_readwrite("featureDeactivateOnAnnihilate",
                       &InteractionConfig::featureDeactivateOnAnnihilate)
        .def_readwrite("featurePhotonOnAnnihilate",
                       &InteractionConfig::featurePhotonOnAnnihilate)
        .def_readwrite("featureQuditBasis",
                       &InteractionConfig::featureQuditBasis)
        .def_readwrite("featureChoiSigmaAB",
                       &InteractionConfig::featureChoiSigmaAB)
        .def_readwrite("j_chargeCharge",
                       &InteractionConfig::j_chargeCharge)
        .def_readwrite("j_spinSpin",
                       &InteractionConfig::j_spinSpin)
        .def_readwrite("massShift",
                       &InteractionConfig::massShift)
        .def_readwrite("gammaCpViolation",
                       &InteractionConfig::gammaCpViolation)
        .def_readwrite("dtPair",
                       &InteractionConfig::dtPair)
        .def_readwrite("cpBias",             &InteractionConfig::cpBias)
        .def_readwrite("initialChargeMode",
                       &InteractionConfig::initialChargeMode)
        .def_readwrite("seed",               &InteractionConfig::seed)
        .def_readwrite("quiet",              &InteractionConfig::quiet);

    py::enum_<tessera::simulations::InitialChargeMode>(m, "InitialChargeMode")
        .value("ALTERNATING",
               tessera::simulations::InitialChargeMode::ALTERNATING)
        .value("RANDOM",
               tessera::simulations::InitialChargeMode::RANDOM);

    py::class_<InteractionSimulation>(m, "InteractionSimulation",
        R"doc(Metropolis Monte Carlo over interaction histories, weighted by
the geometric Regge action on the dual lattice.

Mirrors tessera.CDT: the move primitives interact() / unInteract(), the
driving loop sweep() / thermalize() / tune(), and the diagnostics
computeAction() / getSpectralDimension() / getAcceptanceRates(). The
object of the search is the beta at which the emergent spectral
dimension reaches 4.
)doc")
        .def(py::init<InteractionConfig>(), py::arg("config"))
        .def("interact",   &InteractionSimulation::interact,
             R"doc(Propose + Metropolis-accept one interaction. Returns acceptance.)doc")
        .def("unInteract", &InteractionSimulation::unInteract,
             R"doc(Propose + Metropolis-accept one un-interaction. Returns acceptance.)doc")
        .def("sweep",      &InteractionSimulation::sweep,
             R"doc(One Monte Carlo sweep; returns the number of accepted moves.)doc")
        .def("thermalize", &InteractionSimulation::thermalize,
             R"doc(Tune to the target volume, then sweep to equilibrium.)doc")
        .def("tune",       &InteractionSimulation::tune,
             py::arg("progress") = nullptr,
             R"doc(Grow the complex toward targetInteractions.)doc")
        .def("computeAction", &InteractionSimulation::computeAction,
             R"doc(The geometric Regge action S = sum_h A_h eps_h.)doc")
        .def("getSpectralDimension",
             &InteractionSimulation::getSpectralDimension,
             py::arg("sigmas"), py::arg("krylovDim") = 30,
             R"doc(Heat-kernel spectral dimension D_S(sigma) of the MI-weighted complex.)doc")
        .def("getDeficitAngleDistribution",
             &InteractionSimulation::getDeficitAngleDistribution,
             R"doc(Deficit angles over the interior hinges.)doc")
        .def("getVolumeProfile", &InteractionSimulation::getVolumeProfile,
             R"doc(Interaction-count profile by time slice.)doc")
        .def("getAcceptanceRates",
             &InteractionSimulation::getAcceptanceRates,
             R"doc(Accepted / attempted ratio per move type.)doc")
        .def("annihilate", &InteractionSimulation::annihilate,
             R"doc(v0.1: spontaneous partial-annihilation of a (+, -) frontier pair.)doc")
        .def("pairCreate", &InteractionSimulation::pairCreate,
             R"doc(v0.1: spontaneous (+, -) pair-creation with a Bell joint.)doc")
        .def("getGlobalCharge", &InteractionSimulation::getGlobalCharge,
             R"doc(v0.1: total signed charge across the complex.)doc")
        .def("getChargeProfile", &InteractionSimulation::getChargeProfile,
             R"doc(v0.1: per-time-slice (n_+, n_0, n_-, sum_q).)doc")
        .def("getChargeCorrelation",
             &InteractionSimulation::getChargeCorrelation,
             py::arg("maxDist"),
             R"doc(v0.1: <q_v . q_w> as a function of graph distance.)doc")
        .def("quditChargeOf", &InteractionSimulation::quditChargeOf,
             py::arg("vertex"),
             R"doc(v0.2: a single vertex's continuous charge via Tr[ρ · Q̂].

Q̂ = diag(+1, +1, -1, -1) on the {|+0⟩, |+1⟩, |−0⟩, |−1⟩} basis.
For an integer-charge eigenstate this returns ±1; for the maximally-mixed
I/4 proxy it returns 0; for an arbitrary mixed state, the value sits in
[−1, +1]. Requires ``featureQuditBasis = True``. Returns 0.0 for vertices
the simulation has no qudit state for.)doc")
        .def("quditStateOf",
             [](const InteractionSimulation &self, tessera::mesh::VertexPtr v)
                 -> py::object {
               const auto &m = self.quditStateOfMap();
               auto it = m.find(v);
               if (it == m.end()) return py::none();
               return py::cast(it->second);
             },
             py::arg("vertex"),
             R"doc(v0.2: read a single vertex's 4×4 qudit density matrix, or
``None`` if no qudit state is stored. Useful for tests that inspect
per-vertex purity, charge content, or basis populations directly rather
than going through the projected ``Tr[ρ · Q̂]`` accessor. Requires
``featureQuditBasis = True``.)doc")
        .def("quditJointStateFor",
             &InteractionSimulation::quditJointStateFor,
             py::arg("x"), py::arg("y"),
             R"doc(v0.2: 16×16 joint qudit state ρ_XY for a pair.

Returns the stored correlated joint when (x, y) share an interaction
history or are initial-layer Delaunay neighbours; otherwise the
uncorrelated product ρ_x ⊗ ρ_y.)doc")
        .def("getSpacetime", &InteractionSimulation::getSpacetime,
             R"doc(The interaction-history simplicial complex (the primal).)doc")
        .def_property_readonly("interactionCount",
             &InteractionSimulation::interactionCount)
        .def_property_readonly("frontierSize",
             &InteractionSimulation::frontierSize)
        .def_property("beta", &InteractionSimulation::getBeta,
             &InteractionSimulation::setBeta)
        .def("setSeed", &InteractionSimulation::setSeed, py::arg("seed"));

    // ─── Holography submodule: emergent spectral dimension ─────────────
    auto holo = m.def_submodule("holography",
        R"doc(Emergent spectral dimension from the Schwinger TDVP state.

See ``docs/source/holography-causal-ordering-emergent-dimension.md`` for
the scientific charter. The pipeline runs through one workflow class:

>>> from tessera.quantum import TDVPConfig
>>> from tessera.quantum.holography import HolographyConfig, EmergentSpectralDimension
>>> cfg = HolographyConfig()
>>> cfg.tdvp.N = 10; cfg.tdvp.m = 0.5; cfg.tdvp.g = 1.0
>>> # ... fill in TDVP fields ...
>>> result = EmergentSpectralDimension(cfg).compute()
)doc");

    py::class_<HolographyConfig>(holo, "HolographyConfig",
            R"doc(Configuration for an emergent-spectral-dimension run.

Wraps a TDVPConfig and adds the σ-grid and mutual-information cutoff.
Validated at construction inside EmergentSpectralDimension; invalid
configs raise ValueError before any TDVP work is done.
)doc")
        .def(py::init<>())
        .def_readwrite("tdvp",              &HolographyConfig::tdvp)
        .def_readwrite("sigmaMin",          &HolographyConfig::sigmaMin)
        .def_readwrite("sigmaMax",          &HolographyConfig::sigmaMax)
        .def_readwrite("sigmaCount",        &HolographyConfig::sigmaCount)
        .def_readwrite("epsilonI",          &HolographyConfig::epsilonI)
        .def_readwrite("includeTemporal",   &HolographyConfig::includeTemporal)
        .def_readwrite("maxTemporalStride", &HolographyConfig::maxTemporalStride)
        .def_readwrite("krylovDim",         &HolographyConfig::krylovDim)
        .def_readwrite("seed",              &HolographyConfig::seed)
        .def_readwrite("vertexIds",         &HolographyConfig::vertexIds,
            "Optional spacetime-vertex labels for the site axis. "
            "Sourced from tessera.quantum.Causet.chainFrom(spacetime). "
            "Empty = use flat-site indices 0..N-1 as labels.")
        .def("validate",                    &HolographyConfig::validate);

    py::class_<MutualInformationProfile>(holo, "MutualInformationProfile",
            R"doc(Symmetric site×snapshot mutual-information matrix.

Built from a list of TDVPSnapshots that have ``mutualInformation``
recorded. Provides indexed access via at(site_v, snap_v, site_w, snap_w)
and a COO weighted-adjacency export.
)doc")
        .def(py::init<std::vector<TDVPSnapshot> const&,
                       HolographyConfig const&>(),
             py::arg("snapshots"), py::arg("config"))
        .def_property_readonly("nSites",     &MutualInformationProfile::nSites)
        .def_property_readonly("nSnapshots", &MutualInformationProfile::nSnapshots)
        .def_property_readonly("nLabels",    &MutualInformationProfile::nLabels)
        .def("at",     &MutualInformationProfile::at,
             py::arg("siteV"), py::arg("snapV"),
             py::arg("siteW"), py::arg("snapW"))
        .def("atFlat", &MutualInformationProfile::atFlat,
             py::arg("v"), py::arg("w"))
        .def("siteOf",     &MutualInformationProfile::siteOf,     py::arg("label"))
        .def("snapshotOf", &MutualInformationProfile::snapshotOf, py::arg("label"))
        .def("vertexId",   &MutualInformationProfile::vertexId,   py::arg("site"),
            "Spacetime-vertex ID for a flat-site index. Returns the "
            "flat index itself when the profile was built without "
            "CausetChain labels.")
        .def("weightedAdjacency",
            [](MutualInformationProfile const& p) {
                auto coo = p.weightedAdjacency();
                return py::make_tuple(coo.rows, coo.cols, coo.weights, coo.n);
            },
            R"doc(COO arrays (rows, cols, weights, nVertices) of edges with I > epsilonI.

Each undirected edge appears twice (v→w and w→v).
)doc");

    py::class_<EmergentGraph>(holo, "EmergentGraph",
            R"doc(Weighted graph (V_G, E_G, ℓ_G) on the (site × snapshot) label set.

Edge weights are mutual-information values; the Laplacian L = D - W
follows the convention from
docs/source/holography-causal-ordering-emergent-dimension.md §3.4.
)doc")
        .def(py::init<MutualInformationProfile const&>(), py::arg("profile"))
        .def_property_readonly("nVertices", &EmergentGraph::nVertices)
        .def_property_readonly("nEdges",    &EmergentGraph::nEdges)
        .def("laplacianCOO",
            [](EmergentGraph const& g) {
                // Return the Laplacian as plain COO arrays (rows, cols,
                // values, n). pybind11/eigen.h's sparse-matrix binding
                // currently produces an empty CSC under LTO; the COO
                // path is small and stable.
                auto L = g.laplacian();
                std::vector<int>    rows;
                std::vector<int>    cols;
                std::vector<double> vals;
                rows.reserve(static_cast<std::size_t>(L.nonZeros()));
                cols.reserve(static_cast<std::size_t>(L.nonZeros()));
                vals.reserve(static_cast<std::size_t>(L.nonZeros()));
                for (int k = 0; k < L.outerSize(); ++k) {
                    for (Eigen::SparseMatrix<double>::InnerIterator
                            it(L, k); it; ++it) {
                        rows.push_back(static_cast<int>(it.row()));
                        cols.push_back(static_cast<int>(it.col()));
                        vals.push_back(it.value());
                    }
                }
                return py::make_tuple(rows, cols, vals, g.nVertices());
            },
            R"doc(Weighted Laplacian as a COO tuple (rows, cols, values, n).

Wrap in scipy.sparse for downstream use::

    >>> import scipy.sparse as sp
    >>> rows, cols, vals, n = graph.laplacianCOO()
    >>> L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
)doc")
        .def("returnProbability", &EmergentGraph::returnProbability,
             py::arg("sigmas"), py::arg("krylovDim") = 30,
             py::arg("m") = 0, py::arg("seed") = 0,
             R"doc(P(σ) = (1/|V|) Tr exp(-σ L) via Krylov-Lanczos diagonal estimation.

``m`` is the Hutchinson-style subsample size (issue #28 Tier-1). 0 (default)
uses ``min(n, 3000)`` start vertices, which trades a small variance penalty
for ~100× speedup at large n. Set ``m = n`` for the exact sum.
``seed`` controls the subset RNG for reproducibility.)doc")
        .def_static("spectralDimension", &EmergentGraph::spectralDimension,
             py::arg("sigmas"), py::arg("P"),
             R"doc(D_S(σ) = -2 d log P / d log σ via centered finite differences.)doc")
        .def_static("spectralDimensionSmoothed",
             &EmergentGraph::spectralDimensionSmoothed,
             py::arg("sigmas"), py::arg("P"),
             py::arg("windowSize") = 5, py::arg("polyOrder") = 2,
             R"doc(D_S(σ) via local-polynomial fit on (log σ, log P).

Savitzky-Golay-style smoothing per spec §8: for each grid point, fit
a degree-`polyOrder` polynomial in (log σ, log P) over a centered
window of size `windowSize`, then read the slope at that point.
Defaults match the spec's recommendation (window 5, poly order 2).
)doc")
        .def("toDot", &EmergentGraph::toDot,
             R"doc(Graphviz DOT export. Mirrors Poset.toDot().)doc")
        .def("toGraphML", &EmergentGraph::toGraphML,
             R"doc(GraphML export string; suitable for Gephi / yEd.

Mirrors `tessera.Spacetime.save("*.graphml")`. Edge weights are
exported under the `weight` attribute.
)doc")
        .def_static("fromWeightedEdges",
            [](int n, std::vector<std::tuple<int, int, double>> const& edges) {
                return EmergentGraph::fromWeightedEdges(n, edges);
            },
            py::arg("n"), py::arg("edges"),
            R"doc(Construct an EmergentGraph from a weighted edge list.

`edges` is a list of (u, v, weight) tuples; each undirected edge
should appear once. Used for known-graph acceptance tests (1D chain,
2D lattice, complete graph) per the holography spec §H4.
)doc");

    py::class_<AmbjornLollFit::Result>(holo, "AmbjornLollFitResult")
        .def_readonly("dInfinity",  &AmbjornLollFit::Result::dInfinity)
        .def_readonly("C",          &AmbjornLollFit::Result::C)
        .def_readonly("B",          &AmbjornLollFit::Result::B)
        .def_readonly("chiSquared", &AmbjornLollFit::Result::chiSquared);

    py::class_<AmbjornLollFit>(holo, "AmbjornLollFit",
            R"doc(Static utility: three-parameter D_S(σ) = D_∞ − C / (B + σ) fit.

Stateless; not instantiable. Mirrors the form used by
examples/spectral_dimension.py for CDT comparisons.
)doc")
        .def_static("fit", &AmbjornLollFit::fit,
             py::arg("sigmas"), py::arg("dS"),
             py::arg("sigmaFitMin") = -1.0,
             py::arg("sigmaFitMax") = -1.0);

    py::class_<SpectralDimensionResult>(holo, "SpectralDimensionResult",
            R"doc(Result bundle from EmergentSpectralDimension.compute().

Mirrors the snapshot-style results in tessera.quantum: plain data
container, no MPS/MPO state crosses the boundary.
)doc")
        .def_readonly("sigmas",           &SpectralDimensionResult::sigmas)
        .def_readonly("P",                &SpectralDimensionResult::P)
        .def_readonly("dS",               &SpectralDimensionResult::dS)
        .def_readonly("dSSmoothed",       &SpectralDimensionResult::dSSmoothed)
        .def_readonly("dInfinity",        &SpectralDimensionResult::dInfinity)
        .def_readonly("C",                &SpectralDimensionResult::C)
        .def_readonly("B",                &SpectralDimensionResult::B)
        .def_readonly("fitChiSquared",    &SpectralDimensionResult::fitChiSquared)
        .def_readonly("graphNVertices",   &SpectralDimensionResult::graphNVertices)
        .def_readonly("graphNEdges",      &SpectralDimensionResult::graphNEdges)
        .def_readonly("snapshotTimes",    &SpectralDimensionResult::snapshotTimes)
        .def_readonly("snapshotBondDims", &SpectralDimensionResult::snapshotBondDims)
        .def_readonly("snapshotEnergies", &SpectralDimensionResult::snapshotEnergies)
        .def("toJson",
            [](SpectralDimensionResult const& r, HolographyConfig const& cfg) {
                return r.toJson(cfg);
            },
            py::arg("config"),
            R"doc(Single JSON record matching the schema in spec §10.

Includes the bound config, the TDVP summary, the graph diagnostics,
both raw and smoothed D_S(σ), the Ambjorn-Loll fit, and a provenance
block. Suitable for archiving alongside the experiment results.
)doc");

    // ─── ChoiPropagator (temporal MI engine) ──────────────────────────
    //
    // Exposed for unit-testing the identity-channel acceptance and
    // the single-qubit-unitary checks from the holography spec §H2.
    // The full pipeline reaches the same code via
    // MutualInformationProfile / HolographyConfig.includeTemporal.
    py::class_<ChoiPropagator::TDVPSettings>(holo, "ChoiTDVPSettings",
            R"doc(Sweep settings for the Choi-state TDVP evolution.)doc")
        .def(py::init<>())
        .def_readwrite("dt",         &ChoiPropagator::TDVPSettings::dt)
        .def_readwrite("maxBondDim", &ChoiPropagator::TDVPSettings::maxBondDim)
        .def_readwrite("krylovDim",  &ChoiPropagator::TDVPSettings::krylovDim)
        .def_readwrite("cutoff",     &ChoiPropagator::TDVPSettings::cutoff)
        .def_readwrite("quiet",      &ChoiPropagator::TDVPSettings::quiet);

    py::class_<ChoiPropagator>(holo, "ChoiPropagator",
            R"doc(Static utility for the Choi-state temporal MI.

For a unitary U on N qubits, the Choi state |U⟩ = (U ⊗ I)|Φ+⟩^{⊗N}
encodes the temporal mutual information between (site i at the input
time) and (site j at the output time) as the 2-site reduced-density-
matrix MI on (in_i, out_j). This class exposes the C++ helpers that
build the Choi state under the Schwinger Hamiltonian (acting only on
the output register of an interleaved doubled chain) and extract the
N×N temporal MI matrix.

Not instantiable; call methods on the class.
)doc")
        .def_static("temporalMutualInformation",
            [](SchwingerParams const& p, double duration,
                ChoiPropagator::TDVPSettings const& settings) {
                auto choi = ChoiPropagator::choiState(p, duration, settings);
                return ChoiPropagator::temporalMutualInformation(choi, p.N);
            },
            py::arg("params"), py::arg("duration"), py::arg("settings"),
            R"doc(Temporal MI matrix for the Schwinger propagator over `duration`.

Returns an N×N numpy array; entry (i-1, j-1) is I({in_i} : {out_j})
in nats. At duration = 0 the Choi state is |Φ+⟩^{⊗N} and the matrix
equals 2·ln(2) on the diagonal, zero elsewhere — the identity-channel
acceptance from the spec §H2.
)doc");

    py::class_<SchwingerParams>(holo, "SchwingerParams",
            R"doc(Bare dimensional parameters of the Schwinger Hamiltonian.

Exposed here so the holography test suite can drive ChoiPropagator
directly without going through TDVPConfig.
)doc")
        .def(py::init<>())
        .def_readwrite("N",  &SchwingerParams::N)
        .def_readwrite("a",  &SchwingerParams::a)
        .def_readwrite("m",  &SchwingerParams::m)
        .def_readwrite("g",  &SchwingerParams::g)
        .def_readwrite("L0", &SchwingerParams::L0);

    py::class_<EmergentSpectralDimension>(holo, "EmergentSpectralDimension",
            R"doc(Workflow class: bind a HolographyConfig, run the full pipeline.

DMRG ground state → q-qbar quench → TDVP loop with per-snapshot MI →
weighted (site, time) graph → heat-kernel trace → D_S(σ) →
Ambjorn-Loll fit. Mirrors the SchwingerModel(cfg).solve() and
SchwingerQuench(cfg).evolve() patterns in tessera.quantum.

The recordMutualInformation flag on the underlying TDVPConfig is
forced on by the constructor so callers can't trip themselves up by
leaving it off.
)doc")
        .def(py::init<HolographyConfig>(), py::arg("config"))
        .def_property_readonly("config",
            [](EmergentSpectralDimension const& m) { return m.config(); })
        .def("compute", &EmergentSpectralDimension::compute,
             R"doc(Run the full pipeline; returns SpectralDimensionResult.)doc")
        .def("computeFromSnapshots",
             &EmergentSpectralDimension::computeFromSnapshots,
             py::arg("quench"),
             R"doc(Reuse a single TDVP run across multiple σ-grids or ε_I values.)doc");

    // ─── Koashi-Imoto + QuantumSimplex ─────────────────────────────────
    m.def("partialTraceA", &::tessera::quantum::partialTraceA,
          py::arg("rhoAB"), py::arg("dimA"), py::arg("dimB"),
          R"doc(Partial trace over the A factor of a bipartite ρ_AB.

rhoAB is a (dimA * dimB) x (dimA * dimB) matrix in (A ⊗ B) ordering
(row index = a * dimB + b). Returns a dimB x dimB density matrix.)doc");

    m.def("partialTraceB", &::tessera::quantum::partialTraceB,
          py::arg("rhoAB"), py::arg("dimA"), py::arg("dimB"),
          R"doc(Partial trace over the B factor of a bipartite ρ_AB.

Returns a dimA x dimA density matrix.)doc");

    m.def("mutualInformation",
          py::overload_cast<const Eigen::MatrixXcd&, int, int>(
              &::tessera::quantum::mutualInformation),
          py::arg("rhoAB"), py::arg("dimA"), py::arg("dimB"),
          R"doc(Mutual information I(A:B) = S(A) + S(B) - S(AB) in nats.

Floors at 0. Marginals are computed by partial trace from ρ_AB.)doc");

    m.def("mutualInformation",
          py::overload_cast<const Eigen::MatrixXcd&,
                            const Eigen::MatrixXcd&,
                            const Eigen::MatrixXcd&>(
              &::tessera::quantum::mutualInformation),
          py::arg("rhoAB"), py::arg("rhoA"), py::arg("rhoB"),
          R"doc(Mutual information I(A:B) = S(A) + S(B) - S(AB) with
explicit marginals. Use this when ρ_A and ρ_B are already on hand
(e.g. on QuantumVertex objects) — it skips the partial-trace step.)doc");

    py::class_<::tessera::quantum::KoashiImotoTolerances>(m,
            "KoashiImotoTolerances",
            R"doc(Numerical tolerances for the symmetric KI decomposition.

Each defaults to 1e-10. Tightening or relaxing affects the block /
cond-state clustering and so the resolved L/R structure.)doc")
        .def(py::init<>())
        .def(py::init<double, double, double>(),
             py::arg("epsKiEigen"), py::arg("epsKiCondState"),
             py::arg("epsKiSvd"))
        .def_property("epsKiEigen",
                      &::tessera::quantum::KoashiImotoTolerances::getEpsKiEigen,
                      &::tessera::quantum::KoashiImotoTolerances::setEpsKiEigen)
        .def_property("epsKiCondState",
                      &::tessera::quantum::KoashiImotoTolerances::getEpsKiCondState,
                      &::tessera::quantum::KoashiImotoTolerances::setEpsKiCondState)
        .def_property("epsKiSvd",
                      &::tessera::quantum::KoashiImotoTolerances::getEpsKiSvd,
                      &::tessera::quantum::KoashiImotoTolerances::setEpsKiSvd);

    py::class_<::tessera::quantum::KoashiImotoBlock>(m, "KoashiImotoBlock",
            R"doc(A single j-block of the symmetric KI decomposition.

Outputs of ``koashiImotoDecompose``; immutable.)doc")
        .def_property_readonly("weight",
                      &::tessera::quantum::KoashiImotoBlock::getWeight)
        .def_property_readonly("coreState",
                      &::tessera::quantum::KoashiImotoBlock::getCoreState)
        .def_property_readonly("tailA",
                      &::tessera::quantum::KoashiImotoBlock::getTailA)
        .def_property_readonly("tailB",
                      &::tessera::quantum::KoashiImotoBlock::getTailB)
        .def_property_readonly("dimLeftA",
                      &::tessera::quantum::KoashiImotoBlock::getDimLeftA)
        .def_property_readonly("dimLeftB",
                      &::tessera::quantum::KoashiImotoBlock::getDimLeftB)
        .def_property_readonly("dimRightA",
                      &::tessera::quantum::KoashiImotoBlock::getDimRightA)
        .def_property_readonly("dimRightB",
                      &::tessera::quantum::KoashiImotoBlock::getDimRightB);

    py::class_<::tessera::quantum::KoashiImotoResult>(m,
            "KoashiImotoResult",
            R"doc(Result of the symmetric Koashi-Imoto decomposition.

Holds the three child matrices (sigma = the joint core; aPrime, bPrime
= the uncorrelated tails) and the per-block breakdown.)doc")
        .def_property_readonly("sigma",
                      &::tessera::quantum::KoashiImotoResult::getSigma)
        .def_property_readonly("aPrime",
                      &::tessera::quantum::KoashiImotoResult::getAPrime)
        .def_property_readonly("bPrime",
                      &::tessera::quantum::KoashiImotoResult::getBPrime)
        .def_property_readonly("blocks",
                      &::tessera::quantum::KoashiImotoResult::getBlocks);

    m.def("koashiImotoDecompose",
          py::overload_cast<const Eigen::MatrixXcd&, int, int,
                            const ::tessera::quantum::KoashiImotoTolerances&>(
              &::tessera::quantum::koashiImotoDecompose),
          py::arg("rhoAB"), py::arg("dimA"), py::arg("dimB"),
          py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
          R"doc(Symmetric Koashi-Imoto decomposition of a bipartite ρ_AB.

Returns a KoashiImotoResult with the three child matrices and the
per-block breakdown. Marginals are extracted from ρ_AB by partial
trace.)doc");

    m.def("koashiImotoDecompose",
          py::overload_cast<const Eigen::MatrixXcd&,
                            const Eigen::MatrixXcd&,
                            const Eigen::MatrixXcd&,
                            const ::tessera::quantum::KoashiImotoTolerances&>(
              &::tessera::quantum::koashiImotoDecompose),
          py::arg("rhoAB"), py::arg("rhoA"), py::arg("rhoB"),
          py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
          R"doc(Symmetric Koashi-Imoto decomposition with explicit
marginals ρ_A and ρ_B (preferred when the marginals are already on
hand — avoids the partial-trace step's numerical drift).)doc");

    m.def("createQuantumVertex",
          [](::tessera::spacetime::Spacetime& st,
             Eigen::MatrixXcd                 state) {
              const auto id = st.reserveVertexId();
              auto vlist = st.getVertexList();
              return vlist->template addAs<::tessera::quantum::QuantumVertex>(
                  id, id, std::move(state));
          },
          py::arg("spacetime"), py::arg("state"),
          py::return_value_policy::reference,
          R"doc(Allocate a new QuantumVertex in the spacetime's vertex
list, carrying the given density matrix. The vertex id is assigned
from the spacetime's counter. The returned pointer is owned by the
spacetime.)doc");

    py::class_<::tessera::quantum::QuantumVertex,
               ::tessera::mesh::Vertex,
               std::unique_ptr<::tessera::quantum::QuantumVertex,
                               py::nodelete>>(m, "QuantumVertex",
            R"doc(A mesh.Vertex carrying a density matrix.

QuantumVertex extends mesh.Vertex with an Eigen-typed density
matrix in its local Hilbert space. The matrix dimension is fixed
at construction time and may differ across QuantumVertex objects
in the same VertexList (the KI factories use this to give A, B, Σ,
A', B' their own per-block dimensions).

Construct via ``createQuantumVertex(spacetime, state)`` (which
allocates one inside the spacetime's vertex list) or directly via
``QuantumVertex(id, state)`` for free-standing use.)doc")
        .def(py::init<std::uint64_t, Eigen::MatrixXcd>(),
             py::arg("id"), py::arg("state"))
        .def("getState",
             &::tessera::quantum::QuantumVertex::getState,
             py::return_value_policy::reference_internal,
             R"doc(Return the density matrix ρ on this vertex.)doc")
        .def("setState",
             &::tessera::quantum::QuantumVertex::setState,
             py::arg("state"),
             R"doc(Replace the density matrix.)doc")
        .def("stateDim",
             &::tessera::quantum::QuantumVertex::stateDim,
             R"doc(Return the Hilbert-space dimension of ρ.)doc")
        .def("vanRaamsdonkDistanceTo",
             &::tessera::quantum::QuantumVertex::vanRaamsdonkDistanceTo,
             py::arg("other"), py::arg("iMax"),
             R"doc(Van Raamsdonk distance d_VR = -log(I / iMax)
between this vertex and ``other`` (which must also be a
QuantumVertex). I is the mutual information of the product joint
ρ_self ⊗ ρ_other — i.e. assumes the two vertices carry no
inherited correlation. Returns +∞ when I = 0.

The KI cell factory uses this for the nine non-(A, B) edges and
computes d_VR for the (A, B) edge directly from the input joint
ρ_AB.)doc");

    py::class_<::tessera::quantum::QuantumSimplex>(m, "QuantumSimplex",
            R"doc(Static-only utility: KI factories for a 5-vertex
KI-interaction cell.

QuantumSimplex is not a separate runtime type — it is a namespace
for the four KI factory entry points that build a regular
mesh.Simplex (five vertices, ten edges) inside a Spacetime from
two pre-existing QuantumVertex inputs. The returned mesh.Simplex
is owned by the Spacetime; the per-vertex ρ lives on the
QuantumVertex objects in the vertex list; the per-edge d_VR² is
stored in the standard Edge squaredLength field.

iMax is global to the simulation and is passed to each factory
call — it is not stored on the simplex.)doc")
        .def_static("fromSchmidtPurification",
            &::tessera::quantum::QuantumSimplex::fromSchmidtPurification,
            py::arg("spacetime"),
            py::arg("qva"),
            py::arg("qvb"),
            py::arg("iMax"),
            py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
            py::return_value_policy::reference,
            R"doc(Build ρ_AB = |ψ⟩⟨ψ| with
|ψ⟩ = Σ_i √λ_i |a_i⟩|b_i⟩ from matched marginal spectra of
ρ_A on qva and ρ_B on qvb, then construct the cell.)doc")
        .def_static("fromClassicalCorrelation",
            &::tessera::quantum::QuantumSimplex::fromClassicalCorrelation,
            py::arg("spacetime"),
            py::arg("qva"),
            py::arg("qvb"),
            py::arg("iMax"),
            py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
            py::return_value_policy::reference,
            R"doc(Build the perfectly-correlated classical joint
ρ_AB = Σ_i λ_i |a_i⟩⟨a_i| ⊗ |b_i⟩⟨b_i| in matched eigenbases.)doc")
        .def_static("fromExplicitJoint",
            &::tessera::quantum::QuantumSimplex::fromExplicitJoint,
            py::arg("spacetime"),
            py::arg("qva"),
            py::arg("qvb"),
            py::arg("rhoAB"),
            py::arg("iMax"),
            py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
            py::return_value_policy::reference,
            R"doc(Build the cell from a caller-supplied joint
ρ_AB. The partial traces of ρ_AB must agree with the marginals
on qva, qvb.)doc")
        .def_static("fromTargetMutualInformation",
            &::tessera::quantum::QuantumSimplex::fromTargetMutualInformation,
            py::arg("spacetime"),
            py::arg("qva"),
            py::arg("qvb"),
            py::arg("targetMI"),
            py::arg("iMax"),
            py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
            py::return_value_policy::reference,
            R"doc(Binary-search α in
ρ_AB(α) = (1-α)·(ρ_A ⊗ ρ_B) + α·ρ_AB^Schmidt
to hit ``targetMI``. Requires matched spectra of ρ_A, ρ_B; throws
if ``targetMI`` lies outside [0, 2·H(λ)].)doc");

    py::enum_<::tessera::quantum::QuantumSimplex::Position>(
            m, "QuantumSimplexPosition")
        .value("A",      ::tessera::quantum::QuantumSimplex::A)
        .value("B",      ::tessera::quantum::QuantumSimplex::B)
        .value("Sigma",  ::tessera::quantum::QuantumSimplex::Sigma)
        .value("APrime", ::tessera::quantum::QuantumSimplex::APrime)
        .value("BPrime", ::tessera::quantum::QuantumSimplex::BPrime);
}
