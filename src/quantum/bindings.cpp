// Pybind11 bindings for the quantum subsystem. Lives outside libtessera_quantum
// (which is pybind-free) so the static library can be reused without pulling
// in the Python dependency. This translation unit is added to _tessera's
// sources only when TESSERA_QUANTUM=ON in CMakeLists.txt.
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

#include "quantum/causal_compare.hpp"
#include "quantum/causet_chain.hpp"
#include "quantum/dmrg_runner.hpp"
#include "quantum/majorization.hpp"
#include "quantum/schmidt.hpp"
#include "quantum/tdvp_runner.hpp"
#include "spacetime/Spacetime.h"  // full type needed for py::cast<Spacetime*>()

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_quantum_bindings(py::module_ m) {
    using namespace tessera::quantum;

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

Nodes are integers 0 .. getNodeCount - 1. ``covers`` lists the cover
edges: each entry (a, b) means a ≻ b with no intermediate node.
)doc")
        .def(py::init<>())
        .def_property("getNodeCount",
            [](Poset const& p) { return p.getNodeCount(); },
            [](Poset& p, int n) { p.setNodeCount(n); })
        .def_property("covers",
            [](Poset const& p) { return p.covers(); },
            [](Poset& p, std::vector<std::pair<int, int>> const& covers) {
                p.setCovers(covers);
            })
        .def("toDot", &Poset::toDot,
            "Graphviz DOT representation of the Hasse diagram.")
        .def_static("fromSpacetime",
            [](py::object spacetime_obj) {
                auto const* st = spacetime_obj.cast<tessera::Spacetime const*>();
                return tessera::Poset::fromSpacetime(*st);
            }, py::arg("spacetime"),
            R"doc(Inherit a Hasse-cover Poset from a tessera.Spacetime.)doc")
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
        .def_readwrite("recordPoset",             &TDVPConfig::recordPoset);

    py::class_<TDVPSnapshot>(m, "TDVPSnapshot",
            R"doc(Per-step diagnostics recorded during a TDVP run.

Always-populated fields: time, energy, bondDim, zProfile, lProfile.
Optional fields (populated only if the corresponding TDVPConfig flag
is set): spectra, poset.
)doc")
        .def_readonly("time",      &TDVPSnapshot::time)
        .def_readonly("energy",    &TDVPSnapshot::energy)
        .def_readonly("bondDim",   &TDVPSnapshot::bondDim)
        .def_readonly("zProfile",  &TDVPSnapshot::zProfile)
        .def_readonly("lProfile",  &TDVPSnapshot::lProfile)
        .def_readonly("spectra",   &TDVPSnapshot::spectra)
        .def_readonly("poset",     &TDVPSnapshot::poset)
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
                auto const* st = spacetime_obj.cast<tessera::Spacetime const*>();
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
}
