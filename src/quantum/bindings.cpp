// Pybind11 bindings for the quantum subsystem. Lives outside libtessera_quantum
// (which is pybind-free) so the static library can be reused without pulling
// in the Python dependency. This translation unit is added to _tessera's
// sources only when TESSERA_QUANTUM=ON in CMakeLists.txt.
//
// Surface area is deliberately minimal per PLAN.md §1: scalars in, scalars
// out. No MPS / MPO / ITensor types cross the Python boundary. If callers
// need the wave function itself they should add a C++-side observable
// callback that returns plain data (Phase 4's approach for TDVP snapshots).
//
// Docstrings here are deliberately verbose because they're the primary
// user-facing documentation — `help(tessera.quantum.computeGroundState)`
// in a Python REPL or notebook should be enough to understand both the
// API and the underlying physics conventions.

#include "quantum/causal_compare.hpp"
#include "quantum/causet_chain.hpp"
#include "quantum/dmrg_runner.hpp"
#include "quantum/majorization.hpp"
#include "quantum/pipeline.hpp"
#include "quantum/quench.hpp"
#include "quantum/schmidt.hpp"
#include "quantum/tdvp_runner.hpp"
#include "spacetime/Spacetime.h"  // full type needed for py::cast<Spacetime*>()

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_quantum_bindings(py::module_ m) {
    using namespace tessera::quantum;

    m.doc() = R"doc(
Schwinger model + DMRG (Phase 2 of docs/source/quantum-plan.md).

This submodule exposes a single end-to-end pipeline: feed in a
:class:`QuantumConfig` describing the dimensional Hamiltonian and DMRG
sweep schedule; get back a :class:`GroundStateResult` with the converged
ground-state energy and the bond-dim/truncation diagnostics needed to
judge convergence.

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

The DMRG runs on the U(1) charge-neutral (total Sz = 0) sector by default.

References
----------
* Bañuls, Cichy, Cirac, Jansen, *JHEP* **11**, 158 (2013),
  arXiv:1305.3765 — primary reference for the Hamiltonian and benchmarks.
* Schwinger, *Phys. Rev.* **128**, 2425 (1962) — original gauge theory.
* Coleman, *Ann. Phys.* **101**, 239 (1976) — massive Schwinger model.
)doc";

    py::class_<QuantumConfig>(m, "QuantumConfig",
            R"doc(Configuration for a Schwinger-model DMRG ground-state run.

Bundles the dimensional Hamiltonian parameters and the DMRG sweep
settings into a single struct so calling code only hands one Python
object across the C++ boundary. Default-constructed instances have
N = 0 and must be filled in before passing to :func:`computeGroundState`.

The dt and T fields are reserved for Phase 4 (real-time evolution
after a quark/antiquark quench) and are ignored by
:func:`computeGroundState`.

Attributes
----------
N : int
    Staggered sites, 1-based indexing. Must be ≥ 2; even N is required
    if you want the global GS to live in the Sz = 0 sector (which is
    the standard Bañuls 2013 convention).
a : float
    Lattice spacing. Sets the units of the Hamiltonian — values in the
    output are in those same units. Must be positive. Default 1.0.
m : float
    Bare staggered-fermion mass. Sign matters: positive m with our
    (-1)^n convention favours the Néel pattern with σ^z_n = (-1)^(n+1).
g : float
    Gauge coupling. May be 0 (free-Dirac limit, gauge field decouples).
    Bañuls' dimensionless coupling is x = 1/(g²a²).
L0 : float
    Background electric field on the link to the left of site 1.
    Default 0 puts us in the standard zero-background sector.
maxBondDim : int
    Cap on the MPS bond dimension during DMRG sweeps. Schwinger ground
    states at moderate (m, g, N) typically converge at bond dim ~10-30
    even for N = 100+. Bond-dim-limited runs report ``bondDim`` equal
    to this value in the result.
nSweeps : int
    Total number of DMRG sweeps. Energy convergence is exponential in
    sweep count once bond-dim is sufficient; 12 sweeps is plenty for
    moderate-N tests.
cutoff : float
    SVD truncation threshold per local solve. 1e-12 is conservative for
    these Hamiltonians and well below double-precision noise. Tighter
    cutoff costs sweep time without meaningful precision gain.
krylovDim : int
    Lanczos / Krylov dimension per local 2-site solve. Default 4 is
    fine for gapped problems; raise for harder cases (small mass,
    near-critical points) where eigensolver convergence is slower.
quiet : bool
    If True (default), suppress ITensor's per-sweep diagnostic prints.
    Set False to debug DMRG convergence.
conserveQns : bool
    If True (default), the SiteSet enforces total-Sz QN conservation
    on every MPS bond. The Néel initial state then pins DMRG to the
    Sz = 0 sector. Set False only when you need to apply σ^x / σ^y
    later (these change Sz) — currently exercised by Phase 1's
    charge-conjugation test.
dt : float
    *(Phase 4)* TDVP real-time step size. Unused by
    computeGroundState.
T : float
    *(Phase 4)* Total evolution time. Unused by computeGroundState.

Examples
--------
>>> from tessera.quantum import QuantumConfig
>>> cfg = QuantumConfig()
>>> cfg.N = 20
>>> cfg.a = 1.0
>>> cfg.g = 1.0
>>> cfg.m = 0.0
>>> cfg.L0 = 0.0
>>> cfg.maxBondDim = 100
>>> cfg.nSweeps = 12
)doc")
        .def(py::init<>())
        .def_readwrite("N",            &QuantumConfig::N,
            "Staggered sites, 1-based; must be ≥ 2.")
        .def_readwrite("a",            &QuantumConfig::a,
            "Lattice spacing. Must be positive. Default 1.0.")
        .def_readwrite("m",            &QuantumConfig::m,
            "Bare fermion mass.")
        .def_readwrite("g",            &QuantumConfig::g,
            "Gauge coupling. g = 0 is the free-Dirac limit (allowed).")
        .def_readwrite("L0",           &QuantumConfig::L0,
            "Background electric field on the link left of site 1.")
        .def_readwrite("maxBondDim", &QuantumConfig::maxBondDim,
            "Cap on MPS bond dimension during DMRG.")
        .def_readwrite("nSweeps",     &QuantumConfig::nSweeps,
            "Total DMRG sweep count.")
        .def_readwrite("cutoff",       &QuantumConfig::cutoff,
            "SVD truncation threshold per local solve.")
        .def_readwrite("krylovDim",   &QuantumConfig::krylovDim,
            "Lanczos / Krylov dimension per local solve.")
        .def_readwrite("quiet",        &QuantumConfig::quiet,
            "Suppress ITensor's per-sweep diagnostic output.")
        .def_readwrite("conserveQns", &QuantumConfig::conserveQns,
            "U(1) total-Sz conservation on the SiteSet.")
        .def_readwrite("dt",           &QuantumConfig::dt,
            "(Phase 4) Real-time step size; unused by computeGroundState.")
        .def_readwrite("T",            &QuantumConfig::T,
            "(Phase 4) Total evolution time; unused by computeGroundState.")
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
            R"doc(Result of a Schwinger-model DMRG ground-state run.

The full physical energy is ``energy``. The other fields let callers
sanity-check convergence:

* ``operatorEnergy + constant`` should equal ``energy`` exactly
  (assertable to ~1e-12).
* ``bondDim`` hitting ``config.maxBondDim`` flags a bond-dim-limited
  run — increase the cap and rerun if you need tighter convergence.
* ``truncationErr`` is a conservative upper bound (the configured
  cutoff). Real truncation is typically much smaller.

Attributes
----------
energy : float
    Full physical energy ⟨H⟩ + constant. Use this when comparing to
    Bañuls-style published numerics (after the dimensional rescaling
    E_W = (2/(ag²)) E_dim if you want their dimensionless W).
operatorEnergy : float
    ⟨H⟩ alone, as returned by ITensor's dmrg(). Use this for direct
    cross-checks against dense diagonalization of the operator-valued
    Hamiltonian (without the c-number L_n² shift).
constant : float
    The c-number shift (g²a/2) Σ_n (c_n² + n/4) pulled out of L_n²
    when AutoMPO-encoding the electric term — see the derivation in
    src/quantum/schwinger_model.cpp. ``operatorEnergy + constant ==
    energy`` by construction.
bondDim : int
    Largest bond dimension of the optimized MPS. Equals
    ``config.maxBondDim`` if the run was bond-dim-limited.
truncationErr : float
    Conservative upper bound on the SVD truncation error in the final
    sweep (currently equal to the configured ``cutoff``).
)doc")
        .def_readonly("energy",          &GroundStateResult::energy,
            "Full physical energy ⟨H⟩ + constant.")
        .def_readonly("operatorEnergy", &GroundStateResult::operatorEnergy,
            "Operator part ⟨H⟩, matches ITensor's dmrg() return value.")
        .def_readonly("constant",        &GroundStateResult::constant,
            "C-number shift from L_n² expansion.")
        .def_readonly("bondDim",        &GroundStateResult::bondDim,
            "Largest MPS bond dimension after the final sweep.")
        .def_readonly("truncationErr",  &GroundStateResult::truncationErr,
            "Conservative upper bound on the truncation error.")
        .def("__repr__", [](GroundStateResult const& r) {
            return "GroundStateResult(energy=" + std::to_string(r.energy) +
                   ", bondDim=" + std::to_string(r.bondDim) +
                   ", truncationErr=" + std::to_string(r.truncationErr) + ")";
        });

    // ─── Phase 3 types ──────────────────────────────────────────────────
    py::class_<Interval>(m, "Interval",
            R"doc(1-based contiguous interval [i, j] on the spin chain.

Used as a label on the Schmidt spectra returned by
:func:`computeGroundStateMajorization`.

Attributes
----------
i : int
    First site of the interval (1-based).
j : int
    Last site of the interval (1-based, j ≥ i).
)doc")
        .def(py::init<>())
        .def_readwrite("i", &Interval::i, "First site of the interval (1-based).")
        .def_readwrite("j", &Interval::j, "Last site of the interval (1-based, j ≥ i).")
        .def("__repr__", [](Interval const& iv) {
            return "Interval(i=" + std::to_string(iv.i) +
                   ", j=" + std::to_string(iv.j) + ")";
        });

    py::class_<SchmidtSpectra>(m, "SchmidtSpectra",
            R"doc(All contiguous-cut Schmidt spectra of an MPS.

Cuts are indexed 0 .. len(spectra)-1; ``intervals[k]`` is the
:class:`Interval` label of ``spectra[k]``. The full-chain bipartition
[1, N] | ∅ is excluded.

Each spectrum is sorted non-increasingly and contains the eigenvalues of
ρ_A (= squared Schmidt coefficients), with no zero-padding.

Attributes
----------
N : int
    Length of the spin chain.
intervals : list[Interval]
    Cut labels.
spectra : list[list[float]]
    ``spectra[k]`` is the entanglement spectrum of bipartition
    ``intervals[k]`` | rest.
)doc")
        .def_readonly("N",         &SchmidtSpectra::N)
        .def_readonly("intervals", &SchmidtSpectra::intervals)
        .def_readonly("spectra",   &SchmidtSpectra::spectra);

    py::class_<Poset>(m, "Poset",
            R"doc(Hasse / cover representation of a finite partial order.

Nodes are integers 0 .. getNodeCount - 1. ``covers`` lists the cover edges
(transitive reduction of the strict-majorization graph): each entry
(a, b) means a ≻ b with no third node c satisfying a ≻ c ≻ b.

Attributes
----------
getNodeCount : int
    Total node count.
covers : list[tuple[int, int]]
    Hasse cover edges (a, b) with a ≻ b.
)doc")
        .def(py::init<>())
        // getNodeCount and covers are accessed via getters/setters because the
        // underlying tessera::Poset stores them in VertexList/EdgeList rather
        // than as raw int + vector members. Python sees the same simple
        // {getNodeCount: int, covers: list[tuple[int,int]]} surface as before.
        .def_property("getNodeCount",
            [](Poset const& p) { return p.getNodeCount(); },
            [](Poset& p, int n) { p.setNodeCount(n); },
            "Total node count.")
        .def_property("covers",
            [](Poset const& p) { return p.covers(); },
            [](Poset& p, std::vector<std::pair<int, int>> const& covers) {
                p.setCovers(covers);
            },
            "Hasse cover edges (a, b) with a strictly majorizes b.")
        .def("toDot", &Poset::toDot,
            "Graphviz DOT representation of the Hasse diagram.")
        // Phase 6 — Spacetime → Poset factory. Same py::object indirection
        // as extractCausetChain because Spacetime is registered in the
        // top-level _tessera module, not in this quantum sub-binding TU.
        .def_static("fromSpacetime",
            [](py::object spacetime_obj) {
                auto const* st = spacetime_obj.cast<tessera::Spacetime const*>();
                return tessera::Poset::fromSpacetime(*st);
            }, py::arg("spacetime"),
            R"doc(Inherit a Hasse-cover Poset from a tessera.Spacetime.

Walks the Spacetime's edge list, takes every timelike edge
(``Edge.getSquaredLength() < 0``) as a strict precedes-relation
oriented earliest-time → latest-time, computes the transitive
closure, and emits the cover edges (transitive reduction) as the
Poset's cover list.

Vertex IDs in the returned Poset are a dense 0..n-1 remapping of the
Spacetime's ``Vertex.getId()`` values in ascending order. The
remapping is monotonic, so a caller who sorts Spacetime vertex IDs
ascending recovers the Poset node order trivially.

Parameters
----------
spacetime : tessera.Spacetime
    Source spacetime.

Returns
-------
Poset
)doc")
        .def("__repr__", [](Poset const& p) {
            return "Poset(getNodeCount=" + std::to_string(p.getNodeCount()) +
                   ", covers=" + std::to_string(p.getCoverCount()) + " edges)";
        });

    py::class_<GroundStateMajorizationResult>(m, "GroundStateMajorizationResult",
            R"doc(Result of :func:`computeGroundStateMajorization`.

Bundles the scalar DMRG diagnostics with the Schmidt spectra of the
optimized ground-state MPS and the majorization poset on those spectra.

Attributes
----------
groundState : GroundStateResult
    Same diagnostics returned by :func:`computeGroundState`.
spectra : SchmidtSpectra
    All contiguous-cut Schmidt spectra of the ground-state MPS,
    excluding the trivial full-chain cut.
poset : Poset
    Hasse cover edges of the strict-majorization order on
    ``spectra.spectra``. Cover (a, b) means
    ``spectra.spectra[a] ≻ spectra.spectra[b]`` with no intermediate
    spectrum.
)doc")
        .def_readonly("groundState", &GroundStateMajorizationResult::groundState)
        .def_readonly("spectra",      &GroundStateMajorizationResult::spectra)
        .def_readonly("poset",        &GroundStateMajorizationResult::poset);

    // ─── Phase 3 free functions ─────────────────────────────────────────
    m.def("majorizes",
          &majorizes,
          py::arg("mu"), py::arg("lambda_"), py::arg("tol") = 1e-12,
          R"doc(Test whether μ majorizes λ.

Both vectors are sorted non-increasingly internally and zero-padded to
the longer's length. μ majorizes λ iff every cumulative partial sum
of the sorted-padded μ is at least as large as that of λ, AND the
total masses match within ``tol``.

Parameters
----------
mu, lambda_ : list[float]
    Probability-like distributions (or any non-negative sequences with
    equal total mass within ``tol``).
tol : float, optional
    Slack for partial-sum comparisons and total-mass equality.

Returns
-------
bool
    True if μ majorizes λ.
)doc");

    m.def("strictlyMajorizes",
          &strictlyMajorizes,
          py::arg("mu"), py::arg("lambda_"), py::arg("tol") = 1e-12,
          R"doc(Test whether μ strictly majorizes λ.

Equivalent to ``majorizes(mu, lambda_) and not majorizes(lambda_, mu)``
— the relation is proper, not just sorted-padded equality.
)doc");

    m.def("majorizationPoset",
          &majorizationPoset,
          py::arg("spectra"), py::arg("tol") = 1e-12,
          R"doc(Build the majorization poset on a list of spectra.

``spectra[k]`` becomes node k; the resulting :class:`Poset` stores the
Hasse cover edges only (transitive closure is implicit).

Parameters
----------
spectra : list[list[float]]
    Probability-like distributions, one per node.
tol : float, optional
    Slack passed to the underlying majorizes() comparisons.

Returns
-------
Poset
    Hasse cover representation of the strict-majorization order.

Notes
-----
Complexity is O(M^3) where M = ``len(spectra)``, dominated by the
transitive-reduction pass.
)doc");

    // ─── Phase 4 types: TDVP / quench ───────────────────────────────────
    py::class_<TDVPConfig>(m, "TDVPConfig",
            R"doc(Configuration for the Phase 4 q-qbar quench + TDVP run.

Bundles the Hamiltonian parameters, the DMRG ground-state setup, the
quench location / separation, and the real-time-evolution schedule.
``i0`` and ``d`` describe the q-qbar pair: σ⁻ acts at site ``i0``,
σ⁺ at site ``i0 + d``.

The parity constraint (PLAN.md §5 Phase 4 / quench.hpp): for the
heavy-quark Néel vacuum to admit a non-trivial flux tube, ``i0`` must
be odd (Up sublattice) and ``d`` must be odd. Set
``quenchEnforceParity = False`` to override.

Attributes
----------
N, a, m, g, L0 : Hamiltonian parameters (same as :class:`QuantumConfig`).
dmrgMaxBondDim, dmrgNSweeps, dmrgKrylovDim, dmrgCutoff :
    DMRG sweep settings for the initial ground state.
i0, d : int
    First site of the q-qbar pair (1-based) and separation in lattice sites.
quenchEnforceParity : bool
    If True, the quench raises an exception unless i0 and d satisfy the
    parity constraint above.
dt : float
    Real-time step (units of 1/E_dim). Total evolution is ``T``; the loop
    performs ``round(T / dt)`` TDVP steps.
T : float
    Total real-time evolution time.
maxBondDim, krylovDim, cutoff :
    TDVP sweep settings (per-step bond cap, Krylov dim, SVD cutoff).
snapshotEvery : int
    Record observables every k steps (≥ 1). The initial snapshot at
    t = 0 (post-quench) is always recorded.
quiet, conserveQns : bool
recordSpectra, recordPoset : bool
    Optional per-snapshot recording of full Schmidt spectra and the
    majorization poset (O(N²) SVDs each — off by default).
)doc")
        .def(py::init<>())
        .def_readwrite("N",                       &TDVPConfig::N)
        .def_readwrite("a",                       &TDVPConfig::a)
        .def_readwrite("m",                       &TDVPConfig::m)
        .def_readwrite("g",                       &TDVPConfig::g)
        .def_readwrite("L0",                      &TDVPConfig::L0)
        .def_readwrite("dmrgMaxBondDim",       &TDVPConfig::dmrgMaxBondDim)
        .def_readwrite("dmrgNSweeps",           &TDVPConfig::dmrgNSweeps)
        .def_readwrite("dmrgKrylovDim",         &TDVPConfig::dmrgKrylovDim)
        .def_readwrite("dmrgCutoff",             &TDVPConfig::dmrgCutoff)
        .def_readwrite("i0",                      &TDVPConfig::i0)
        .def_readwrite("d",                       &TDVPConfig::d)
        .def_readwrite("quenchEnforceParity",   &TDVPConfig::quenchEnforceParity)
        .def_readwrite("dt",                      &TDVPConfig::dt)
        .def_readwrite("T",                       &TDVPConfig::T)
        .def_readwrite("maxBondDim",            &TDVPConfig::maxBondDim)
        .def_readwrite("krylovDim",              &TDVPConfig::krylovDim)
        .def_readwrite("cutoff",                  &TDVPConfig::cutoff)
        .def_readwrite("snapshotEvery",          &TDVPConfig::snapshotEvery)
        .def_readwrite("quiet",                   &TDVPConfig::quiet)
        .def_readwrite("conserveQns",            &TDVPConfig::conserveQns)
        .def_readwrite("recordSpectra",          &TDVPConfig::recordSpectra)
        .def_readwrite("recordPoset",            &TDVPConfig::recordPoset);

    py::class_<TDVPSnapshot>(m, "TDVPSnapshot",
            R"doc(Per-step diagnostics recorded during a TDVP run.

Always-populated fields:

    time : float            -- elapsed real time
    energy : float          -- ⟨ψ(t)|H|ψ(t)⟩ + constant
    bondDim : int          -- maxLinkDim of the MPS at this time
    zProfile : list[float] -- ⟨σ^z_n⟩ for n = 1..N
    lProfile : list[float] -- ⟨L_n⟩  for n = 1..N-1

Optional fields (populated only if the corresponding TDVPConfig flag is set):

    spectra : SchmidtSpectra
    poset : Poset
)doc")
        .def_readonly("time",       &TDVPSnapshot::time)
        .def_readonly("energy",     &TDVPSnapshot::energy)
        .def_readonly("bondDim",   &TDVPSnapshot::bondDim)
        .def_readonly("zProfile",  &TDVPSnapshot::zProfile)
        .def_readonly("lProfile",  &TDVPSnapshot::lProfile)
        .def_readonly("spectra",    &TDVPSnapshot::spectra)
        .def_readonly("poset",      &TDVPSnapshot::poset)
        .def("__repr__", [](TDVPSnapshot const& s) {
            return "TDVPSnapshot(time=" + std::to_string(s.time) +
                   ", energy=" + std::to_string(s.energy) +
                   ", bondDim=" + std::to_string(s.bondDim) + ")";
        });

    py::class_<QuenchResult>(m, "QuenchResult",
            R"doc(Result of :func:`runQqbarQuench`.

Attributes
----------
groundState : GroundStateResult
    DMRG ground-state diagnostics for the pre-quench state.
snapshots : list[TDVPSnapshot]
    Per-step diagnostics. ``snapshots[0]`` is the post-quench state at
    t = 0; the rest are spaced every ``config.snapshotEvery`` TDVP
    steps. The final TDVP step is always recorded.
)doc")
        .def_readonly("groundState", &QuenchResult::groundState)
        .def_readonly("snapshots",    &QuenchResult::snapshots);

    // ─── Phase 5 types: causal-order comparison ─────────────────────────
    py::class_<LabelSpacetime>(m, "LabelSpacetime",
            R"doc(One label in a (cut, time) spacetime.

Used as a node in the Phase 5 causal-comparison posets. Carries the
contiguous interval label, the snapshot index it belongs to, and the
physical time of that snapshot.

Attributes
----------
cutIdx : int
    Index of the cut in the snapshot's spectra/intervals list.
tIdx : int
    Index of the snapshot in the QuenchResult's snapshot list.
intervalI, intervalJ : int
    The contiguous interval [intervalI, intervalJ] (1-based).
time : float
    Physical time of the snapshot.
)doc")
        .def_readonly("cutIdx",     &LabelSpacetime::cutIdx)
        .def_readonly("tIdx",       &LabelSpacetime::tIdx)
        .def_readonly("intervalI",  &LabelSpacetime::intervalI)
        .def_readonly("intervalJ",  &LabelSpacetime::intervalJ)
        .def_readonly("time",        &LabelSpacetime::time);

    py::class_<CausalOrders>(m, "CausalOrders",
            R"doc(Three Hasse-cover posets on a shared (cut, time) label set.

Attributes
----------
labels : list[LabelSpacetime]
    The shared node set. ``labels[k]`` is node k.
maj : Poset
    Strict-majorization order from Phase 3, applied across cuts and times.
lr : Poset
    Lieb-Robinson cone: (a, b) iff time_a < time_b and the
    interval-distance is ≤ vLr · (time_b - time_a).
cs : Poset
    Causet order. On the regular chain (Phase 5), this is just the
    time-only order: (a, b) iff t_idx_a < t_idx_b.
)doc")
        .def_readonly("labels", &CausalOrders::labels)
        .def_readonly("maj",    &CausalOrders::maj)
        .def_readonly("lr",     &CausalOrders::lr)
        .def_readonly("cs",     &CausalOrders::cs);

    py::class_<OrderAgreement>(m, "OrderAgreement",
            R"doc(Pairwise agreement statistics between two posets.

Counted over unordered pairs (i, j) with i < j:

* "comparable in P" — transitive closure of P relates i to j.
* "concordant" — both posets relate the pair, in the same direction.
* "discordant" — both posets relate the pair, in opposite directions.
* "only_a" / "only_b" — one poset relates the pair, the other does not.

The five categories (concordant, discordant, only_a, only_b, neither)
partition the C(nLabels, 2) unordered pairs.

The strong-falsification probe (quantum-methodology.md §1.2 #1) reads
off ``nOnlyA`` when ``(a, b) = (≼_maj, ≼_LR)``: it's the count of
majorization-related pairs whose endpoints lie outside the Lieb–
Robinson cone.

Attributes
----------
kendallTau : float
    (nConcordant - nDiscordant) / nComparableBoth, in [-1, 1].
    1 = perfect agreement; -1 = perfect disagreement.
discordantFraction : float
    nDiscordant / nComparableBoth, in [0, 1].
hasseEditDistance : float
    |E_a △ E_b| / |E_a ∪ E_b| — symmetric difference of cover edges
    normalized by their union size, in [0, 1].
nConcordant, nDiscordant, nComparableBoth : int
nOnlyA, nOnlyB : int
    Pairs related by exactly one of the two posets.
)doc")
        .def_readonly("kendallTau",         &OrderAgreement::kendallTau)
        .def_readonly("discordantFraction", &OrderAgreement::discordantFraction)
        .def_readonly("hasseEditDistance", &OrderAgreement::hasseEditDistance)
        .def_readonly("nConcordant",        &OrderAgreement::nConcordant)
        .def_readonly("nDiscordant",        &OrderAgreement::nDiscordant)
        .def_readonly("nComparableBoth",   &OrderAgreement::nComparableBoth)
        .def_readonly("nOnlyA",            &OrderAgreement::nOnlyA)
        .def_readonly("nOnlyB",            &OrderAgreement::nOnlyB);

    py::class_<CausalComparisonReport>(m, "CausalComparisonReport",
            R"doc(Pairwise agreement statistics across all three Phase 5 orders.

Attributes
----------
majVsLr, majVsCs, lrVsCs : OrderAgreement
    Pairwise comparisons of the three orders.
nLabels : int
    Total number of (cut, time) labels.
nSnapshots : int
    Number of TDVP snapshots used.
vLr : float
    Lieb-Robinson velocity used to build ≼_LR.
)doc")
        .def_readonly("majVsLr",   &CausalComparisonReport::majVsLr)
        .def_readonly("majVsCs",   &CausalComparisonReport::majVsCs)
        .def_readonly("lrVsCs",    &CausalComparisonReport::lrVsCs)
        .def_readonly("nLabels",    &CausalComparisonReport::nLabels)
        .def_readonly("nSnapshots", &CausalComparisonReport::nSnapshots)
        .def_readonly("vLr",        &CausalComparisonReport::vLr);

    m.def("compareOrders",
          &compareOrders,
          py::arg("a"), py::arg("b"), py::arg("nLabels"),
          R"doc(Pairwise agreement statistics between two Posets on the same
label set of size nLabels. Returns an :class:`OrderAgreement`.
)doc");

    m.def("computeCausalComparison",
          &computeCausalComparison,
          py::arg("config"), py::arg("vLr") = 1.0,
          R"doc(End-to-end Phase 5 pipeline: TDVP + causal-order comparison.

Runs ``runQqbarQuench`` (forcing ``recordSpectra = True``), then
builds the three partial orders on the (cut, time) label set:

* ≼_maj from Phase 3 majorization on Schmidt spectra (across cuts AND
  times).
* ≼_LR — Lieb-Robinson cone: a ≺ b iff time_a < time_b and
  interval distance ≤ vLr · (time_b - time_a).
* ≼_cs — causet order; on the regular chain (Phase 5 scope) this is
  the time-only order. Phase 6 replaces the lattice with a non-trivial
  causet, at which point ≼_cs gains within-time-slice structure.

Pairwise agreement is reported as Kendall-τ, the discordant-pair
fraction, and the Hasse-graph edit distance.

Parameters
----------
config : TDVPConfig
    Same as :func:`runQqbarQuench`. ``recordSpectra`` will be forced
    to ``True``.
vLr : float, optional
    Lieb-Robinson velocity in lattice units (sites / time). Default 1.0
    matches the free-fermion group velocity for our hopping coefficient.

Returns
-------
CausalComparisonReport
)doc");

    m.def("runQqbarQuench",
          &runQqbarQuench,
          py::arg("config"),
          R"doc(Run the Phase 4 q-qbar quench + TDVP pipeline.

Steps:

1. Build the Schwinger MPO and run DMRG to the ground state (Phase 2).
2. Apply the q-qbar quench σ⁻_{i0} σ⁺_{i0+d} (Phase 4 / Buyens 2014
   string state) to flip two spins on opposite sublattices.
3. Record the post-quench observables at t = 0.
4. Step TDVP forward by ``config.dt`` for ``round(config.T / config.dt)``
   steps. Record observables every ``config.snapshotEvery`` steps.
5. Return a :class:`QuenchResult` containing the GS diagnostics and
   the snapshot list.

Parameters
----------
config : TDVPConfig
    Hamiltonian + quench + TDVP settings. ``config.N`` ≥ 2,
    ``config.a > 0``, ``config.dt > 0``, ``config.T > 0``. With
    ``config.quenchEnforceParity = True`` (default) ``config.i0``
    must be odd and ``config.d`` must be odd as well.

Returns
-------
QuenchResult
    Ground-state diagnostics + snapshot list.

Examples
--------
>>> from tessera.quantum import TDVPConfig, runQqbarQuench
>>> cfg = TDVPConfig()
>>> cfg.N = 14; cfg.m = 20.0; cfg.g = 1.0          # heavy-quark limit
>>> cfg.i0 = 5; cfg.d = 5                           # odd-odd parity
>>> cfg.dt = 0.05; cfg.T = 5.0; cfg.snapshotEvery = 5
>>> result = runQqbarQuench(cfg)                  # doctest: +SKIP
>>> result.snapshots[0].lProfile[:3]               # doctest: +SKIP
[-1.0, -0.0, -1.0]
)doc");

    m.def("computeGroundStateMajorization",
          &computeGroundStateMajorization,
          py::arg("config"), py::arg("tol") = 1e-12,
          R"doc(Run DMRG ground-state, then extract Schmidt spectra and poset.

Single-shot pipeline that performs all three Phase 3 steps in one C++
call:

1. Build the Schwinger MPO from ``config`` and run DMRG (same as
   :func:`computeGroundState`).
2. Compute the Schmidt spectrum of every contiguous bipartition of the
   optimized MPS, excluding the trivial full-chain cut.
3. Build the majorization poset on those spectra (Hasse cover edges
   only).

Parameters
----------
config : QuantumConfig
    Hamiltonian + DMRG parameters; see :func:`computeGroundState`.
tol : float, optional
    Slack for the majorization comparisons used to build the poset.

Returns
-------
GroundStateMajorizationResult
    Ground-state diagnostics, Schmidt spectra, and the Hasse poset.

Examples
--------
>>> from tessera.quantum import QuantumConfig, computeGroundStateMajorization
>>> cfg = QuantumConfig()
>>> cfg.N = 6; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.0; cfg.L0 = 0.0
>>> cfg.maxBondDim = 32; cfg.nSweeps = 8
>>> r = computeGroundStateMajorization(cfg)
>>> r.spectra.N
6
>>> all(abs(sum(s) - 1.0) < 1e-10 for s in r.spectra.spectra)
True
)doc");

    py::class_<CausetChain>(m, "CausetChain",
            R"doc(Phase 6 — Spacetime-derived chain layout for the Schwinger MPO.

A flattened mapping from a :class:`tessera._tessera.Spacetime` (or a future
:class:`tessera.Causet`) to a 1D lattice with hopping pairs, plus the
inherited Hasse-cover :class:`Poset` on the lattice sites.

For the simplest case where every time slice has a single vertex, this
collapses to a regular chain and ``hoppingPairs == [(0,1), (1,2), …]``.
For non-trivial antichains the chain can still hold the state — but
multi-site antichains may produce hopping pairs with stride > 1, which
is the trigger for moving to a tree tensor network in the longer-term
plan (PLAN.md §6).

Attributes
----------
nSites : int
    Total number of lattice sites = sum over slices of antichain size.
times : list[int]
    Sorted ascending list of integer time slices present in the
    Spacetime.
antichains : list[list[int]]
    ``antichains[s]`` is the ascending-ID list of Spacetime vertex IDs
    at ``times[s]``.
vertexIds : list[int]
    Flat lattice site → Spacetime vertex ID. Concatenation of all
    antichains in time order.
hoppingPairs : list[tuple[int, int]]
    Pairs ``(i, j)`` with ``i < j`` of flat lattice sites coupled by
    timelike causet edges that cross exactly one slice boundary.
partialOrder : Poset
    Hasse cover Poset on the nSites label set, inherited via
    :func:`tessera._tessera.Poset.fromSpacetime`.
)doc")
        .def_readonly("nSites",      &CausetChain::nSites)
        .def_readonly("times",        &CausetChain::times)
        .def_readonly("antichains",   &CausetChain::antichains)
        .def_readonly("vertexIds",   &CausetChain::vertexIds)
        .def_readonly("hoppingPairs",&CausetChain::hoppingPairs)
        .def_readonly("partialOrder",&CausetChain::partialOrder)
        .def("__repr__", [](CausetChain const& c) {
            return "CausetChain(nSites=" + std::to_string(c.nSites) +
                   ", times=" + std::to_string(c.times.size()) +
                   ", hops=" + std::to_string(c.hoppingPairs.size()) + ")";
        });

    // Spacetime is bound in the top-level _tessera module (src/bindings.cpp),
    // not here, so pybind11 can't statically deduce its descriptor for a
    // raw Spacetime const& parameter. Take it through py::object and cast
    // at runtime — the registry lookup resolves to the same class object.
    m.def("extractCausetChain", [](py::object spacetime_obj) {
              auto const* st = spacetime_obj.cast<tessera::Spacetime const*>();
              return extractCausetChain(*st);
          }, py::arg("spacetime"),
          R"doc(Phase 6 — extract a chain-of-antichains adapter from a Spacetime.

Walks the Spacetime's vertex list, groups vertices by integer time slice
(``Vertex.getTime()`` truncated to int), and produces a
:class:`CausetChain` describing:

* the antichain layering (one antichain per time slice, ordered by
  ascending Spacetime vertex ID),
* the flat lattice ↔ Spacetime ID mapping,
* the adjacent-slice timelike-edge hopping pairs (with squaredLength
  < 0 and time-slice index difference exactly 1; non-cover edges
  spanning multiple slices are skipped),
* the Hasse cover :class:`Poset` on the flat lattice sites.

This is the data shape the Phase 6 causet-embedded Schwinger MPO
construction would consume to replace the regular 1D lattice. For the
trivial case where every time slice has a single vertex,
``hoppingPairs`` reduces to ``[(0, 1), (1, 2), …]`` and the existing
:func:`computeGroundState` runs unchanged on top.

Parameters
----------
spacetime : tessera._tessera.Spacetime
    Source spacetime. Vertices must have at least 1D coordinates so
    that ``Vertex.getTime()`` is well-defined.

Returns
-------
CausetChain
)doc");

    m.def("computeGroundState", &computeGroundState,
          py::arg("config"),
          R"doc(Run DMRG to the Schwinger-model ground state.

Builds the MPO Hamiltonian from ``config`` (using the dimensional H
described in the module docstring), initialises a Néel |↑↓↑↓…⟩ MPS
in the Sz = 0 sector, and runs ITensor's two-site DMRG with the
sweep schedule encoded in ``config``. Returns a
:class:`GroundStateResult` with the converged energy plus structural
diagnostics.

The DMRG schedule ramps the bond-dim cap (20 → 40 → 80 → max) over
the first sweeps so early iterations don't commit truncation errors
that later sweeps have to undo, and injects a small noise term
(1e-7 → 1e-8 → 0) in the first two sweeps to escape local minima.

Parameters
----------
config : QuantumConfig
    Hamiltonian parameters and DMRG sweep settings. ``config.N`` must
    be ≥ 2; ``config.a`` must be positive. Other fields have working
    defaults.

Returns
-------
GroundStateResult
    Energy, bond-dim, and truncation diagnostics.

Raises
------
RuntimeError or ValueError
    If ``config.N < 2`` or ``config.a <= 0`` (validation forwarded
    from the underlying SchwingerMPO builder).

Examples
--------
Reproduce the Phase 1 N=4, m/g=0 reference value::

    >>> from tessera.quantum import QuantumConfig, computeGroundState
    >>> cfg = QuantumConfig()
    >>> cfg.N = 4
    >>> cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.0; cfg.L0 = 0.0
    >>> cfg.maxBondDim = 32; cfg.nSweeps = 8
    >>> r = computeGroundState(cfg)
    >>> abs(r.operatorEnergy - (-1.738676174)) < 1e-8
    True

Scan the bond-dim cap to see variational descent::

    >>> energies = []
    >>> for D in (4, 8, 16, 32):
    ...     cfg.maxBondDim = D
    ...     energies.append(computeGroundState(cfg).operatorEnergy)
    >>> all(energies[i] >= energies[i+1] - 1e-10 for i in range(3))
    True

See Also
--------
QuantumConfig : Input configuration struct.
GroundStateResult : Output result struct.

Notes
-----
For N ≳ 30 expect the run to take seconds; for N ≳ 100 with bondDim
≳ 100, tens of seconds. The CPU work is in BLAS/LAPACK calls inside
ITensor — set OMP_NUM_THREADS to use multiple cores.
)doc");
}
