// Pybind11 bindings for the quantum subsystem. Lives outside libcaset_quantum
// (which is pybind-free) so the static library can be reused without pulling
// in the Python dependency. This translation unit is added to _caset's
// sources only when CASET_QUANTUM=ON in CMakeLists.txt.
//
// Surface area is deliberately minimal per PLAN.md §1: scalars in, scalars
// out. No MPS / MPO / ITensor types cross the Python boundary. If callers
// need the wave function itself they should add a C++-side observable
// callback that returns plain data (Phase 4's approach for TDVP snapshots).
//
// Docstrings here are deliberately verbose because they're the primary
// user-facing documentation — `help(caset.quantum.compute_ground_state)`
// in a Python REPL or notebook should be enough to understand both the
// API and the underlying physics conventions.

#include "quantum/dmrg_runner.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_quantum_bindings(py::module_ m) {
    using namespace caset::quantum;

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
N = 0 and must be filled in before passing to :func:`compute_ground_state`.

The dt and T fields are reserved for Phase 4 (real-time evolution
after a quark/antiquark quench) and are ignored by
:func:`compute_ground_state`.

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
max_bond_dim : int
    Cap on the MPS bond dimension during DMRG sweeps. Schwinger ground
    states at moderate (m, g, N) typically converge at bond dim ~10-30
    even for N = 100+. Bond-dim-limited runs report ``bond_dim`` equal
    to this value in the result.
n_sweeps : int
    Total number of DMRG sweeps. Energy convergence is exponential in
    sweep count once bond-dim is sufficient; 12 sweeps is plenty for
    moderate-N tests.
cutoff : float
    SVD truncation threshold per local solve. 1e-12 is conservative for
    these Hamiltonians and well below double-precision noise. Tighter
    cutoff costs sweep time without meaningful precision gain.
krylov_dim : int
    Lanczos / Krylov dimension per local 2-site solve. Default 4 is
    fine for gapped problems; raise for harder cases (small mass,
    near-critical points) where eigensolver convergence is slower.
quiet : bool
    If True (default), suppress ITensor's per-sweep diagnostic prints.
    Set False to debug DMRG convergence.
conserve_qns : bool
    If True (default), the SiteSet enforces total-Sz QN conservation
    on every MPS bond. The Néel initial state then pins DMRG to the
    Sz = 0 sector. Set False only when you need to apply σ^x / σ^y
    later (these change Sz) — currently exercised by Phase 1's
    charge-conjugation test.
dt : float
    *(Phase 4)* TDVP real-time step size. Unused by
    compute_ground_state.
T : float
    *(Phase 4)* Total evolution time. Unused by compute_ground_state.

Examples
--------
>>> from caset.quantum import QuantumConfig
>>> cfg = QuantumConfig()
>>> cfg.N = 20
>>> cfg.a = 1.0
>>> cfg.g = 1.0
>>> cfg.m = 0.0
>>> cfg.L0 = 0.0
>>> cfg.max_bond_dim = 100
>>> cfg.n_sweeps = 12
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
        .def_readwrite("max_bond_dim", &QuantumConfig::max_bond_dim,
            "Cap on MPS bond dimension during DMRG.")
        .def_readwrite("n_sweeps",     &QuantumConfig::n_sweeps,
            "Total DMRG sweep count.")
        .def_readwrite("cutoff",       &QuantumConfig::cutoff,
            "SVD truncation threshold per local solve.")
        .def_readwrite("krylov_dim",   &QuantumConfig::krylov_dim,
            "Lanczos / Krylov dimension per local solve.")
        .def_readwrite("quiet",        &QuantumConfig::quiet,
            "Suppress ITensor's per-sweep diagnostic output.")
        .def_readwrite("conserve_qns", &QuantumConfig::conserve_qns,
            "U(1) total-Sz conservation on the SiteSet.")
        .def_readwrite("dt",           &QuantumConfig::dt,
            "(Phase 4) Real-time step size; unused by compute_ground_state.")
        .def_readwrite("T",            &QuantumConfig::T,
            "(Phase 4) Total evolution time; unused by compute_ground_state.")
        .def("__repr__", [](QuantumConfig const& c) {
            return "QuantumConfig(N=" + std::to_string(c.N) +
                   ", a=" + std::to_string(c.a) +
                   ", m=" + std::to_string(c.m) +
                   ", g=" + std::to_string(c.g) +
                   ", L0=" + std::to_string(c.L0) +
                   ", max_bond_dim=" + std::to_string(c.max_bond_dim) +
                   ", n_sweeps=" + std::to_string(c.n_sweeps) + ")";
        });

    py::class_<GroundStateResult>(m, "GroundStateResult",
            R"doc(Result of a Schwinger-model DMRG ground-state run.

The full physical energy is ``energy``. The other fields let callers
sanity-check convergence:

* ``operator_energy + constant`` should equal ``energy`` exactly
  (assertable to ~1e-12).
* ``bond_dim`` hitting ``config.max_bond_dim`` flags a bond-dim-limited
  run — increase the cap and rerun if you need tighter convergence.
* ``truncation_err`` is a conservative upper bound (the configured
  cutoff). Real truncation is typically much smaller.

Attributes
----------
energy : float
    Full physical energy ⟨H⟩ + constant. Use this when comparing to
    Bañuls-style published numerics (after the dimensional rescaling
    E_W = (2/(ag²)) E_dim if you want their dimensionless W).
operator_energy : float
    ⟨H⟩ alone, as returned by ITensor's dmrg(). Use this for direct
    cross-checks against dense diagonalization of the operator-valued
    Hamiltonian (without the c-number L_n² shift).
constant : float
    The c-number shift (g²a/2) Σ_n (c_n² + n/4) pulled out of L_n²
    when AutoMPO-encoding the electric term — see the derivation in
    src/quantum/schwinger_model.cpp. ``operator_energy + constant ==
    energy`` by construction.
bond_dim : int
    Largest bond dimension of the optimized MPS. Equals
    ``config.max_bond_dim`` if the run was bond-dim-limited.
truncation_err : float
    Conservative upper bound on the SVD truncation error in the final
    sweep (currently equal to the configured ``cutoff``).
)doc")
        .def_readonly("energy",          &GroundStateResult::energy,
            "Full physical energy ⟨H⟩ + constant.")
        .def_readonly("operator_energy", &GroundStateResult::operator_energy,
            "Operator part ⟨H⟩, matches ITensor's dmrg() return value.")
        .def_readonly("constant",        &GroundStateResult::constant,
            "C-number shift from L_n² expansion.")
        .def_readonly("bond_dim",        &GroundStateResult::bond_dim,
            "Largest MPS bond dimension after the final sweep.")
        .def_readonly("truncation_err",  &GroundStateResult::truncation_err,
            "Conservative upper bound on the truncation error.")
        .def("__repr__", [](GroundStateResult const& r) {
            return "GroundStateResult(energy=" + std::to_string(r.energy) +
                   ", bond_dim=" + std::to_string(r.bond_dim) +
                   ", truncation_err=" + std::to_string(r.truncation_err) + ")";
        });

    m.def("compute_ground_state", &compute_ground_state,
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

    >>> from caset.quantum import QuantumConfig, compute_ground_state
    >>> cfg = QuantumConfig()
    >>> cfg.N = 4
    >>> cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.0; cfg.L0 = 0.0
    >>> cfg.max_bond_dim = 32; cfg.n_sweeps = 8
    >>> r = compute_ground_state(cfg)
    >>> abs(r.operator_energy - (-1.738676174)) < 1e-8
    True

Scan the bond-dim cap to see variational descent::

    >>> energies = []
    >>> for D in (4, 8, 16, 32):
    ...     cfg.max_bond_dim = D
    ...     energies.append(compute_ground_state(cfg).operator_energy)
    >>> all(energies[i] >= energies[i+1] - 1e-10 for i in range(3))
    True

See Also
--------
QuantumConfig : Input configuration struct.
GroundStateResult : Output result struct.

Notes
-----
For N ≳ 30 expect the run to take seconds; for N ≳ 100 with bond_dim
≳ 100, tens of seconds. The CPU work is in BLAS/LAPACK calls inside
ITensor — set OMP_NUM_THREADS to use multiple cores.
)doc");
}
