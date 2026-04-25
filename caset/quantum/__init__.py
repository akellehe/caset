"""caset.quantum -- Schwinger model on a Jordan-Wigner spin chain via DMRG.

This subpackage implements Phases 2-5 of ``docs/source/quantum-plan.md``:
ground-state DMRG for the 1+1D Kogut-Susskind Schwinger model with U(1)
total-charge conservation, contiguous-cut Schmidt spectra and the
majorization poset on those spectra, a q-qbar quench + 2-site TDVP
pipeline that produces real-time-evolution snapshots, and an end-to-end
causal-order comparison between the majorization order, the Lieb-
Robinson cone, and the (regular-chain) causet order — the experimental
harness for the methodology page (``docs/source/quantum-methodology.md``).
The C++ backend is ITensor v3 vendored under ``third_party/itensor``
(plus the ITensor TDVP add-on under ``third_party/itensor_tdvp``);
this Python layer is a thin result viewer per the architectural
principle in PLAN.md §1 ("minimize Python/C++ crossings"). No MPS or
MPO objects cross the language barrier — only scalar configs in and
scalar / list diagnostics out.

Availability
------------
This module requires caset to be built with ``CASET_QUANTUM=1``::

    CASET_QUANTUM=1 pip install -e .

Without that flag, importing this module raises :class:`ImportError`
with the rebuild instruction. The default (CASET_QUANTUM=0) build of
caset is unaffected.

Hamiltonian (PLAN.md §4 / Bañuls et al. 2013 eq. 2.6)
-----------------------------------------------------

After Jordan-Wigner mapping and Gauss's-law elimination, the staggered
Schwinger Hamiltonian on N sites with open boundary conditions is::

    H = H_hop + H_m + H_E

    H_hop = (1/(4a)) Σ_{n=1..N-1} (X_n X_{n+1} + Y_n Y_{n+1})
    H_m   = (m/2)    Σ_{n=1..N}   (-1)^n σ^z_n
    H_E   = (g²a/2)  Σ_{n=1..N-1} L_n²

    L_n   = L0 + Σ_{k=1..n} [(1 - σ^z_k)/2 - (1 - (-1)^k)/2]

with parameters:

* ``a`` — lattice spacing (sets the units of E)
* ``m`` — bare staggered-fermion mass
* ``g`` — gauge coupling (g = 0 is the free-Dirac limit)
* ``L0`` — background electric field on the link to the left of site 1
* ``N`` — number of sites (1-based; even N for charge-neutral GS)

Bañuls' dimensionless parameters are::

    x = 1/(g² a²)
    μ = 2m / (g² a)

The continuum limit corresponds to x → ∞ at fixed m/g; finite-volume
effects are controlled by N ≳ 20·√x (Bañuls' prescription).

Quickstart
----------

Compute the ground state at the Phase 1 / PLAN.md spec parameters::

    >>> from caset.quantum import QuantumConfig, compute_ground_state
    >>> cfg = QuantumConfig()
    >>> cfg.N = 20             # 1-based, even
    >>> cfg.a = 1.0; cfg.g = 1.0
    >>> cfg.m = 0.0            # massless
    >>> cfg.L0 = 0.0           # zero background field
    >>> cfg.max_bond_dim = 100
    >>> cfg.n_sweeps = 12
    >>> result = compute_ground_state(cfg)
    >>> result.energy < 0
    True

The returned object exposes diagnostics::

    >>> print(result)              # doctest: +SKIP
    GroundStateResult(energy=-4.31..., bond_dim=..., truncation_err=...)
    >>> # operator_energy + constant == energy by construction
    >>> abs(result.energy - (result.operator_energy + result.constant)) < 1e-12
    True

Convergence checks
------------------

The result's ``bond_dim`` field is the achieved MPS bond dimension. If
it equals ``config.max_bond_dim``, the run was bond-dim-limited and
the energy may not be fully converged. Bumping the cap and rerunning
should give a (variationally) lower energy::

    >>> cfg.max_bond_dim = 50; e1 = compute_ground_state(cfg).energy
    >>> cfg.max_bond_dim = 100; e2 = compute_ground_state(cfg).energy
    >>> e2 <= e1 + 1e-10  # variational: more bond-dim → lower energy
    True

Phase 3 — Schmidt spectra and majorization poset
------------------------------------------------

For each contiguous interval A = [i, j] on the spin chain, the Schmidt
spectrum λ_A is the list of eigenvalues of ρ_A = Tr_{Ā}|ψ⟩⟨ψ|, sorted
non-increasingly. Majorization (μ ≼ λ iff "λ is at least as
concentrated as μ") defines a partial order on the cuts; the Hasse
diagram is the transitive reduction of the strict-majorization graph.
This is the hypothesis substrate for the methodology page
(``docs/source/quantum-methodology.md``).

The pure-function API works on plain Python lists::

    >>> from caset.quantum import majorizes, strictly_majorizes, majorization_poset
    >>> majorizes([1.0, 0.0], [0.5, 0.5])     # (1, 0) ≻ (½, ½)
    True
    >>> strictly_majorizes([0.5, 0.5], [1.0, 0.0])
    False
    >>> p = majorization_poset([[1/3]*3, [0.5, 0.5], [1.0]])
    >>> p.n_nodes, sorted(p.covers)
    (3, [(1, 0), (2, 1)])

The end-to-end pipeline (DMRG → Schmidt → poset) is one call::

    >>> from caset.quantum import compute_ground_state_majorization
    >>> r = compute_ground_state_majorization(cfg)  # cfg from above
    >>> r.spectra.N
    20
    >>> all(abs(sum(s) - 1.0) < 1e-10 for s in r.spectra.spectra)
    True

Phase 3 cut family is contiguous intervals 1 ≤ i ≤ j ≤ N excluding the
trivial full-chain bipartition [1, N] | ∅; this is N(N+1)/2 - 1 cuts
total.

Phase 4 — q-qbar quench and TDVP real-time evolution
----------------------------------------------------

The `run_qqbar_quench(config)` function runs the full DMRG → quench →
TDVP pipeline in a single C++ call. The quench operator is

    U_qqbar(i0, d)  =  σ⁻_{i0} · σ⁺_{i0 + d}

which on the heavy-quark Néel vacuum ``|↑↓↑↓ … ⟩`` creates a +1 flux tube
on the d links between sites i0 and i0+d (Buyens 2014 string state).
The parity constraint i0 odd + d odd applies (PLAN.md §5 Phase 4 /
quench.hpp). Each TDVP step records ⟨L_n⟩(t), ⟨σ^z_n⟩(t), the energy,
and the bond dimension; optionally the full contiguous-cut Schmidt
spectra and majorization poset (off by default — they cost O(N²) SVDs
per snapshot). Example::

    >>> from caset.quantum import TDVPConfig, run_qqbar_quench
    >>> cfg = TDVPConfig()
    >>> cfg.N = 14; cfg.m = 20.0; cfg.g = 1.0          # heavy-quark
    >>> cfg.i0 = 5; cfg.d = 5                           # odd-odd parity
    >>> cfg.dt = 0.05; cfg.T = 5.0; cfg.snapshot_every = 5
    >>> r = run_qqbar_quench(cfg)
    >>> r.snapshots[0].L_profile[:3]                   # +1 tube starts at link 3
    [-1.0, -0.0, -0.0]

In the heavy-quark limit the flux tube is approximately stable for the
duration of the run (PLAN.md §5 Phase 4 acceptance: ⟨L_n⟩(t) matches
the reference to within 0.05 at t = T/2; |ΔE|/|E0| < 1e-3).

Phase 5 — causal-order comparison
---------------------------------

The `compute_causal_comparison(config, v_LR)` function ties Phases 1-4
together: DMRG ground state → q-qbar quench → TDVP loop → build three
partial orders on (cut, time) labels → compare. The orders are:

  ≼_maj: strict-majorization on Schmidt spectra (Phase 3, across time)
  ≼_LR:  Lieb-Robinson cone, dist(A, B) ≤ v_LR · (t_B - t_A)
  ≼_cs:  causet — time-only on regular chain (Phase 6 makes it richer)

Each comparison reports Kendall-τ, discordant-pair fraction, and Hasse-
graph edit distance. Example::

    >>> from caset.quantum import TDVPConfig, compute_causal_comparison
    >>> cfg = TDVPConfig()
    >>> cfg.N = 10; cfg.m = 0.5; cfg.g = 1.0
    >>> cfg.i0 = 3; cfg.d = 3
    >>> cfg.dt = 0.1; cfg.T = 1.0; cfg.snapshot_every = 1
    >>> r = compute_causal_comparison(cfg, v_LR=1.0)
    >>> r.lr_vs_cs.kendall_tau
    1.0

The ≼_LR ⊂ ≼_cs invariant gives the strongest sanity check: τ = 1.0
exactly because every ≼_LR pair is also a ≼_cs pair in the same
direction (LR adds a spatial constraint to time-only).

Tested benchmarks
-----------------

The C++ test suite (``tests/quantum/test_schwinger_*.cpp``) cross-checks
this implementation against:

* Dense Eigen ED on the full 2^N basis for N ∈ {4, 6, 8} ×
  m/g ∈ {0, 0.125, 0.25} × L0 ∈ {0, 0.5} — agreement to 1e-12.
* Free-fermion analytic limit (g = m = 0): half-filled OBC chain GS
  energy Σ_j (1/a) cos(πj/(N+1)).
* Strong-coupling vacuum (m → ∞, g = a = 1, L0 = 0): asymptotic
  -mN/2 + g²aN/4.
* Continuum trend ω₀ → -1/π at fixed (x, N) scaling (Bañuls Fig. 6).
* Vector mass gap M_V/g ≈ 1/√π in the continuum (Bañuls table).
* Chiral condensate ⟨Σ̄Σ⟩: nonzero at m=0 (anomaly), saturated at
  large m.
* Charge-conjugation parity ⟨S_R⟩ = ⟨σ^x_odd · T⁽¹⁾⟩ (Bañuls page 8).
* Schmidt spectra of the Schwinger ground state at small N agree with
  dense Eigen ED to 1e-9 across 8 (N, m, L0) parameter combos.
* Majorization poset acceptance: product → 0 Hasse edges; GHZ → 0
  strict edges; Bell vs. product → (1, 0) ≻ (½, ½) cover edge present.

Phase 2-3's pytest layer (``tests/quantum/test_phase{2,3}_*.py``)
reproduces the small-N reference values and Hasse-poset structure
through this Python API.

References
----------

* Bañuls, Cichy, Cirac, Jansen, *JHEP* **11**, 158 (2013), arXiv:1305.3765
  -- primary reference for the Hamiltonian convention and benchmarks.
* Schwinger, *Phys. Rev.* **128**, 2425 (1962) -- original gauge theory.
* Coleman, *Ann. Phys.* **101**, 239 (1976) -- massive Schwinger model.
* Kogut, Susskind, *Phys. Rev. D* **11**, 395 (1975) -- staggered
  fermion lattice formulation.
* ITensor library: https://itensor.org -- the MPS/MPO backend.

API
---

The full API is the contents of ``__all__`` below — see the per-symbol
docstrings for parameter / return-value documentation, and
``docs/source/quantum.md`` for narrative usage. The Python signatures
are pybind11-generated from the C++ types in ``include/quantum/``.

Phase 2 — DMRG ground state:
    QuantumConfig, GroundStateResult, compute_ground_state

Phase 3 — Schmidt spectra and majorization poset:
    Interval, SchmidtSpectra, Poset, GroundStateMajorizationResult,
    majorizes, strictly_majorizes, majorization_poset,
    compute_ground_state_majorization

Phase 4 — TDVP q-qbar quench:
    TDVPConfig, TDVPSnapshot, QuenchResult, run_qqbar_quench

Phase 5 — causal-order comparison:
    LabelSpacetime, CausalOrders, OrderAgreement, CausalComparisonReport,
    compare_orders, compute_causal_comparison

Implementation notes
--------------------

* The DMRG runs in the U(1) total-Sz = 0 sector by default
  (``config.conserve_qns = True``). Set ``conserve_qns = False`` only
  when you need to apply non-Sz-conserving operators (σ^x, σ^y) to
  the resulting MPS.
* The MPO is built via ITensor's ``AutoMPO`` from the operator
  expansion of L_n² (linear σ^z plus pair σ^z σ^z plus a c-number
  shift). The c-number shift is returned as ``result.constant`` so the
  full physical energy is ``result.energy = operator_energy + constant``.
* Bond dimension for the Schwinger MPO grows linearly with N (typical
  values: 5 for N ≤ 30, ~10 for N ≤ 100). The MPS bond dimension
  needed for ground-state convergence is also modest at small mass.

"""

try:
    # caset._caset is a single C extension (.so), not a Python package, so
    # the submodule is exposed as an attribute rather than a separate
    # importable module. Pybind11's def_submodule installs it on the parent
    # module's __dict__ at import time; we just look it up.
    from caset import _caset
    _qm = _caset.quantum

    # Phase 2 — DMRG ground state.
    QuantumConfig         = _qm.QuantumConfig
    GroundStateResult     = _qm.GroundStateResult
    compute_ground_state  = _qm.compute_ground_state

    # Phase 3 — Schmidt spectra + majorization poset.
    Interval                            = _qm.Interval
    SchmidtSpectra                      = _qm.SchmidtSpectra
    Poset                               = _qm.Poset
    GroundStateMajorizationResult       = _qm.GroundStateMajorizationResult
    majorizes                           = _qm.majorizes
    strictly_majorizes                  = _qm.strictly_majorizes
    majorization_poset                  = _qm.majorization_poset
    compute_ground_state_majorization   = _qm.compute_ground_state_majorization

    # Phase 4 — TDVP real-time evolution after a q-qbar quench.
    TDVPConfig         = _qm.TDVPConfig
    TDVPSnapshot       = _qm.TDVPSnapshot
    QuenchResult       = _qm.QuenchResult
    run_qqbar_quench   = _qm.run_qqbar_quench

    # Phase 5 — causal-order comparison (maj vs LR vs caset).
    LabelSpacetime              = _qm.LabelSpacetime
    CausalOrders                = _qm.CausalOrders
    OrderAgreement              = _qm.OrderAgreement
    CausalComparisonReport      = _qm.CausalComparisonReport
    compare_orders              = _qm.compare_orders
    compute_causal_comparison   = _qm.compute_causal_comparison

    # Phase 6 — caset-Spacetime → chain-of-antichains adapter.
    CausetChain           = _qm.CausetChain
    extract_causet_chain  = _qm.extract_causet_chain
except (ImportError, AttributeError) as exc:
    raise ImportError(
        "caset.quantum is unavailable: this caset build does not include "
        "the quantum subsystem. Rebuild with CASET_QUANTUM=1 (e.g. "
        "`CASET_QUANTUM=1 pip install -e .`) to enable it. "
        "See docs/source/quantum-plan.md for the broader plan."
    ) from exc

__all__ = [
    # Phase 2
    "QuantumConfig",
    "GroundStateResult",
    "compute_ground_state",
    # Phase 3
    "Interval",
    "SchmidtSpectra",
    "Poset",
    "GroundStateMajorizationResult",
    "majorizes",
    "strictly_majorizes",
    "majorization_poset",
    "compute_ground_state_majorization",
    # Phase 4
    "TDVPConfig",
    "TDVPSnapshot",
    "QuenchResult",
    "run_qqbar_quench",
    # Phase 5
    "LabelSpacetime",
    "CausalOrders",
    "OrderAgreement",
    "CausalComparisonReport",
    "compare_orders",
    "compute_causal_comparison",
    # Phase 6
    "CausetChain",
    "extract_causet_chain",
]
