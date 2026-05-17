"""tessera.quantum -- Schwinger model on a Jordan-Wigner spin chain via DMRG.

Ground-state DMRG for the 1+1D Kogut-Susskind Schwinger model with U(1)
total-charge conservation, contiguous-cut Schmidt spectra and the
majorization poset on those spectra, a q-qbar quench + 2-site TDVP
pipeline that produces real-time-evolution snapshots, and an end-to-end
causal-order comparison between the majorization order, the Lieb-
Robinson cone, and the (regular-chain) causet order — the experimental
harness for the methodology page (``docs/source/quantum-methodology.md``).
The C++ backend is ITensor v3 vendored under ``third_party/itensor``;
this Python layer is a thin result viewer per the architectural
principle in PLAN.md §1 ("minimize Python/C++ crossings"). No MPS or
MPO objects cross the language barrier — only scalar configs in and
scalar / list diagnostics out.

API style
---------

Every Python-visible operation is a method on a coarse-grained class —
there are no free functions in ``tessera.quantum``. The four user-facing
classes are:

* :class:`SchwingerModel`  — DMRG ground-state pipeline.
* :class:`SchwingerQuench` — q-qbar quench + TDVP + causal-order
                              comparison pipeline.
* :class:`Majorization`    — static utility: predicate-driven poset
                              construction and pairwise order-agreement
                              statistics.
* :class:`Causet`          — static utility: tessera.Spacetime → causet
                              adapters.

Plus the data classes (:class:`QuantumConfig`, :class:`GroundStateResult`,
:class:`SchmidtSpectra`, :class:`TDVPConfig`, :class:`TDVPSnapshot`, …)
and the :class:`MajorizationPredicate` hierarchy
(:class:`StandardMajorization`, :class:`LogConcaveMajorization`,
:class:`PeakRadialMajorization`) that callers configure workflows with.

Availability
------------
This module requires tessera to be built with ``TESSERA_QUANTUM=1``::

    TESSERA_QUANTUM=1 pip install -e .

Without that flag, importing this module raises :class:`ImportError`
with the rebuild instruction. The default (TESSERA_QUANTUM=0) build of
tessera is unaffected.

Hamiltonian (PLAN.md §4 / Bañuls et al. 2013 eq. 2.6)
-----------------------------------------------------

After Jordan-Wigner mapping and Gauss's-law elimination, the staggered
Schwinger Hamiltonian on N sites with open boundary conditions is::

    H = H_hop + H_m + H_E

    H_hop = (1/(4a)) Σ_{n=1..N-1} (X_n X_{n+1} + Y_n Y_{n+1})
    H_m   = (m/2)    Σ_{n=1..N}   (-1)^n σ^z_n
    H_E   = (g²a/2)  Σ_{n=1..N-1} L_n²

    L_n   = L0 + Σ_{k=1..n} [(1 - σ^z_k)/2 - (1 - (-1)^k)/2]

Quickstart
----------

Compute the ground state at PLAN.md §4 spec parameters::

    >>> from tessera.quantum import QuantumConfig, SchwingerModel
    >>> cfg = QuantumConfig()
    >>> cfg.N = 20
    >>> cfg.a = 1.0; cfg.g = 1.0
    >>> cfg.m = 0.0
    >>> cfg.L0 = 0.0
    >>> cfg.maxBondDim = 100
    >>> cfg.nSweeps = 12
    >>> result = SchwingerModel(cfg).solve()
    >>> result.energy < 0
    True

Schmidt spectra and majorization poset
--------------------------------------

For each contiguous interval A = [i, j] on the spin chain, the Schmidt
spectrum λ_A is the list of eigenvalues of ρ_A = Tr_{Ā}|ψ⟩⟨ψ|, sorted
non-increasingly. Majorization (μ ≼ λ iff "λ is at least as
concentrated as μ") defines a partial order on the cuts; the Hasse
diagram is the transitive reduction of the strict-majorization graph.

Construct a poset directly from a list of spectra::

    >>> from tessera.quantum import Majorization, StandardMajorization
    >>> p = Majorization.posetOf([[1/3]*3, [0.5, 0.5], [1.0]])
    >>> p.getNodeCount, sorted(p.covers)
    (3, [(1, 0), (2, 1)])

Or run the full DMRG → Schmidt → poset pipeline in one call::

    >>> r = SchwingerModel(cfg).solveWithMajorization()
    >>> r.spectra.N
    20

The cut family is contiguous intervals 1 ≤ i ≤ j ≤ N excluding the
trivial full-chain bipartition [1, N] | ∅; this is N(N+1)/2 - 1 cuts
total.

q-qbar quench and TDVP real-time evolution
------------------------------------------

The :meth:`SchwingerQuench.evolve` method runs the full DMRG → quench →
TDVP pipeline. The quench operator is

    U_qqbar(i0, d)  =  σ⁻_{i0} · σ⁺_{i0 + d}

which on the heavy-quark Néel vacuum ``|↑↓↑↓ … ⟩`` creates a +1 flux tube
on the d links between sites i0 and i0+d (Buyens 2014 string state).
Parity constraint: i0 odd + d odd. Example::

    >>> from tessera.quantum import TDVPConfig, SchwingerQuench
    >>> cfg = TDVPConfig()
    >>> cfg.N = 14; cfg.m = 20.0; cfg.g = 1.0
    >>> cfg.i0 = 5; cfg.d = 5
    >>> cfg.dt = 0.05; cfg.T = 5.0; cfg.snapshotEvery = 5
    >>> r = SchwingerQuench(cfg).evolve()
    >>> r.snapshots[0].lProfile[:3]
    [-1.0, -0.0, -0.0]

Causal-order comparison
-----------------------

:meth:`SchwingerQuench.compareCausalOrders` ties the ground-state,
Schmidt, and TDVP pipelines together:
DMRG ground state → q-qbar quench → TDVP loop → build three partial
orders on (cut, time) labels → compare. The orders are:

  ≼_maj: strict-majorization on Schmidt spectra (across time)
  ≼_LR:  Lieb-Robinson cone, dist(A, B) ≤ vLr · (t_B - t_A)
  ≼_cs:  causet — time-only on regular chain

Each comparison reports Kendall-τ, discordant-pair fraction, and Hasse-
graph edit distance. Example::

    >>> cfg = TDVPConfig()
    >>> cfg.N = 10; cfg.m = 0.5; cfg.g = 1.0
    >>> cfg.i0 = 3; cfg.d = 3
    >>> cfg.dt = 0.1; cfg.T = 1.0; cfg.snapshotEvery = 1
    >>> r = SchwingerQuench(cfg).compareCausalOrders(vLr=1.0)
    >>> r.lrVsCs.kendallTau
    1.0

The ≼_LR ⊂ ≼_cs invariant gives the strongest sanity check: τ = 1.0
exactly because every ≼_LR pair is also a ≼_cs pair in the same
direction.

References
----------

* Bañuls, Cichy, Cirac, Jansen, *JHEP* **11**, 158 (2013), arXiv:1305.3765
  -- primary reference for the Hamiltonian convention and benchmarks.
* Schwinger, *Phys. Rev.* **128**, 2425 (1962) -- original gauge theory.
* Coleman, *Ann. Phys.* **101**, 239 (1976) -- massive Schwinger model.
* Kogut, Susskind, *Phys. Rev. D* **11**, 395 (1975) -- staggered
  fermion lattice formulation.
* ITensor library: https://itensor.org -- the MPS/MPO backend.

"""

try:
    # tessera._tessera is a single C extension (.so), not a Python package, so
    # the submodule is exposed as an attribute rather than a separate
    # importable module. Pybind11's def_submodule installs it on the parent
    # module's __dict__ at import time; we just look it up.
    from tessera import _tessera
    _qm = _tessera.quantum

    # ─── Data classes (configs, results, labels, posets) ───────────────────
    QuantumConfig                 = _qm.QuantumConfig
    GroundStateResult             = _qm.GroundStateResult
    Interval                      = _qm.Interval
    SchmidtSpectra                = _qm.SchmidtSpectra
    Poset                         = _qm.Poset
    GroundStateMajorizationResult = _qm.GroundStateMajorizationResult
    TDVPConfig                    = _qm.TDVPConfig
    TDVPSnapshot                  = _qm.TDVPSnapshot
    QuenchResult                  = _qm.QuenchResult
    InteractionConfig             = _qm.InteractionConfig
    InitialChargeMode             = _qm.InitialChargeMode
    LabelSpacetime                = _qm.LabelSpacetime
    CausalOrders                  = _qm.CausalOrders
    OrderAgreement                = _qm.OrderAgreement
    CausalComparisonReport        = _qm.CausalComparisonReport
    CausetChain                   = _qm.CausetChain

    # ─── MajorizationPredicate hierarchy ───────────────────────────────────
    MajorizationPredicate  = _qm.MajorizationPredicate
    StandardMajorization   = _qm.StandardMajorization
    LogConcaveMajorization = _qm.LogConcaveMajorization
    PeakRadialMajorization = _qm.PeakRadialMajorization

    # ─── Coarse-grained workflow classes ───────────────────────────────────
    SchwingerModel        = _qm.SchwingerModel
    SchwingerQuench       = _qm.SchwingerQuench
    InteractionSimulation = _qm.InteractionSimulation
    Majorization          = _qm.Majorization
    Causet                = _qm.Causet
    MutualInformation     = _qm.MutualInformation

    # ─── Free functions ───────────────────────────────────────────────────
    # Pairwise agreement statistics between two Posets on a shared label
    # set; returns an :class:`OrderAgreement`. See
    # ``docs/source/causal_sets.md`` for the methodology context.
    compareOrders = _qm.compareOrders
except (ImportError, AttributeError) as exc:
    raise ImportError(
        "tessera.quantum is unavailable: this tessera build does not include "
        "the quantum subsystem. Rebuild with TESSERA_QUANTUM=1 (e.g. "
        "`TESSERA_QUANTUM=1 pip install -e .`) to enable it. "
        "See docs/source/quantum-plan.md for the broader plan."
    ) from exc

__all__ = [
    # Data classes
    "QuantumConfig",
    "GroundStateResult",
    "Interval",
    "SchmidtSpectra",
    "Poset",
    "GroundStateMajorizationResult",
    "TDVPConfig",
    "TDVPSnapshot",
    "QuenchResult",
    "InteractionConfig",
    "InitialChargeMode",
    "LabelSpacetime",
    "CausalOrders",
    "OrderAgreement",
    "CausalComparisonReport",
    "CausetChain",
    # Majorization predicate hierarchy
    "MajorizationPredicate",
    "StandardMajorization",
    "LogConcaveMajorization",
    "PeakRadialMajorization",
    # Coarse-grained workflow classes
    "SchwingerModel",
    "SchwingerQuench",
    "InteractionSimulation",
    "Majorization",
    "MutualInformation",
    "Causet",
    # Free functions
    "compareOrders",
]
