# Quantum subsystem (Schwinger MPS / DMRG)

The `tessera.quantum` subpackage extends tessera with a tensor-network
treatment of the 1+1D Kogut-Susskind Schwinger model — the simplest
non-trivial gauge theory and a standard benchmark for lattice tensor-
network methods. It implements the staged plan in
`docs/source/quantum-plan.md`. As of this writing Phases 0-5 are
complete (scaffolding, MPO + DMRG ground states, Python bindings,
Schmidt spectra + majorization poset, q-qbar quench + 2-site TDVP,
causal-order comparison); Phase 6 (tessera-embedded chain) and
Phase 7 (MERA / KAK / EML, research) remain.

This page is the **user-facing** reference: how to build the subsystem,
how to call the Python API, what each diagnostic field means. The two
companion documents are:

* [quantum-methodology.md](quantum-methodology.md) — the scientific
  charter: the hypothesis being tested (majorization order vs.
  Lieb–Robinson cone vs. causet order), the falsification criteria,
  scope and limitations.
* [quantum-plan.md](quantum-plan.md) — the implementation tracker:
  phase-by-phase deliverables, file layout, acceptance tests.

The C++ backend is [ITensor v3](https://itensor.org), vendored as a git
submodule under `third_party/itensor/`. The Python layer is a thin
result viewer that exposes only scalar config in / scalar diagnostics
out — no MPS or MPO objects cross the language barrier.

## Build

The quantum subsystem is opt-in via a CMake flag. Default builds of
tessera are unaffected:

```bash
# Full build with quantum support:
TESSERA_QUANTUM=1 pip install -e ".[dev]"

# Or for raw cmake:
cmake -S . -B build -DTESSERA_QUANTUM=ON
cmake --build build
```

Importing `tessera.quantum` from a build without the flag raises a
clear `ImportError` with the rebuild instruction.

## Hamiltonian

We implement the dimensional spin Hamiltonian from PLAN.md §4 (1-based
site indexing $n = 1\ldots N$), unitarily equivalent to Bañuls et al.
(2013) eq. 2.6 at $L_0 = 0$:

$$
H = H_\text{hop} + H_m + H_E
$$

$$
H_\text{hop} = \frac{1}{4a} \sum_{n=1}^{N-1}
              \bigl(X_n X_{n+1} + Y_n Y_{n+1}\bigr)
$$

$$
H_m = \frac{m}{2} \sum_{n=1}^{N} (-1)^n \sigma^z_n
$$

$$
H_E = \frac{g^2 a}{2} \sum_{n=1}^{N-1} L_n^2 ,
\quad
L_n = L_0 + \sum_{k=1}^{n}\!\Bigl[\tfrac{1-\sigma^z_k}{2} - \tfrac{1-(-1)^k}{2}\Bigr] .
$$

Bañuls' dimensionless parameters are $x = 1/(g^2 a^2)$ and
$\mu = 2m/(g^2 a)$. The continuum limit corresponds to $x \to \infty$
at fixed $m/g$.

## Quickstart

```python
from tessera.quantum import QuantumConfig, computeGroundState

cfg = QuantumConfig()
cfg.N = 20            # 1-based, even
cfg.a = 1.0
cfg.g = 1.0
cfg.m = 0.0           # massless
cfg.L0 = 0.0          # zero background field
cfg.maxBondDim = 100
cfg.nSweeps = 12

result = computeGroundState(cfg)
print(f"E = {result.energy:.6f}")
print(f"   = {result.operatorEnergy:.6f} (operator) "
      f"+ {result.constant:.6f} (c-number)")
print(f"bondDim = {result.bondDim}, "
      f"trunc_err ≤ {result.truncationErr:.0e}")
```

## Convergence checks

`computeGroundState` returns three diagnostic fields beyond the
energy:

* **`bondDim`** — the achieved MPS bond dimension. If it equals
  `config.maxBondDim`, the run is bond-dim-limited; bumping the cap
  and rerunning should give a (variationally) lower energy.
* **`truncationErr`** — a conservative upper bound on the SVD
  truncation error in the final sweep (currently equal to `cutoff`).
* **`operatorEnergy + constant`** must equal `energy` to ~1e-12. The
  split is useful when comparing against published numerics that
  include or exclude the c-number L_n² shift.

## Example: continuum approach

The script `examples/quantum/run_schwinger.py` runs scans over $x$ or
the bond-dim cap. The classic continuum-trend test (Bañuls fig. 6) at
$m/g = 0$ has the energy density $\omega_0 = E_\text{total} \cdot a/N$
descending toward Schwinger's exact value $\omega_0 = -1/\pi$:

```bash
$ python examples/quantum/run_schwinger.py --scan-x 1 4 16 --N 80 --max-bond-dim 120

Continuum approach: ω₀ → -1/π ≈ -0.318310 as x → ∞
   N    m/g      x    L0        E_total        omega_0  bondDim   trunc_err
--------------------------------------------------------------------------------
  80  0.000   1.00  0.00   -17.25453000    -0.21568163        13    1.00e-12
  80  0.000   4.00  0.00   -22.61543000    -0.28269000        29    1.00e-12
  80  0.000  16.00  0.00   -24.54466000    -0.30680600        67    1.00e-12
```

(Numbers vary slightly across runs but the trend is robust.)

## Phase 3 — Schmidt spectra and majorization poset

Phase 3 of the plan provides the entanglement-structure data that the
methodology charter (`docs/source/quantum-methodology.md`) builds its
hypothesis around. For each contiguous interval $A = [i, j]$ on the
chain, the Schmidt spectrum $\lambda_A$ is the list of eigenvalues of
$\rho_A = \mathrm{Tr}_{\bar A}|\psi\rangle\langle\psi|$, sorted
non-increasingly. Majorization $\lambda_A \preceq \lambda_B$ ("$B$ is at
least as concentrated as $A$") defines a partial order on the cuts, and
its Hasse diagram is the transitive reduction of the strict-majorization
relation.

### Pure-function API

The majorization predicate and poset constructor are exposed as pure
functions on plain Python lists:

```python
from tessera.quantum import majorizes, strictlyMajorizes, majorizationPoset

assert majorizes([1.0, 0.0], [0.5, 0.5])      # (1, 0) ≻ (½, ½)
assert not majorizes([0.5, 0.5], [1.0, 0.0])  # not the other way

poset = majorizationPoset([
    [1.0/3, 1.0/3, 1.0/3],   # node 0  — most uniform
    [0.5, 0.5],              # node 1  — middle
    [1.0],                   # node 2  — most concentrated
])
print(poset.getNodeCount, poset.covers)
# 3 [(2, 1), (1, 0)]
# (the direct (2, 0) edge has been transitively reduced away)
```

### End-to-end pipeline

`computeGroundStateMajorization(config)` runs DMRG, extracts every
contiguous-cut Schmidt spectrum, and builds the majorization poset in
one call:

```python
from tessera.quantum import QuantumConfig, computeGroundStateMajorization

cfg = QuantumConfig()
cfg.N = 10; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.0; cfg.L0 = 0.0
cfg.maxBondDim = 64; cfg.nSweeps = 10

r = computeGroundStateMajorization(cfg)

print(f"E_total = {r.groundState.energy:.6f}")
print(f"Schmidt cuts: {len(r.spectra.intervals)}")
for iv, spec in zip(r.spectra.intervals, r.spectra.spectra):
    print(f"  [{iv.i:>2}, {iv.j:>2}]  rank={len(spec)}  "
          f"top={spec[0]:.4f}")
print(f"Hasse cover edges: {len(r.poset.covers)}")
```

The cut family $\mathcal{F}$ excludes the trivial full-chain bipartition
$[1, N] \mid \emptyset$ — it has spectrum $(1)$ on every state and adds
no information. There are $N(N+1)/2 - 1$ contiguous cuts in total.

### Worked example: massless Schwinger at N=8

The script `examples/quantum/run_majorization.py` runs the full pipeline
and prints both the entropy-ranked interval table and the Hasse cover
edges:

```bash
$ python examples/quantum/run_majorization.py --N 8 --top 5

Schwinger ground state — N=8, m/g=0.0, x=1.0, L0=0.0
  E_total       = -1.62923162
  E_op          = -5.62923162
  E_const       = 4.00000000
  bondDim      = 13
  n_intervals   = 35
  n_cover_edges = 91

  i   j    entropy  rank  spectrum (top 4)
----------------------------------------------------------------------
  2   7   1.006393     4  0.6369 0.1610 0.1610 0.0410
  4   7   0.930880    16  0.6795 0.1700 0.1189 0.0306
  2   5   0.930880    16  0.6795 0.1700 0.1189 0.0306
  ...
```

The peak entropy is on the wide interior cut [2, 7], and boundary
intervals (such as the single-site cuts [1,1] and [8,8]) sit *above*
interior cuts in the poset because they are the *most concentrated*
spectra (least entangled). The Hasse direction is therefore "boundary
≻ interior", which is the expected behaviour for finite OBC chains.

## Phase 4 — q-qbar quench and TDVP real-time evolution

Phase 4 takes the Schwinger ground state into a non-equilibrium regime by
applying a *q-qbar quench operator* and integrating forward with two-site
TDVP. The quench is the Schwinger-language analogue of pair-creating a
heavy quark and antiquark connected by an electric flux string. The
construction follows Buyens et al. 2014 string-state formalism reduced to
the Gauss-eliminated spin chain.

### The quench operator

The pair is created by

$$
U_{q\bar q}(i_0, d) \;=\; \sigma^-_{i_0}\,\sigma^+_{i_0 + d}
$$

acting on the DMRG ground state. The two single-site flips shift the
electric-field variables $L_n$ on the $d$ links between $i_0$ and
$i_0+d$ by $+1$ relative to the vacuum, leaving links outside the
interval unchanged — i.e. they create a +1 flux tube of length $d$.

**Parity constraint**: the operator only acts non-trivially when the
two ends of the pair land on opposite sublattices of the heavy-quark
Néel vacuum $|\!\uparrow\downarrow\uparrow\downarrow\ldots\rangle$.
This requires $i_0$ to be odd (Up site) and $i_0 + d$ to be even (Dn
site), which forces $d$ to be **odd**. PLAN.md mentions $d=4$; the
acceptance test uses $d=5$ as the closest odd value.

### End-to-end pipeline: `runQqbarQuench`

The `runQqbarQuench(config)` Python function runs the full
DMRG → quench → TDVP loop in a single C++ call:

```python
from tessera.quantum import TDVPConfig, runQqbarQuench

cfg = TDVPConfig()
cfg.N = 14; cfg.a = 1.0; cfg.g = 1.0
cfg.m = 20.0                  # heavy-quark limit (m/g = 20)
cfg.L0 = 0.0
cfg.dmrgMaxBondDim = 64; cfg.dmrgNSweeps = 12
cfg.i0 = 5; cfg.d = 5         # odd-odd: σ⁻ at site 5, σ⁺ at site 10
cfg.dt = 0.05; cfg.T = 5.0    # 100 TDVP steps; T = d·a per PLAN.md
cfg.maxBondDim = 100
cfg.snapshotEvery = 5
cfg.recordSpectra = False    # ON for the Phase-5 majorization comparison
cfg.recordPoset   = False

result = runQqbarQuench(cfg)
print(f"GS energy:        {result.groundState.energy:.6f}")
print(f"Post-quench E:    {result.snapshots[0].energy:.6f}")
print(f"Snapshots:        {len(result.snapshots)}")
print(f"Final bondDim:   {result.snapshots[-1].bondDim}")
```

Each `TDVPSnapshot` carries `time`, `energy`, `bondDim`, the per-site
$\langle\sigma^z_n\rangle(t)$ (`zProfile`), and the per-link
$\langle L_n\rangle(t)$ (`lProfile`). With `recordSpectra = True` the
snapshot also gains the all-cuts Schmidt spectra; `recordPoset = True`
additionally builds the majorization poset of those spectra (each
costs $O(N^2)$ SVDs, so they're off by default).

### Worked example: heavy-quark flux tube

The script `examples/quantum/runQqbarQuench.py` runs the pipeline and
prints the $\langle L_n\rangle(t)$ profile at every snapshot. In the
heavy-quark limit the tube is rock-stable for the duration of the run:

```text
$ python examples/quantum/runQqbarQuench.py --N 12 --i0 3 --d 5 --T 2.0

DMRG ground state: E = -117.068809, bondDim = 3

    time         energy  bond  ⟨L_n⟩ profile (links 1..N-1)
   0.000     -77.543965     3   -1.000  0.000  0.000  1.000  0.000  1.000  0.000  0.000 -1.000 -0.000 -1.000
   0.200     -77.543965     3   -1.000  0.000 -0.000  1.000  0.000  1.000 -0.000  0.000 -1.000 -0.000 -1.000
   ...
   2.000     -77.543965     3   -1.000  0.000 -0.000  1.000  0.000  1.000 -0.000  0.000 -1.000 -0.000 -1.000
```

Reading the profile: the heavy-quark vacuum has $\langle L_n\rangle$
alternating $-1$ (odd link), $0$ (even link). The quench shifts links
3-7 by $+1$, giving the visible "0, 0, 1, 0, 1" plateau over the
expected interval — a perfect +1 flux tube of length 5. Energy stays
constant to machine precision; bond dimension stays at 3 because the
heavy mass suppresses entanglement growth.

At smaller $m/g$ (light-quark regime) the tube spreads / breaks
within $T \sim 1/(m/g)$ — that's the string-breaking dynamics Buyens
et al. study. Try ``--m-over-g 0.5 --T 4.0`` to see it in action.

## Phase 5 — causal-order comparison

Phase 5 ties Phases 1-4 together to test the methodology charter's
hypothesis (`docs/source/quantum-methodology.md`): that on a
TDVP-evolved Schwinger state, three independently-defined partial orders
on the (cut, time) label set agree.

### The three orders

For a TDVP run of duration $T$ with snapshots at $t_0=0, t_1, \dots, t_K$
and $\mathcal{F}$ the contiguous-interval cut family, the labels are
$\mathcal{L} = \mathcal{F} \times \{t_0, \dots, t_K\}$. We define:

* $(A, s) \preceq_{\mathrm{maj}} (B, t)$ — strict-majorization order on
  the Schmidt spectra; the Phase 3 majorizationPoset extended across
  cuts AND times.
* $(A, s) \preceq_{\mathrm{LR}} (B, t)$ iff $s < t$ and the
  interval-distance $d(A, B) \le v_{\mathrm{LR}} \cdot (t - s)$ —
  the Lieb-Robinson cone.
* $(A, s) \preceq_{\mathrm{cs}} (B, t)$ iff $s < t$ — the trivial
  causet order on a regular chain (Phase 6 makes this informative
  within a time slice too).

Each order is stored as a Hasse-cover :class:`Poset` over a shared
label list. Pairwise agreement is reported as Kendall-τ, the discordant
fraction, and the Hasse-graph edit distance.

### End-to-end pipeline

```python
from tessera.quantum import TDVPConfig, computeCausalComparison

cfg = TDVPConfig()
cfg.N = 12; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5; cfg.L0 = 0.0
cfg.dmrgMaxBondDim = 64; cfg.dmrgNSweeps = 12
cfg.i0 = 5; cfg.d = 3            # odd-odd parity
cfg.dt = 0.1; cfg.T = 1.0        # 10 TDVP steps
cfg.maxBondDim = 80
cfg.snapshotEvery = 1

report = computeCausalComparison(cfg, vLr=1.0)
print(f"nLabels = {report.nLabels}")
print(f"maj vs LR: τ = {report.majVsLr.kendallTau:.4f}")
print(f"LR vs cs:  τ = {report.lrVsCs.kendallTau:.4f}  "
      f"(must be 1.0; ≼_LR is a subset of ≼_cs by construction)")
```

The function forces ``recordSpectra = True`` regardless of the input,
because the spectra are required to build ≼_maj. ``vLr`` defaults to
1.0 (free-fermion group velocity for our hopping coefficient); the
plan §7 mentions OTOC-based extraction for higher-precision use.

### vLr-monotonicity sanity

Larger $v_{\mathrm{LR}}$ ⇒ more pairs satisfy $d(A, B) \le v_{\mathrm{LR}}(t-s)$
⇒ ≼_LR has at least as many cover edges. Because ≼_cs is fixed,
``lrVsCs.nComparableBoth`` is monotone non-decreasing in $v_{\mathrm{LR}}$.
This is asserted in the C++ acceptance test and the Python pytest.

### Worked example

The script `examples/quantum/run_causal_compare.py` runs the pipeline
and prints all three pairwise comparisons:

```bash
$ python examples/quantum/run_causal_compare.py --N 8 --T 0.4 --scan-vlr 1.0 16.0

=== vLr = 1.0 ===
  nLabels    = 175
  nSnapshots = 5
  pair           Kendall-τ     discord   edit-dist   comparable
  majVsLr         0.1577      0.4212      0.9724         5670
  majVsCs         0.1525      0.4238      0.9704         6650
  lrVsCs          1.0000      0.0000      0.3429        10564

=== vLr = 16.0 ===
  ... same maj_vs_*; lrVsCs.comparable larger (cone wider)
```

Key invariant: ``lrVsCs.kendallTau = 1.0`` exactly (≼_LR is a strict
subset of ≼_cs by construction). Any deviation flags an implementation
bug.

## Phase 6 — tessera-Spacetime → causet-embedded chain

Phase 6 (`docs/source/quantum-plan.md` §6) replaces the regular 1D
lattice with a "chain of antichains" sourced from a
:class:`tessera.Spacetime`. Two pieces of that integration have landed
and are usable from Python today:

1. **Inherited Hasse-cover Poset on the Spacetime vertex set**
   — :meth:`tessera.quantum.Poset.fromSpacetime` walks every timelike
   edge in the Spacetime (an edge whose ``squaredLength < 0``),
   orients it earliest-time → latest-time, transitively closes the
   resulting DAG, and emits the cover edges (transitive reduction)
   as a :class:`tessera.quantum.Poset`. This is the natural "≼_cs" of
   Phase 5 promoted to a non-trivial within-time-slice structure.

2. **Chain-of-antichains adapter**
   — :func:`tessera.quantum.extractCausetChain` walks the Spacetime,
   groups vertices by integer time slice (`Vertex.getTime()`
   truncated to int), and packages four pieces of data:

   * `antichains[s]` — the vertex IDs at time slice
     `times[s]`, in ascending-ID order;
   * `vertexIds[i]` — the flat lattice-site → Spacetime vertex ID
     mapping (concatenation of antichains in time order);
   * `hoppingPairs` — pairs `(i, j)` of flat-site indices coupled
     by adjacent-slice timelike edges (this would replace the
     `Σ_n (X_n X_{n+1} + Y_n Y_{n+1})` hopping term in `H_hop` on a
     causet-embedded chain);
   * `partialOrder` — the inherited Hasse-cover :class:`Poset` on
     the same flat-lattice label set, equivalent to
     ``Poset.fromSpacetime(spacetime)`` modulo the dense remap.

```python
import tessera
from tessera.quantum import extractCausetChain, Poset

# Tiny CDT spacetime as a stand-in for a causet — same APIs apply.
metric = tessera.Metric(True, tessera.Signature(4, tessera.Lorentzian))
st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                     tessera.PREFERRED, tessera.Toroid())
st.build(20)

chain = extractCausetChain(st)
print(f"nSites = {chain.nSites}")
print(f"layers  = {[len(a) for a in chain.antichains]}")
print(f"hops    = {len(chain.hoppingPairs)}")
print(f"poset   = {chain.partialOrder}")
```

Sample output on a 20-simplex Toroid CDT::

    nSites = 15
    layers  = [5, 5, 5]
    hops    = 30
    poset   = Poset(getNodeCount=15, covers=30 edges)

For the trivial case where every time slice has a single vertex,
`hoppingPairs == [(0, 1), (1, 2), …]` and the existing
:func:`computeGroundState` runs directly with `params.N =
chain.nSites`. Multi-vertex antichains (genuine causet branching)
still flatten to a 1D MPS layout, but the resulting hopping graph
includes pairs with stride > 1; that's the trigger for moving to a
tree-tensor-network MPO down the line. The MPO construction on top
of `CausetChain` is left for a follow-up — this commit delivers the
data extraction and the inherited partial order.

### Visualisation

`Poset.toDot()` renders the Hasse diagram in Graphviz DOT format,
which makes the inherited causet structure straightforward to plot::

    poset = Poset.fromSpacetime(st)
    open("/tmp/causet.dot", "w").write(poset.toDot())
    # then: dot -Tsvg /tmp/causet.dot -o /tmp/causet.svg

Cover edges run from earlier-time vertex to later-time vertex (the
strict-precedes direction); the diagram should layer cleanly when
the underlying foliation is :data:`tessera.PREFERRED`.

## Tested benchmarks

The C++ and Python test suites cross-check every layer of the pipeline:

| Test | Layer | What it verifies |
|---|---|---|
| `test_itensor_hello.cpp` | Phase 0 | Heisenberg $N{=}8$ DMRG vs. dense Eigen ED ($\sim$1e-14). |
| `test_schwinger_spectrum.cpp` | Phase 1 | DMRG vs. dense Eigen ED on the full $2^N$ basis, $N \in \{4,6,8\}$, $m/g \in \{0, 0.125, 0.25\}$, $L_0 \in \{0, 0.5\}$ — agreement to $10^{-12}$. |
| `test_schwinger_limits.cpp`   | Phase 1 | Free-fermion analytic limit ($g{=}m{=}0$) and strong-coupling vacuum ($m \to \infty$). |
| `test_schwinger_paper.cpp`    | Phase 1 | Continuum trend $\omega_0 \to -1/\pi$, vector mass gap, chiral condensate, charge-conjugation parity. |
| `test_phase2_compute_ground_state.py` | Phase 2 | Phase 1 numerics reproduced through the Python API to $10^{-8}$. |
| `test_phase2_api_robustness.py` | Phase 2 | Validation, variational descent, reproducibility, conserveQns flag, L0 dependence, doctests. |
| `test_majorization.cpp` | Phase 3 | Majorization predicate properties (reflexivity, transitivity, anti-symmetry, transitive reduction). |
| `test_schmidt_spectra.cpp` | Phase 3 | Schmidt extraction on product, GHZ, Bell, singlet inputs. |
| `test_majorization_poset.cpp` | Phase 3 | Acceptance: product → 0 Hasse edges, GHZ → 0 strict edges, Bell + product → $(1,0) \succ (\tfrac{1}{2},\tfrac{1}{2})$ edge present. |
| `test_schwinger_schmidt_cross_check.cpp` | Phase 3 | MPS-side Schmidt vs. dense ED on Schwinger ground states for 8 (N, m, L₀) cases — 131 individual cuts to $10^{-9}$. |
| `test_phase3_majorization_python.py` | Phase 3 | Pipeline reproduction through Python: majorization predicate, poset properties (acyclic, irreflexive, transitively reduced), complement symmetry, strong-mass collapse. |
| `test_tdvp_string.cpp` | Phase 4 | Heavy-quark flux-tube test: $\langle L_n\rangle(t)$ matches +1-tube reference at $t=0$ and $t=T/2$ to within 0.05; $\|\Delta E\|/\|E_0\| < 10^{-3}$ over $T = d \cdot a$. |
| `test_phase4_tdvp_python.py` | Phase 4 | Python pipeline tests: parity validation, snapshot schedule, energy conservation, total-charge conservation, optional spectra/poset recording. |
| `test_tdvp_vs_dense.cpp` | Phase 4 | TDVP integrator vs $e^{-iHt}$ dense unitary evolution on the full $2^N$ Hilbert space, $N \le 8$, four parameter regimes — agreement to $3\times10^{-6}$. |
| `test_schwinger_n4_hamiltonian.cpp` | Phase 1 | PLAN.md §7 trap: independent symbolic evaluation of the H formula on every N=4 basis state, machine-precision match. |
| `test_causal_compare.cpp` | Phase 5 | End-to-end three-order comparison; sanity checks on Kendall-τ ranges; vLr-monotonicity sanity. |
| `test_phase5_causal_compare_python.py` | Phase 5 | Python pipeline: identical/reversed/disjoint pure compareOrders cases, end-to-end pipeline checks, ≼_LR ⊂ ≼_cs invariant (τ = 1.0 exactly), vLr monotonicity. |
| `test_poset_from_spacetime.cpp` | Phase 6 | Hand-crafted Spacetimes (2-slice ladder, 3-slice chain with skip, empty, spacelike-only, sparse-ID, self-comparison, DOT export) — verifies `Poset::fromSpacetime` Hasse covers, transitive reduction, dense ID remapping. |
| `test_causet_chain.cpp` | Phase 6 | `extractCausetChain` invariants on trivial chain, branching antichain, sparse IDs, slice-spanning skip edges. |
| `test_phase6_causet_chain_python.py` | Phase 6 | Python-side self-consistency on a default CDT Spacetime: layer sizes, flat-index alignment, partialOrder ⊆ hoppingPairs invariant, DOT round-trip. |
| `test_phase6_cdt_invariants.cpp` | Phase 6 | Foliated-CDT structural invariants (Ambjorn 2004): every Hasse cover spans exactly one slice; covers ≡ hoppingPairs (no transitive reduction on a foliated complex); Hasse height = num_layers − 1; total extraction; top-layer out-degree / bottom-layer in-degree both 0. |
| `test_schwinger_paper.cpp::check_vector_mass_continuum_trend` | Phase 1 | Bañuls 2013 fig. 7a M_V/g continuum trend at $m/g = 0$: scan $(x, N) \in \{(4, 40), (16, 80)\}$, verify monotone descent of the first-excited-state gap toward $1/\sqrt{\pi}$ and $\Delta E/g$ in $[0.55, 0.75]$ at $1/\sqrt{x}=0.25$ (matching their published 0.61). |
| `test_schmidt_invariants_dmrg.cpp` | Phase 1 / 3 | Universal density-matrix invariants on every contiguous-cut Schmidt spectrum of Schwinger DMRG ground states across $(N, m, L_0)$ grid: $\sum \lambda = 1$, $\lambda \in [0, 1]$, descending order, rank $\le 2^{\min(w, N-w)}$. 330 spectra validated to $\sim 10^{-15}$. |

## API reference

```{eval-rst}
.. automodule:: tessera.quantum
    :members:
    :undoc-members:
    :show-inheritance:
```

## References

* Buyens, Haegeman, Van Acoleyen, Verschelde, Verstraete,
  *Matrix product states for gauge field theories*,
  Phys. Rev. Lett. **113**, 091601 (2014),
  [arXiv:1312.6654](https://arxiv.org/abs/1312.6654) — the
  string-state construction underlying the Phase 4 q-qbar quench.
* Haegeman, Lubich, Oseledets, Vandereycken, Verstraete,
  *Unifying time evolution and optimization with matrix product states*,
  Phys. Rev. B **94**, 165116 (2016),
  [arXiv:1408.5056](https://arxiv.org/abs/1408.5056) — the TDVP
  algorithm we wrap.
* Bañuls, Cichy, Cirac, Jansen,
  *The mass spectrum of the Schwinger model with Matrix Product States*,
  JHEP **11**, 158 (2013), [arXiv:1305.3765](https://arxiv.org/abs/1305.3765).
* Nielsen,
  *Conditions for a class of entanglement transformations*,
  Phys. Rev. Lett. **83**, 436 (1999),
  [quant-ph/9811053](https://arxiv.org/abs/quant-ph/9811053) — the
  majorization characterization of LOCC entanglement convertibility,
  which underpins the partial order this code constructs.
* Schwinger, *Gauge Invariance and Mass II*, Phys. Rev. **128**,
  2425 (1962).
* Coleman, *More about the massive Schwinger model*, Ann. Phys.
  **101**, 239 (1976).
* Kogut, Susskind, *Hamiltonian formulation of Wilson's lattice gauge
  theories*, Phys. Rev. D **11**, 395 (1975).
* Pichler, Dalmonte, Rico, Zoller, Montangero,
  *Real-time dynamics in U(1) lattice gauge theories with tensor
  networks*, Phys. Rev. X **6**, 011023 (2016).

## See also

* [`docs/source/quantum-plan.md`](quantum-plan.md) — the full multi-phase
  plan including TDVP and majorization-poset extensions still to come.
