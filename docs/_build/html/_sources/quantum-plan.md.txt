# PLAN.md — Schwinger-Model Majorization Poset Extension for `caset`

Hand this file to Claude Code as the working spec. Each phase has a
concrete deliverable, a file layout, and an acceptance test. Do not
skip acceptance tests; every phase gates the next.

The hypothesis this plan is built to test, plus falsification criteria,
limitations, and bibliographic context, are in
[quantum-methodology.md](quantum-methodology.md). The user-facing API
documentation for what's built so far is in [quantum.md](quantum.md).

## 0. Goal and scope

Extend `caset` with a C++ subsystem that:

1. Represents the 1+1D Kogut–Susskind Schwinger-model Hamiltonian as
   a Matrix Product Operator (MPO) on a Jordan–Wigner'd spin chain.
2. Finds its ground state by DMRG.
3. Evolves it in real time by TDVP after a local quench creating a
   $q\bar{q}$ pair at separation $d$.
4. At each time slice, computes Schmidt spectra $\lambda_{[i,j]}(t)$
   for all contiguous-interval cuts $[i,j]$, builds a majorization
   poset, and compares it against (a) the Lieb–Robinson cone and
   (b) a causet causal order when the chain is embedded in a causet.

## 1. Architectural principle — minimize Python/C++ crossings

The entire simulation loop stays in C++. Python is a result viewer,
not a driver.

```
   C++ side:  build MPO  ->  DMRG  ->  TDVP loop  ->  spectra  ->  poset
                                                                    |
   Py side:                                                      export
                                                                    |
                                                    plot, save, interop
```

Only these cross the barrier:

- **In** (Py → C++): scalar config (N, m, g, Δt, T, cut family, seed).
- **Out** (C++ → Py): final/periodic snapshots of observables and
  the poset as plain data (`std::vector<double>`, adjacency lists).

Do not expose MPS tensors or MPOs to Python. Do not call Python
callbacks inside the time loop.

## 2. Dependencies

| Purpose | Library | Why |
|---|---|---|
| MPS/MPO, DMRG, TDVP | **ITensor (C++)** v3 | Mature, header-heavy C++, built-in DMRG+TDVP, handles quantum-number conservation |
| Linear algebra (small ops outside ITensor) | Eigen 3 | already likely in caset's toolchain |
| Graph / poset | Boost.Graph | Hasse diagram, transitive reduction |
| Python bindings | pybind11 | already in caset |
| Tests | Catch2 or doctest | whichever caset already uses |

Add ITensor as a git submodule under `third_party/itensor` and wire
via CMake `add_subdirectory`. ITensor depends on BLAS/LAPACK — reuse
what caset already has. Do not pull ITensor through a package manager;
pin a known-good commit.

## 3. File layout

```
caset/
  include/caset/quantum/
    schwinger_model.hpp        # MPO construction
    dmrg_runner.hpp            # ground-state driver
    tdvp_runner.hpp            # real-time driver
    schmidt.hpp                # spectrum extraction
    majorization.hpp           # partial order, poset
    causal_compare.hpp         # LR / caset comparison
  src/quantum/
    schwinger_model.cpp
    dmrg_runner.cpp
    tdvp_runner.cpp
    schmidt.cpp
    majorization.cpp
    causal_compare.cpp
  tests/quantum/
    test_schwinger_spectrum.cpp   # Phase 1 acceptance
    test_majorization.cpp         # Phase 3 acceptance
    test_tdvp_string.cpp          # Phase 4 acceptance
    test_causal_compare.cpp       # Phase 5 acceptance
  python/caset/quantum/
    __init__.py
    _bindings.cpp              # pybind11, only exports final data
  examples/quantum/
    run_schwinger.cpp
    qqbar_quench.cpp
```

## 4. Physics spec (fix once, reference everywhere)

Open-BC staggered-fermion Schwinger model, Jordan–Wigner'd,
Gauss-law-eliminated. One qubit per site, $N$ sites total.

$$H = H_\text{hop} + H_m + H_E$$

$$H_\text{hop} = \frac{1}{4a}\sum_{n=1}^{N-1}\bigl(X_n X_{n+1} + Y_n Y_{n+1}\bigr)$$

$$H_m = \frac{m}{2}\sum_{n=1}^{N}(-1)^{n}\,Z_n$$

$$H_E = \frac{g^2 a}{2}\sum_{n=1}^{N-1} L_n^2, \qquad L_n = L_0 + \sum_{k=1}^{n}\left[\frac{1-Z_k}{2} - \frac{1-(-1)^k}{2}\right]$$

with background field $L_0$ (keep as a parameter, default 0). Expand
$L_n^2$; the result is a long-range $Z_m Z_{m'}$ Hamiltonian whose
MPO bond dimension grows with $N$. ITensor's `AutoMPO` handles this
natively — use it, do not roll a hand-built MPO.

Reference for all numerics: Bañuls, Cichy, Cirac, Jansen,
*JHEP* **11**, 158 (2013).

## 5. Phased implementation

### Phase 0 — scaffolding (≤ 2 days)

- Add ITensor submodule; CMake build verifies with ITensor's own
  `sample/dmrg.cc` compiled against the caset toolchain.
- Create the directory layout above with empty stubs.
- CI runs on each PR.

**Acceptance**: `make test_itensor_hello` returns the ground state
energy of the Heisenberg chain for $N=20$ to within $10^{-6}$ of the
ITensor reference value.

### Phase 1 — Schwinger MPO (2–3 days)

- Implement `schwinger_model.hpp`: a class that takes $(N, a, m, g, L_0)$
  and returns an `ITensor::MPO`.
- Use `AutoMPO` to avoid manually tracking the $L_n^2$ expansion.
- Conserve $U(1)$ total charge; quantum numbers on every index.

**Acceptance** (`test_schwinger_spectrum.cpp`):
DMRG ground-state energy per site for $N=20$, $a=1$, $m/g \in \{0.0,
0.125, 0.25\}$ matches Bañuls et al. 2013 Table 1 to $< 10^{-3}$.
Bond dimension 100 is enough for this check.

### Phase 2 — DMRG runner (2 days)

- `dmrg_runner.hpp`: thin wrapper around `ITensor::dmrg`.
  Inputs: `SchwingerMPO`, sweep schedule, max bond dim, noise, Krylov
  dim. Outputs: `MPS` ground state, converged energy, final bond dim.
- Expose `compute_ground_state(config)` through pybind11 returning
  only $(E_0, \text{bond\_dim}, \text{truncation\_err})$.

**Acceptance**: re-runs Phase 1 with the wrapper; numerics unchanged.

### Phase 3 — Schmidt spectra and majorization poset (3 days)

- `schmidt.hpp`: given an MPS in right canonical form and an interval
  $[i,j]$, return $\lambda_{[i,j]}$ via SVD at the appropriate bond
  (ITensor exposes Schmidt values directly after orthogonalization).
- For contiguous cuts this is $O(N^2)$ spectra; store as a
  `std::vector<std::vector<double>>`.
- `majorization.hpp`:

  ```cpp
  // lambda <_maj mu iff, with both sorted descending, zero-padded,
  //   sum_{i=1..k} lambda_i  <=  sum_{i=1..k} mu_i   for all k,
  //   and total probability equal (trivially 1 here).
  bool majorizes(const std::vector<double>& mu,
                 const std::vector<double>& lambda,
                 double tol = 1e-12);
  ```

- Build a DAG whose nodes are intervals `[i,j]` and edges represent
  the majorization relation; apply Boost.Graph transitive reduction
  to obtain the Hasse diagram.
- Store as `caset::Poset` (define this type; reuse caset's existing
  causet adjacency structure if the API fits).

**Acceptance** (`test_majorization.cpp`):

1. Product state $|0\rangle^{\otimes N}$: every spectrum is
   $(1, 0, \ldots)$; Hasse diagram has no edges.
2. $N$-qubit GHZ: every single-site spectrum is $(\tfrac{1}{2},
   \tfrac{1}{2})$; pairwise majorization relations all equal (no
   strict edges).
3. Bell state across the center cut of $N=2$: spectrum
   $(\tfrac{1}{2}, \tfrac{1}{2})$; compared against product state
   $(\tfrac{1}{2}, \tfrac{1}{2}) \prec (1,0)$ correctly.

### Phase 4 — TDVP with $q\bar{q}$ quench (4 days)

- `tdvp_runner.hpp`: wraps ITensor's 2-site TDVP. Inputs: MPS, MPO,
  $\Delta t$, $T$, max bond dim, observables callback (pure C++).
- Initial state: DMRG ground state, then apply $\sigma^+_{i_0}
  \sigma^-_{i_0 + d}$ (normalized) to create a $q\bar{q}$ pair.
  Gauss's law requires appropriate string of $L$ shifts between them
  — handle via the electric-field string operator; reference Buyens
  et al. 2014 for the exact construction.
- At each recorded timestep, invoke observables callback that records:
  - $\langle L_n \rangle_t$ (electric-field profile)
  - $\langle Z_n \rangle_t$ (charge density)
  - $\lambda_{[i,j]}(t)$ for all contiguous cuts
  - full majorization poset

**Acceptance** (`test_tdvp_string.cpp`):

- Heavy-quark limit $m/g \gg 1$, $d=4$, $T = d\cdot a$ at $a=1$, $g=1$:
  $\langle L_n \rangle_t$ is approximately uniform at value $\pm 1$
  between $i_0$ and $i_0 + d$ (flux tube) and zero outside, to within
  $0.05$ after $t = T/2$.
- Energy conservation: $|\Delta E|/|E_0| < 10^{-3}$ over the run.

### Phase 5 — causal-order comparison (3 days)

- `causal_compare.hpp`: builds three partial orders on the label set
  $\{(\text{cut}, t)\}$:
  1. $\prec_\text{maj}$: from Phase 3 Hasse diagram, per time slice.
  2. $\prec_\text{LR}$: from the Lieb–Robinson velocity $v_\text{LR}$
     extracted by fitting OTOC ($\langle [\sigma^+_i(t), \sigma^-_j]^\dagger
     [\sigma^+_i(t), \sigma^-_j]\rangle$) front propagation.
  3. $\prec_\text{caset}$: if the chain is embedded in a causet, inherit
     from caset's existing causal order.
- Compute agreement statistics: Kendall-$\tau$, fraction of discordant
  pairs, graph edit distance on Hasse diagrams.
- Output: a single struct
  `struct CausalComparisonReport { double tau_maj_lr; double tau_maj_cs; ... }`

**Acceptance** (`test_causal_compare.cpp`):

- On a regular (total-order) causet chain: $\prec_\text{maj}$ agrees
  with $\prec_\text{LR}$ above some $v_E \leq v_\text{LR}$; both agree
  with $\prec_\text{caset}$ since the causet is trivially totally
  ordered per time slice.
- Agreement rate quoted with uncertainty from bootstrap over Trotter
  seeds.

### Phase 6 — caset integration (1–2 weeks, optional for v1)

- Replace the regular 1D lattice with a chain of antichains sourced
  from `caset::CausalSet`.
- Hopping in $H_\text{hop}$ follows causet links, not lattice neighbors.
- Rebuild MPO as a tree-tensor-network when the causet is not totally
  ordered across a time slice (ITensor has tree TN support; otherwise
  fall back to a chain with reindexing).
- Compare the three partial orders on the causet-embedded state.

### Phase 7 — MERA / KAK / EML (research, not v1)

Do not attempt until Phases 1–5 produce publication-quality numbers.
Replace MPS with binary MERA; KAK-decompose disentanglers; EML-
parameterize KAK angles for symbolic regression over couplings.
Reference: Evenbly–Vidal 2009 for MERA algorithms; Hauru–Van Damme–
Haegeman 2021 for Riemannian optimization; Odrzywołek 2026 for EML.

## 6. Python layer (minimal)

`python/caset/quantum/_bindings.cpp` exposes exactly:

```cpp
py::class_<QuantumConfig>(m, "QuantumConfig")
    .def(py::init<>())
    .def_readwrite("N", &QuantumConfig::N)
    .def_readwrite("a", &QuantumConfig::a)
    .def_readwrite("m", &QuantumConfig::m)
    .def_readwrite("g", &QuantumConfig::g)
    .def_readwrite("L0", &QuantumConfig::L0)
    .def_readwrite("max_bond_dim", &QuantumConfig::max_bond_dim)
    .def_readwrite("dt", &QuantumConfig::dt)
    .def_readwrite("T", &QuantumConfig::T);

py::class_<GroundStateResult>(m, "GroundStateResult")
    .def_readonly("energy", &GroundStateResult::energy)
    .def_readonly("bond_dim", &GroundStateResult::bond_dim)
    .def_readonly("truncation_err", &GroundStateResult::truncation_err);

py::class_<QuenchResult>(m, "QuenchResult")
    .def_readonly("times", &QuenchResult::times)
    .def_readonly("L_profile", &QuenchResult::L_profile)  // [t][n]
    .def_readonly("Z_profile", &QuenchResult::Z_profile)
    .def_readonly("posets", &QuenchResult::posets);       // one per t

m.def("compute_ground_state", &compute_ground_state);
m.def("run_qqbar_quench", &run_qqbar_quench);
```

No MPS, MPO, or ITensor type crosses the barrier.

## 7. Conventions and traps

- **Sign of $(-1)^n$ in $H_m$**: staggered convention, site index starts
  at $n=1$. Bañuls et al. use the same; match their sign.
- **Gauss's law**: after elimination, the long-range $Z_m Z_{m'}$ term
  has coefficients that depend on the parity of sites between $m$ and
  $m'$. Verify by independent sum on $N=4$ before trusting the MPO.
- **Truncation in TDVP**: 2-site TDVP grows bond dim; 1-site TDVP
  conserves it. Start with 2-site for the quench, switch to 1-site
  once bond dim saturates your budget.
- **Schmidt spectrum zeros**: pad to equal length before majorization
  comparison; otherwise partial-sum test is miscounted.
- **Boost.Graph transitive reduction**: not in Boost's public API
  directly; implement via DFS over comparability graph if needed.
- **OTOC computation**: expensive. Compute only at the final time and
  at one or two intermediate times for $v_\text{LR}$ estimation; do
  not run every step.

## 8. First week targets

- Day 1–2: Phase 0 done, ITensor builds inside caset's CMake tree.
- Day 3–4: Phase 1 MPO constructed; matches Bañuls 2013 values.
- Day 5–6: Phase 2 wrapper; Phase 3 majorization infrastructure on
  trivial and GHZ test states.
- End of week 1: DMRG + majorization pipeline end-to-end on $N=20$
  Schwinger ground state.

## 9. Open questions to defer

- Whether to quantum-number-decompose spectra before majorization
  (majorization within each $U(1)$ sector vs. global). Default: global
  for v1; revisit.
- Whether to normalize spectra per cut (Schmidt coefficients squared
  sum to 1 automatically for normalized states; sanity-check this
  after TDVP truncation).
- Whether the "cut family" should include non-contiguous subsets
  eventually. Default: contiguous only, per the majorization
  well-definedness argument.

## 10. Non-goals for v1

- GPU acceleration of the tensor contractions. ITensor has partial
  CUDA support; ignore until CPU performance is a bottleneck.
- Non-abelian gauge groups ($SU(2)$, $SU(3)$). Schwinger is $U(1)$;
  that is the full v1 scope.
- 2+1D. Chain only for v1.
- Replacing ITensor. Do not write a bespoke MPS library.
