# Quantum subsystem (Schwinger MPS / DMRG)

The `caset.quantum` subpackage extends caset with a tensor-network
treatment of the 1+1D Kogut-Susskind Schwinger model — the simplest
non-trivial gauge theory and a standard benchmark for lattice tensor-
network methods. It implements the staged plan in
`docs/source/quantum-plan.md`. As of this writing Phases 0-3 are
complete (scaffolding, MPO + DMRG ground states, Python bindings,
Schmidt spectra + majorization poset); later phases (real-time TDVP,
causal-order comparison) build on the same primitives.

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
caset are unaffected:

```bash
# Full build with quantum support:
CASET_QUANTUM=1 pip install -e ".[dev]"

# Or for raw cmake:
cmake -S . -B build -DCASET_QUANTUM=ON
cmake --build build
```

Importing `caset.quantum` from a build without the flag raises a
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
from caset.quantum import QuantumConfig, compute_ground_state

cfg = QuantumConfig()
cfg.N = 20            # 1-based, even
cfg.a = 1.0
cfg.g = 1.0
cfg.m = 0.0           # massless
cfg.L0 = 0.0          # zero background field
cfg.max_bond_dim = 100
cfg.n_sweeps = 12

result = compute_ground_state(cfg)
print(f"E = {result.energy:.6f}")
print(f"   = {result.operator_energy:.6f} (operator) "
      f"+ {result.constant:.6f} (c-number)")
print(f"bond_dim = {result.bond_dim}, "
      f"trunc_err ≤ {result.truncation_err:.0e}")
```

## Convergence checks

`compute_ground_state` returns three diagnostic fields beyond the
energy:

* **`bond_dim`** — the achieved MPS bond dimension. If it equals
  `config.max_bond_dim`, the run is bond-dim-limited; bumping the cap
  and rerunning should give a (variationally) lower energy.
* **`truncation_err`** — a conservative upper bound on the SVD
  truncation error in the final sweep (currently equal to `cutoff`).
* **`operator_energy + constant`** must equal `energy` to ~1e-12. The
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
   N    m/g      x    L0        E_total        omega_0  bond_dim   trunc_err
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
from caset.quantum import majorizes, strictly_majorizes, majorization_poset

assert majorizes([1.0, 0.0], [0.5, 0.5])      # (1, 0) ≻ (½, ½)
assert not majorizes([0.5, 0.5], [1.0, 0.0])  # not the other way

poset = majorization_poset([
    [1.0/3, 1.0/3, 1.0/3],   # node 0  — most uniform
    [0.5, 0.5],              # node 1  — middle
    [1.0],                   # node 2  — most concentrated
])
print(poset.n_nodes, poset.covers)
# 3 [(2, 1), (1, 0)]
# (the direct (2, 0) edge has been transitively reduced away)
```

### End-to-end pipeline

`compute_ground_state_majorization(config)` runs DMRG, extracts every
contiguous-cut Schmidt spectrum, and builds the majorization poset in
one call:

```python
from caset.quantum import QuantumConfig, compute_ground_state_majorization

cfg = QuantumConfig()
cfg.N = 10; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.0; cfg.L0 = 0.0
cfg.max_bond_dim = 64; cfg.n_sweeps = 10

r = compute_ground_state_majorization(cfg)

print(f"E_total = {r.ground_state.energy:.6f}")
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
  bond_dim      = 13
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

## Tested benchmarks

The C++ and Python test suites cross-check every layer of the pipeline:

| Test | Layer | What it verifies |
|---|---|---|
| `test_itensor_hello.cpp` | Phase 0 | Heisenberg $N{=}8$ DMRG vs. dense Eigen ED ($\sim$1e-14). |
| `test_schwinger_spectrum.cpp` | Phase 1 | DMRG vs. dense Eigen ED on the full $2^N$ basis, $N \in \{4,6,8\}$, $m/g \in \{0, 0.125, 0.25\}$, $L_0 \in \{0, 0.5\}$ — agreement to $10^{-12}$. |
| `test_schwinger_limits.cpp`   | Phase 1 | Free-fermion analytic limit ($g{=}m{=}0$) and strong-coupling vacuum ($m \to \infty$). |
| `test_schwinger_paper.cpp`    | Phase 1 | Continuum trend $\omega_0 \to -1/\pi$, vector mass gap, chiral condensate, charge-conjugation parity. |
| `test_phase2_compute_ground_state.py` | Phase 2 | Phase 1 numerics reproduced through the Python API to $10^{-8}$. |
| `test_phase2_api_robustness.py` | Phase 2 | Validation, variational descent, reproducibility, conserve_qns flag, L0 dependence, doctests. |
| `test_majorization.cpp` | Phase 3 | Majorization predicate properties (reflexivity, transitivity, anti-symmetry, transitive reduction). |
| `test_schmidt_spectra.cpp` | Phase 3 | Schmidt extraction on product, GHZ, Bell, singlet inputs. |
| `test_majorization_poset.cpp` | Phase 3 | Acceptance: product → 0 Hasse edges, GHZ → 0 strict edges, Bell + product → $(1,0) \succ (\tfrac{1}{2},\tfrac{1}{2})$ edge present. |
| `test_schwinger_schmidt_cross_check.cpp` | Phase 3 | MPS-side Schmidt vs. dense ED on Schwinger ground states for 8 (N, m, L₀) cases — 131 individual cuts to $10^{-9}$. |
| `test_phase3_majorization_python.py` | Phase 3 | Pipeline reproduction through Python: majorization predicate, poset properties (acyclic, irreflexive, transitively reduced), complement symmetry, strong-mass collapse. |

## API reference

```{eval-rst}
.. automodule:: caset.quantum
    :members:
    :undoc-members:
    :show-inheritance:
```

## References

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
