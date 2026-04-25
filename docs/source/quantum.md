# Quantum subsystem (Schwinger MPS / DMRG)

The `caset.quantum` subpackage extends caset with a tensor-network
treatment of the 1+1D Kogut-Susskind Schwinger model — the simplest
non-trivial gauge theory and a standard benchmark for lattice tensor-
network methods. It implements the staged plan in
`docs/source/quantum-plan.md`. As of this writing Phases 0-2 are
complete (scaffolding, MPO + DMRG ground states, Python bindings); later
phases (Schmidt poset, real-time TDVP, causal-order comparison) build on
the same primitives.

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

## Tested benchmarks

The C++ test suite (`tests/quantum/test_schwinger_*.cpp`) cross-checks
the MPO/DMRG implementation against:

| Test | What it verifies |
|---|---|
| `test_schwinger_spectrum.cpp` | DMRG vs. dense Eigen ED on the full $2^N$ basis, $N \in \{4,6,8\}$, $m/g \in \{0, 0.125, 0.25\}$, $L_0 \in \{0, 0.5\}$ — agreement to $10^{-12}$. |
| `test_schwinger_limits.cpp`   | Free-fermion analytic limit ($g{=}m{=}0$) and strong-coupling vacuum ($m \to \infty$). |
| `test_schwinger_paper.cpp`    | Continuum trend $\omega_0 \to -1/\pi$, vector mass gap, chiral condensate, charge-conjugation parity. |
| `test_phase2_compute_ground_state.py` | Phase 1 numerics reproduced through the Python API to $10^{-8}$. |

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
