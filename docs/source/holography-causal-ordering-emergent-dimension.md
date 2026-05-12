# Emergent Spectral Dimension from the Schwinger TDVP State

**Status:** implemented end-to-end under `tessera.quantum.holography`.
All acceptance criteria in §H1, §H2, §H3, §H4 pass; §8 threats are
addressed by a Savitzky-Golay smoother for D_S(σ), a convergence
sweep script over (χ, K, ε_I, max_temporal_stride), and a regression
test cross-checking the holography pipeline against the causal-order
comparison. §10 JSON output is implemented. §H5 documents the
remaining tuning knobs. §H6 (CausetChain integration) is the only
unlanded piece — the spec marks it optional.

**Companion docs (read together):**
- `docs/source/quantum.md` — Schwinger MPS / DMRG / TDVP user-facing
  reference; the **Emergent spectral dimension** section there is
  the user-facing quickstart for the implementation that this
  document specifies.
- `docs/source/quantum-methodology.md` — scientific charter for the
  entanglement → causal-order programme
- `docs/source/quantum-plan.md` — feature-by-feature implementation
  tracker
- `docs/source/cpp_api.md` — C++ API (Spacetime, Simplex, etc.)

**Code landed:**
- `include/quantum/mutual_information.{hpp,cpp}` —
  `MutualInformation` utility class (2-site reduced density matrix,
  von Neumann entropy, all-pairs site-site MI).
- `include/quantum/choi_state.{hpp,cpp}` — `ChoiPropagator` utility
  class: interleaved doubled-chain SiteSet, Bell-chain initial MPS,
  output-register-only Schwinger MPO, TDVP evolution, all-pairs
  temporal MI extraction.
- `include/quantum/holography.{hpp,cpp}` — `HolographyConfig`,
  `MutualInformationProfile` (consumes both spatial and temporal MI),
  `EmergentGraph`, `AmbjornLollFit`, `SpectralDimensionResult`,
  `EmergentSpectralDimension`.
- `tessera/quantum/holography/__init__.py` — Python re-export shim
  (everything is a thin wrapper around the C++ classes).
- Acceptance tests (one per spec phase):
  * `test_mutual_information_python.py` — §H1: von Neumann entropy on
    hand-built density matrices, edgeLength cutoff.
  * `test_holography_python.py` — §H3+H4: HolographyConfig validation,
    profile / graph / Laplacian invariants, Ambjorn-Loll fit recovery,
    m/g sensitivity.
  * `test_choi_state_python.py` — §H2 #1/#2 at machine precision;
    edge-count and peak-D_S responses when temporal MI flips on.
  * `test_holography_acceptance_python.py` — §H1 #1/#2/#3, §H2 #3
    (heavy-quark off-diagonal suppression as a stand-in for the
    single-site-unitary idealisation).
  * `test_spectral_dimension_known_graphs_python.py` — §H4 #1/#2/#3
    (1D chain D_S→1, 2D lattice D_S→2, complete graph D_S→0).
  * `test_holography_causal_compare_consistency_python.py` — §8 #4
    cross-check regression.
- `examples/quantum/run_emergent_spectral_dimension.py` — m/g scan
  driver script with hypothesis-falsification checks. Writes JSON
  records matching the §10 schema when `--out-json-dir` is set.
- `examples/quantum/run_holography_convergence.py` — §H5 convergence
  sweep over (χ, K, ε_I, max_temporal_stride).

This document is the **scientific charter + implementation plan** for
one new observable: the spectral dimension of the graph whose vertices
are (lattice-site, snapshot-time) pairs and whose edge lengths are
derived uniformly from mutual information on a Schwinger TDVP state.
It lives alongside the existing causal-order comparison; it does not
replace it.

Beautiful is better than ugly. Simple is better than complex. The graph
is the boundary state's mutual-information structure made flat; the
spectral dimension is its random-walk return probability. Both are one
line of math each; the work is in plumbing them through what already
exists.

---

## 1. Hypothesis

Let $|\psi(t)\rangle$ be the TDVP-evolved Schwinger state on $N$ staggered sites at sampled times $t_0, t_1, \dots, t_K$. Define the (site, time) label set $\mathcal{L} = [N] \times \{t_0, \dots, t_K\}$ and a weighted graph $G$ on $\mathcal{L}$ with edge weights derived uniformly from mutual information:

$$\ell((i, s), (j, t)) = -\log I(i_s : j_t),$$

with $I(i_s : j_t)$ the spatial (Sub)-additive mutual information when $s = t$, and the Choi-state mutual information of the TDVP propagator $U_{s \to t}$ when $s \neq t$.

Let $D_S^{(G)}(\sigma) = -2\, d\log P_G(\sigma) / d\log\sigma$ be the spectral dimension of $G$ derived from the heat-kernel return probability of a continuous-time random walk on $G$.

**Hypothesis (H_SD).** For physical Schwinger states arising from the q-qbar quench, $D_S^{(G)}(\sigma)$ exhibits a $\sigma$-dependent profile that:

1. Approaches the "lattice dimension" $D_S \to 2$ at short diffusion times (the chain is 1+1D and locally regular), and
2. Approaches a "small-world" value $D_S < 2$ at long diffusion times when long-range mutual-information edges dominate.

**Falsification criteria.**

1. *Strong falsification.* $D_S^{(G)}(\sigma)$ is non-monotonic outside the small-$\sigma$ lattice-artefact regime, or is independent of $m/g$ (i.e. independent of the underlying physics).
2. *Trivial confirmation.* $D_S^{(G)}(\sigma) \equiv 2$ for all $\sigma$ would mean the long-range mutual-information edges contribute nothing; the test must show $\sigma$-sensitivity before its results carry weight.

The hypothesis is descriptive, not predictive of a number; we are not claiming a particular asymptotic dimension.

## 2. Limitations and scope conditions

These bound the interpretive weight of any result and mirror `quantum-methodology` §2:

- *Pure-state, contiguous-cut regime for I*. All mutual information is between site-singletons or contiguous intervals on a pure global state. Non-contiguous regions are not in scope.
- *Closed unitary dynamics.* The Schwinger TDVP propagator is unitary, so the temporal Choi state is pure and $I(i_s : j_t)$ has the entanglement-entropy-of-the-purification interpretation. Open-system extension is out of scope here.
- *Truncated electric basis and finite bond dimension.* As in the q-qbar quench acceptance: convergence in $\Lambda$ (electric cutoff) and $\chi$ (bond dimension) must be characterized per run.
- *Spectral dimension is finite-size dependent.* As `examples/spectral_dimension.py` shows for CDT, $D_S$ at small system size is offset from the asymptotic value. We report converged trends, not absolute numbers, at moderate $N$.
- *Spatial vs. temporal mutual information have different cost.* Spatial $I(i_s : j_s)$ is one MPS two-site reduced-state contraction. Temporal $I(i_s : j_t)$ requires the Choi state of the TDVP propagator over $t - s$, which is itself an MPS-MPO contraction. The implementation must amortize this carefully.

## 3. System and notation

### 3.1 Inputs

A converged TDVP run is the only required input. We reuse exactly the existing `TDVPConfig` and `runQqbarQuench` from `tessera.quantum`. No new Hamiltonian, no new evolution code.

### 3.2 Mutual-information observables

The construction needs two new pure functions on top of the existing TDVP snapshots:

- $I_{\text{spatial}}(\rho, A, B) = S_{\text{vN}}(\rho_A) + S_{\text{vN}}(\rho_B) - S_{\text{vN}}(\rho_{AB})$, with $A, B$ disjoint contiguous intervals.
- $I_{\text{temporal}}(U_{s \to t}, i, j)$ = mutual information between site $i$ on the input register and site $j$ on the output register of the Choi state of $U_{s \to t}$.

Both are defined on existing tessera objects; neither requires new state.

### 3.3 Emergent graph $G$

$$V_G = [N] \times \{t_0, \dots, t_K\}, \quad E_G = \{((i,s), (j,t)) : I((i,s) : (j,t)) > \varepsilon_I\}, \quad \ell_G = -\log I.$$

The graph is sparse-by-cutoff: edges with $I$ below numerical-floor $\varepsilon_I$ (default $10^{-10}$) are dropped. The graph fits comfortably as a Python object via `scipy.sparse`.

### 3.4 Spectral dimension on $G$

$$P_G(\sigma) = \frac{1}{|V_G|} \mathrm{Tr}\, e^{-\sigma L_G}, \quad L_G = D_G - W_G, \quad (W_G)_{vw} = I_{vw}, \quad (D_G)_{vv} = \sum_w I_{vw},$$

$$D_S^{(G)}(\sigma) = -2 \frac{d \log P_G}{d \log \sigma}.$$

Note: the *weighted Laplacian* uses $W_{vw} = I_{vw}$ (mutual information as edge capacity), not $W_{vw} = 1/\ell_{vw}$. This is the convention consistent with $\ell = -\log I$ and gives the random walk transition rates proportional to mutual information directly.

---

## 4. API design

The submodule sits at `tessera.quantum.holography`, mirroring the layout and conventions of `tessera.quantum`. Three principles, taken from the existing code:

- **Pure predicates and pure compute as module-level functions.** No state.
- **Configuration as a class.** Validated at construction.
- **Pipeline as one function.** Returns a single result dataclass.

### 4.1 Module surface

```python
from tessera.quantum.holography import (
    # Pure functions
    mutualInformation,                # I(A : B) on a density matrix
    temporalMutualInformation,        # I(i_s : j_t) on a Choi state
    edgeLength,                       # -log I, with cutoff handling
    returnProbability,                # heat-kernel trace on a weighted graph
    spectralDimension,                # -2 d log P / d log σ via finite differences
    fitAmbjornLollProfile,            # D_S(σ) = D_∞ - C/(B + σ); returns (D_∞, C, B, χ²)

    # Configuration
    HolographyConfig,                 # extends TDVPConfig with σ-grid, ε_I, etc.

    # Classes (state holders)
    MutualInformationProfile,         # full I(v, w) on V_G, supports __getitem__
    EmergentGraph,                    # sparse adjacency, Laplacian, exports
    SpectralDimensionResult,          # P(σ), D_S(σ), fit parameters

    # Pipelines
    computeMutualInformationProfile,  # TDVPConfig -> MutualInformationProfile
    buildEmergentGraph,               # MutualInformationProfile -> EmergentGraph
    computeEmergentSpectralDimension, # HolographyConfig -> SpectralDimensionResult
)
```

### 4.2 `HolographyConfig`

Composed from `TDVPConfig` to keep configuration explicit and flat. Errors are surfaced at construction; no late-bound failures.

```python
from dataclasses import dataclass, field
from tessera.quantum import TDVPConfig

@dataclass
class HolographyConfig:
    """Configuration for emergent-spectral-dimension runs.

    Wraps a TDVPConfig and adds the σ-grid and mutual-information cutoff.
    All fields are validated in __post_init__; invalid configs raise
    ValueError before any C++ call is made.
    """
    tdvp: TDVPConfig

    # σ-grid for D_S(σ); logarithmically spaced
    sigma_min: float = 1e-2
    sigma_max: float = 1e3
    sigma_count: int = 50

    # Mutual-information cutoff (edges with I < epsilon_I are dropped)
    epsilon_I: float = 1e-10

    # Whether to compute temporal MI (expensive: requires Choi states)
    include_temporal: bool = True

    # Cap on temporal stride |t - s| to limit Choi-state cost; None = unlimited
    max_temporal_stride: int | None = None

    # Random seed for any stochastic steps (currently unused; reserved for
    # future random-walk Monte Carlo estimator of P(σ))
    seed: int = 0

    def __post_init__(self) -> None:
        if self.sigma_min <= 0 or self.sigma_max <= self.sigma_min:
            raise ValueError("require 0 < sigma_min < sigma_max")
        if self.sigma_count < 8:
            raise ValueError("sigma_count must be >= 8 for finite-difference D_S")
        if self.epsilon_I < 0:
            raise ValueError("epsilon_I must be non-negative")
        if self.max_temporal_stride is not None and self.max_temporal_stride < 1:
            raise ValueError("max_temporal_stride must be >= 1 if set")
        # tdvp.recordSpectra is auto-forced True downstream; warn if the user
        # explicitly set it False.
```

### 4.3 Pure functions

Pure-function style mirrors the existing `majorizes`, `strictlyMajorizes`, and `majorizationPoset`. Each takes plain arrays and returns plain arrays or scalars.

```python
def mutualInformation(rho: np.ndarray,
                      qubits_A: Sequence[int],
                      qubits_B: Sequence[int]) -> float:
    """I(A : B) = S(ρ_A) + S(ρ_B) - S(ρ_AB), in nats.

    rho is a 2^n × 2^n density matrix (n = total qubit count). A and B are
    disjoint subsets of qubit indices. Returns 0 with a warning if A or B
    is empty; raises ValueError if A and B overlap.

    Pure function; no global state.
    """
    ...

def temporalMutualInformation(choi: np.ndarray,
                              n_qubits: int,
                              qubit_in: int,
                              qubit_out: int) -> float:
    """I(i_in : j_out) from a Choi state.

    choi is the (2^n × 2^n) ⊗ (2^n × 2^n) Choi matrix of a unitary or CPTP map
    on n qubits. Returns the mutual information between the i-th input qubit
    factor and the j-th output qubit factor.

    Pure function; reduces to spatial mutualInformation when the channel is
    the identity (Choi = |Φ+⟩⟨Φ+|), as verified in test_holography.py.
    """
    ...

def edgeLength(I: float, epsilon: float = 1e-10) -> float:
    """ℓ = -log I, with floor at -log(epsilon).

    Returns float('inf') when I < epsilon. Pure function.
    """
    ...

def returnProbability(L: scipy.sparse.csr_matrix,
                      sigmas: np.ndarray) -> np.ndarray:
    """P(σ) = (1/|V|) Tr exp(-σ L) for a weighted graph Laplacian.

    Uses scipy.sparse.linalg.expm_multiply for memory efficiency. Returns an
    array of the same length as sigmas. Pure function.
    """
    ...

def spectralDimension(sigmas: np.ndarray, P: np.ndarray) -> np.ndarray:
    """D_S(σ) = -2 d log P / d log σ via centered finite differences.

    Endpoints use one-sided differences. Pure function.
    """
    ...

def fitAmbjornLollProfile(
    sigmas: np.ndarray,
    D_S: np.ndarray,
    sigma_fit_min: float | None = None,
    sigma_fit_max: float | None = None,
) -> tuple[float, float, float, float]:
    """Fit D_S(σ) = D_∞ - C / (B + σ) over the chosen σ range.

    Returns (D_∞, C, B, reduced_chi_squared). This is the same three-parameter
    form used by examples/spectral_dimension.py for CDT. Pure function.
    """
    ...
```

### 4.4 State-holding classes

```python
class MutualInformationProfile:
    """Per-snapshot mutual information on the (site, time) label set.

    Built once from a TDVP run. Supports both spatial and temporal queries
    via a single __call__: profile(v, w) for v, w ∈ V_G.

    Internal storage is a sparse dict-of-dicts keyed by (site, snapshot_idx);
    spatial entries are populated densely up to N^2 per snapshot, temporal
    entries up to (K × N)^2 but cut by max_temporal_stride.
    """

    def __init__(self,
                 snapshots: Sequence[TDVPSnapshot],
                 epsilon_I: float = 1e-10,
                 max_temporal_stride: int | None = None) -> None:
        ...

    def __call__(self, v: tuple[int, int], w: tuple[int, int]) -> float:
        """Mutual information between labels v = (site_i, snap_s) and w."""
        ...

    @property
    def nLabels(self) -> int: ...
    @property
    def labels(self) -> list[tuple[int, int]]: ...

    def asWeightedAdjacency(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """COO arrays (rows, cols, weights, N) matching the convention of
        tessera.Spacetime.getDualAdjacency. Edges with I < epsilon are absent."""
        ...


class EmergentGraph:
    """The (V_G, E_G, ℓ_G) graph derived from a MutualInformationProfile."""

    def __init__(self, profile: MutualInformationProfile) -> None:
        ...

    @property
    def nVertices(self) -> int: ...
    @property
    def nEdges(self) -> int: ...

    def laplacian(self) -> scipy.sparse.csr_matrix:
        """Weighted Laplacian L = D - W with W_{vw} = I(v,w)."""
        ...

    def shortestPaths(self, source: tuple[int, int]) -> dict[tuple[int, int], float]:
        """Dijkstra distances under ℓ_G. Diagnostic; not used by spectral dim."""
        ...

    def toDot(self) -> str:
        """Graphviz DOT export, mirrors Poset.toDot()."""
        ...

    def toGraphML(self, path: str) -> None:
        """GraphML export for Gephi / yEd, mirrors Spacetime.save('*.graphml')."""
        ...


@dataclass(frozen=True)
class SpectralDimensionResult:
    """Result of computeEmergentSpectralDimension.

    Mirrors the snapshot-style results in tessera.quantum: plain dataclass,
    frozen, all arrays NumPy. No MPS/MPO state crosses the boundary.
    """
    sigmas: np.ndarray            # (sigma_count,) log-spaced
    P: np.ndarray                 # P(σ); same shape
    D_S: np.ndarray               # D_S(σ); same shape
    D_infinity: float             # Ambjorn–Loll fit asymptote
    C: float
    B: float
    fit_chi_squared: float
    graph_n_vertices: int
    graph_n_edges: int
    tdvp_summary: dict            # snapshot times, bond dims, energies; cheap
```

### 4.5 The pipeline

```python
def computeEmergentSpectralDimension(config: HolographyConfig) -> SpectralDimensionResult:
    """Run TDVP → mutual info → graph → D_S(σ) in one call.

    The function is purely sequential and has no hidden state. Each stage
    is also available as a standalone callable for users who want to
    introspect intermediate objects (see Quickstart in §6).

    Steps:
      1. Force config.tdvp.recordSpectra = True; warn if user set it False.
      2. result = runQqbarQuench(config.tdvp)
      3. profile = computeMutualInformationProfile(result.snapshots, config)
      4. graph = buildEmergentGraph(profile)
      5. sigmas = np.logspace(log10(config.sigma_min), log10(config.sigma_max),
                              config.sigma_count)
      6. P = returnProbability(graph.laplacian(), sigmas)
      7. D_S = spectralDimension(sigmas, P)
      8. (D_inf, C, B, chi2) = fitAmbjornLollProfile(sigmas, D_S)
      9. return SpectralDimensionResult(...)

    All numerical parameters and random seeds are stored in tdvp_summary
    for reproducibility, per quantum-methodology §5.
    """
    ...
```

---

## 5. Integration with existing tessera

Each integration point is a use of something that already exists in the codebase; the holography submodule introduces no replacements.

| Existing piece | How it is reused |
| --- | --- |
| `TDVPConfig`, `runQqbarQuench`, `TDVPSnapshot` | Sole source of the boundary state trajectory. `HolographyConfig` composes `TDVPConfig` rather than reinventing it. |
| `TDVPSnapshot.spectra` (the all-cuts Schmidt spectra) | Source of $S_{\text{vN}}$ for spatial mutual information. No new SVDs needed for the spatial half. |
| `TDVPSnapshot.bondDim`, `TDVPSnapshot.energy` | Reproduced verbatim in `SpectralDimensionResult.tdvp_summary` for reproducibility. |
| `tessera.quantum.Poset` (`getNodeCount`, `covers`, `toDot`) | Pattern reuse for `EmergentGraph` API surface; `EmergentGraph.toDot()` matches `Poset.toDot()`. |
| `tessera.quantum.CausetChain` (`antichains`, `vertexIds`, `hoppingPairs`, `partialOrder`) | Pattern reuse for `MutualInformationProfile` (snapshot-indexed access via `(site, snap_idx)` tuples). Also: when a non-trivial `CausetChain` is the source rather than a regular lattice, the holography pipeline still works — only the label set $\mathcal{L}$ changes. |
| `tessera.quantum.computeCausalComparison` | Independent observable; we sit alongside it, not on top of it. Cross-tests should verify that on a given run, the two observables are computed from the same TDVP snapshots without disagreement on the underlying spectra. |
| `examples/spectral_dimension.py` | The CDT script. We add a sibling `examples/quantum/run_emergent_spectral_dimension.py` that calls our pipeline. The shared math (return probability, finite-difference $D_S$, Ambjorn–Loll fit) is factored into `tessera.quantum.holography` and called from both — eliminating a duplicate implementation. |
| `tessera.Spacetime.getDualAdjacency()` | Convention reuse for `MutualInformationProfile.asWeightedAdjacency()` (COO format with `(rows, cols, weights, N)`). |
| ITensor v3 (C++ side) | Choi-state construction is added as a new C++ helper, `quantum/choi_state.{hpp,cpp}`, that takes the TDVP propagator MPO over an interval and constructs the Choi MPS. **The Choi MPS never crosses the language boundary** — only its two-site reduced density matrices do (small numpy arrays). |

The C++/Python contract from `quantum.md` is preserved: "scalar config in / scalar diagnostics out, no MPS or MPO objects cross the language barrier."

---

## 6. Quickstart

```python
import tessera
from tessera.quantum import TDVPConfig
from tessera.quantum.holography import (
    HolographyConfig,
    computeEmergentSpectralDimension,
)

tdvp = TDVPConfig()
tdvp.N = 14
tdvp.a = 1.0
tdvp.g = 1.0
tdvp.m = 0.5
tdvp.L0 = 0.0
tdvp.dmrgMaxBondDim = 64
tdvp.dmrgNSweeps = 12
tdvp.i0 = 5
tdvp.d = 5
tdvp.dt = 0.05
tdvp.T = 5.0
tdvp.maxBondDim = 100
tdvp.snapshotEvery = 5
tdvp.recordSpectra = True   # required by the pipeline; forced if False

cfg = HolographyConfig(
    tdvp=tdvp,
    sigma_min=1e-2,
    sigma_max=1e3,
    sigma_count=64,
    epsilon_I=1e-10,
    include_temporal=True,
    max_temporal_stride=4,
)

result = computeEmergentSpectralDimension(cfg)
print(f"|V_G|       = {result.graph_n_vertices}")
print(f"|E_G|       = {result.graph_n_edges}")
print(f"D_∞ fit    = {result.D_infinity:.3f}")
print(f"χ²/dof     = {result.fit_chi_squared:.3f}")

# Plot the profile
import matplotlib.pyplot as plt
plt.semilogx(result.sigmas, result.D_S, "o-")
plt.xlabel(r"$\sigma$"); plt.ylabel(r"$D_S(\sigma)$")
plt.axhline(result.D_infinity, ls="--", color="red", label="fit asymptote")
plt.legend(); plt.show()
```

Decomposed pipeline, for users who want intermediate objects:

```python
from tessera.quantum import runQqbarQuench
from tessera.quantum.holography import (
    computeMutualInformationProfile, buildEmergentGraph,
    returnProbability, spectralDimension, fitAmbjornLollProfile,
)
import numpy as np

quench = runQqbarQuench(cfg.tdvp)
profile = computeMutualInformationProfile(quench.snapshots, cfg)
graph = buildEmergentGraph(profile)
graph.toGraphML("/tmp/emergent.graphml")     # peek at it in Gephi

sigmas = np.logspace(np.log10(cfg.sigma_min), np.log10(cfg.sigma_max), cfg.sigma_count)
P = returnProbability(graph.laplacian(), sigmas)
D_S = spectralDimension(sigmas, P)
D_inf, C, B, chi2 = fitAmbjornLollProfile(sigmas, D_S)
```

---

## 7. Phase plan and acceptance criteria

Mirroring the phase structure of `quantum-plan.md`. Each phase has a numerical acceptance check; phases are merge-gated on those checks.

### Phase H0 — Scaffolding

- `tessera/quantum/holography/__init__.py` exporting the symbols in §4.1.
- `HolographyConfig` validated; doctests.
- **Acceptance:** `python -c "from tessera.quantum.holography import *"` works; invalid configs raise `ValueError` at construction.

### Phase H1 — Spatial mutual information

- `mutualInformation(rho, A, B)` on dense density matrices.
- C++ helper `quantum/reduced_density_matrix.hpp` that extracts arbitrary contiguous-interval reduced states from a `MPS` snapshot, exposed via a thin Python binding returning `np.ndarray`.
- **Acceptance:**
  - Bell pair $|\Phi^+\rangle$ between sites $i, j$ gives $I = 2\ln 2 = 1.386$ to 1e-12.
  - Two-qubit product state gives $I = 0$ to 1e-12.
  - On a converged Schwinger ground state, $I(\text{site}_i : \text{site}_{i+1})$ decreases monotonically with $|i - i'|$ along a 1D chain in the gapped regime.

### Phase H2 — Choi states and temporal mutual information

- C++ helper `quantum/choi_state.hpp`: takes a TDVP-built propagator MPO between snapshots $s$ and $t$ and constructs its Choi MPS.
- `temporalMutualInformation(choi, n_qubits, i_in, j_out)` reads two-site marginals from the Choi MPS (via the same `reduced_density_matrix` helper, doubled).
- **Acceptance:**
  - Identity channel: Choi = $|\Phi^+\rangle$, $I(i_\text{in} : i_\text{out}) = 2\ln 2$.
  - Identity channel: $I(i_\text{in} : j_\text{out}) = 0$ for $i \neq j$.
  - Single-qubit unitary on site $i$: $I(i_\text{in} : i_\text{out}) = 2\ln 2$, all others 0.
  - Energy / charge conservation check on the constructed Choi MPS (tracing out the input register must reproduce the depolarizing-output behaviour for a maximally mixed input, by Stinespring duality).

### Phase H3 — `MutualInformationProfile` and `EmergentGraph`

- Build `profile` from a sequence of `TDVPSnapshot`s.
- Build sparse-Laplacian `graph` from `profile`.
- DOT and GraphML exports.
- **Acceptance:**
  - On a heavy-quark quench with $m/g = 20$ (a near-frozen evolution): $|E_G|$ is small (only nearest-neighbour spatial + same-site temporal edges survive), and `graph.laplacian()` is block-diagonal across time.
  - On a light-quark quench with $m/g = 0.5$ (string-breaking dynamics): $|E_G|$ grows significantly with the snapshot count, reflecting entanglement spread.
  - DOT and GraphML files round-trip through Graphviz and Gephi without errors.

### Phase H4 — Spectral dimension

- `returnProbability`, `spectralDimension`, `fitAmbjornLollProfile` as pure functions.
- `computeEmergentSpectralDimension` pipeline.
- **Acceptance:**
  - 1D chain graph ($N$ vertices, nearest-neighbour edges, unit weights): $D_S(\sigma) \to 1$ at large $\sigma$, to within 0.1.
  - 2D square lattice graph: $D_S(\sigma) \to 2$, to within 0.1.
  - Complete graph $K_N$: $D_S(\sigma) \to 0$ at $\sigma \gg \log N$ (small-world saturation).
  - On a Schwinger heavy-quark quench: $D_S \to 1$ asymptote (graph is approximately a 1D chain).
  - On a Schwinger light-quark quench: $D_S$ profile is sensitive to $m/g$; trivial confirmation (i.e. $D_S \equiv 2$) is rejected.

### Phase H5 — Example script and docs

- `examples/quantum/run_emergent_spectral_dimension.py` running the pipeline at three values of $m/g$ and writing a comparison figure.
- New doc page `docs/source/holography.md` linked from `quantum.md` as a sibling of Phases 3–6.
- Convergence sweeps: $D_S(\sigma)$ vs. bond dimension $\chi$, electric cutoff $\Lambda$, snapshot count $K$, $\varepsilon_I$.
- **Acceptance:** documented convergence within ±0.1 in $D_\infty$ over a doubling of each control parameter; reproducibility from the recorded `tdvp_summary` and the deterministic seed.

### Phase H6 — `CausetChain` integration (optional, after H5)

- Allow `HolographyConfig.tdvp` to be sourced from `extractCausetChain(spacetime)` rather than a regular lattice.
- `MutualInformationProfile` accepts the chain's `(antichains, vertexIds, hoppingPairs)` to label vertices by (spacetime-vertex-id, snapshot).
- **Acceptance:** on a trivial chain $\mathrm{CausetChain}$ the result agrees numerically with the regular-lattice path; on a branching causet it does not.

---

## 8. Threats to validity

Beyond the standard `quantum-methodology` §4.5 list:

- *Edge-cutoff sensitivity.* $D_S(\sigma)$ depends on $\varepsilon_I$ because the graph topology changes when $\varepsilon_I$ crosses a mutual-info value. The convergence sweep in Phase H5 must check this explicitly; report $D_\infty$ as a function of $\log_{10}\varepsilon_I$.

- *Choi-state bond-dimension blow-up.* The propagator MPO between distant snapshots has bond dimension up to $\chi^2$, and its Choi state can be expensive. `max_temporal_stride` is the operational lever; results must converge in this limit.

- *Finite-difference noise on $D_S(\sigma)$.* Central differences on noisy $\log P(\sigma)$ produce oscillation. Smoothing via local polynomial fit (Savitzky–Golay, window 5) before differentiation is acceptable; report both raw and smoothed $D_S$.

- *Comparison with the causal-order comparison.* The same TDVP snapshots feed both the existing causal-order comparison and our spectral dimension. The two observables should agree on the underlying spectra (same Schmidt values). A regression test cross-checks both pipelines run from a single shared `SchwingerQuench(cfg).evolve()` call without divergence in $E$, bond dim, or $\langle L_n\rangle$.

---

## 9. Tested benchmarks

Following the test table format of `quantum.md`:

| Test | Layer | What it verifies |
| --- | --- | --- |
| `test_mutual_information.py` | H1 | Bell/GHZ/product state limits; symmetry $I(A:B) = I(B:A)$; non-negativity. |
| `test_reduced_density_matrix.cpp` | H1 | MPS-side two-site reduced state vs. dense ED on Schwinger ground state to $10^{-10}$. |
| `test_choi_state.cpp` | H2 | Identity and Pauli-rotation channels: Choi state matches analytic forms. Energy/charge conservation traced through. |
| `test_temporal_mutual_information.py` | H2 | Identity channel temporal MI = $2\ln 2$ at same site, $0$ at distinct sites. Depolarizing channel temporal MI = $0$. |
| `test_emergent_graph.py` | H3 | Heavy- vs. light-quark edge counts; DOT and GraphML round-trips; Laplacian self-adjointness ($L = L^T$). |
| `test_spectral_dimension_known_graphs.py` | H4 | 1D chain, 2D lattice, complete-graph $D_S$ targets within 0.1. |
| `test_spectral_dimension_schwinger.py` | H4 | $m/g$-sensitivity asserted; trivial-confirmation rejection ($D_S \not\equiv 2$). |
| `test_holography_causal_compare_consistency.py` | H4 | Joint run with the causal-order comparison: same $E$, $\langle L_n\rangle$, spectra to $10^{-12}$. |
| `test_holography_convergence.py` | H5 | $D_\infty$ stable within ±0.1 under doubling of $\chi$, $\Lambda$, $K$. |
| `test_holography_causetchain.py` | H6 | Regular lattice equivalent to a trivial `CausetChain`; branching causet differs. |

---

## 10. Deliverables and reproducibility

Each run writes a single JSON record (matching the recorded-config convention in `quantum-methodology` §5):

```json
{
  "config": { ... HolographyConfig as dict ... },
  "tdvp_summary": { "snapshot_times": [...], "bond_dims": [...], "energies": [...] },
  "graph": { "n_vertices": ..., "n_edges": ..., "edge_density": ... },
  "spectral_dimension": {
    "sigmas": [...], "P": [...], "D_S": [...],
    "fit": { "D_infinity": ..., "C": ..., "B": ..., "chi_squared": ... }
  },
  "provenance": { "tessera_version": "...", "git_sha": "...", "seed": ... }
}
```

The JSON is sufficient to regenerate every figure; the same `HolographyConfig` re-run produces bit-identical numerics modulo BLAS nondeterminism.

---

## 11. What this is not

- *Not* a new Hamiltonian. The Schwinger model from `quantum.md` is the only physical system.
- *Not* a new evolution scheme. The existing TDVP integrator is the only propagator.
- *Not* a claim about quantum gravity. The construction is a mutual-information graph of a 1+1D gauge theory; agreement of $D_S$ with any continuum-gravity number (CDT, asymptotic safety) at this $N$ would be coincidence.
- *Not* a replacement for the causal-order comparison. The majorization-order comparison and the spectral-dimension test answer different questions on the same data; both should be reported.

---

## References

- Ambjorn, Jurkiewicz, Loll, *Reconstructing the Universe*, [hep-th/0505154](https://arxiv.org/abs/hep-th/0505154) — spectral-dimension definition and CDT result used as the methodological template.
- Van Raamsdonk, *Building up spacetime with quantum entanglement*, [1005.3035](https://arxiv.org/abs/1005.3035) — the $d \propto -\log I$ relation between bulk distance and boundary mutual information.
- Pollock, Rodríguez-Rosario, Frenzel, Modi, Modi, *Operational Markov condition for quantum processes*, [1801.09811](https://arxiv.org/abs/1801.09811) — process tensor / Choi state for multi-time correlations.
- Nielsen, *Conditions for a class of entanglement transformations*, [quant-ph/9811053](https://arxiv.org/abs/quant-ph/9811053) — used by the causal-order comparison; mentioned here for the joint regression test.
- Haegeman, Lubich, Oseledets, Vandereycken, Verstraete, *Unifying time evolution and optimization with matrix product states*, [1408.5056](https://arxiv.org/abs/1408.5056) — the TDVP integrator we evolve under.
- Bañuls, Cichy, Cirac, Jansen, *The mass spectrum of the Schwinger model with MPS*, [1305.3765](https://arxiv.org/abs/1305.3765) — the Schwinger conventions.
- Cao, Carroll, Michalakis, *Space from Hilbert space*, [1606.08444](https://arxiv.org/abs/1606.08444) — graph distance from mutual information, the same construction at a different scale.
