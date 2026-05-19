---
orphan: true
---

# Interaction-history Monte Carlo: emergent spacetime from mutual information

This is the scientific charter for a Metropolis Monte Carlo that samples
*interaction histories* of a set of quantum systems, weighted by the
geometric Regge action on the simplicial complex those interactions
build. Edge lengths come from mutual information, `d = -log I`. The
object of the search is the coupling at which the emergent heat-kernel
spectral dimension reaches **4** — the 3+1-dimensional phase.

It sits alongside the two existing observables — the causal-order
comparison and the emergent spectral dimension — and reuses the tessera
simplicial machinery (`Spacetime`, `Simplex`, `ReggeSolver`) the way
`CDTSimulation` does. The implementation surface is one C++ class,
`InteractionSimulation`, shaped like `CDTSimulation`.

---

## 1. The picture

Quantum systems interact pairwise. Each interaction is an *event*: two
systems `A`, `B` interact and the event spawns a third worldline `AB`,
the interaction product, leaving `A'` and `B'` carried forward. The five
systems `{A, B, A', AB, B'}` are the vertices of a `(2,3)` 4-simplex —
two on the earlier time slice, three on the later one — and the history
of accepted interactions is a simplicial complex.

The mutual information between systems is an edge length, `d = -log I`,
in the van Raamsdonk sense. The geometric Regge action `S = Σ_h A_h ε_h`
on the MI-lengthed complex weights the ensemble of interaction
histories. Which interactions actually occur is **sampled** from that
ensemble by Metropolis–Hastings — the dynamics is not dictated, it is
drawn from the partition function.

**Hypothesis.** As the inverse-temperature coupling `β` (and the
interaction parameters) are varied, the emergent spectral dimension of
the interaction-history complex passes through a phase structure, and
there is a locus where `D_S → 4`. Finding that locus is the experiment.

---

## 2. The initial layer

`N` quantum systems, each prepared in a **known, randomized mixed
state** `ρ_i` with `S(ρ_i) > 0`. The mixedness is essential: the
conservation law in §4 is trivial for a pure system (`S = 0`), so the
systems must carry genuine entropy.

The systems are Poisson-distributed in a 2D patch and Delaunay-
triangulated; the Delaunay edges are the `t = 0` spatial adjacency, and
the Delaunay triangulation is the Voronoi dual. This supersedes the
DMRG-ground-state initial layer of the earlier single-cell experiments
— here the layer is randomized, not a solved ground state.

---

## 3. The interaction event

When `A` and `B` interact through a two-system unitary `U`, the
interaction product `AB` is the **genuine joint state** of the two
systems:

$$\rho_{AB} = U\,(\rho_A \otimes \rho_B)\,U^\dagger.$$

`AB` is a genuine new node — a worldline created by the event — and its
quantum content is `ρ_AB`, a concrete object that simply exists. There
is no Choi isomorphism, no reference legs, no co-existence puzzle: `AB`,
its marginals, and the input states are all reduced-density-matrix
quantities on states that exist.

### 3.1 Factorizing the joint state

`ρ_AB` is factorized to extract the entanglement content. Any two-qubit
state can be written

$$\rho_{AB} = \tfrac14\Big(I + \sum_i a_i\,\sigma^A_i + \sum_j b_j\,\sigma^B_j + \sum_{ij} c_{ij}\,\sigma^A_i\sigma^B_j\Big),$$

and under local rotations the correlation matrix `c_ij` diagonalizes to
three invariants `\vec c = (c_1, c_2, c_3)` — the **Cartan coordinates
of the state**, the honest measure of how much the interaction coupled
the two systems. The Bloch vectors `a_i`, `b_j` are the local content
and peel off as `A'` and `B'` (the marginals carried forward). `\vec c`
is zero for a non-entangling interaction and grows with the coupling.

---

## 4. Edge bookkeeping

The `(2,3)` cell `{A, B, A', AB, B'}` has ten edges. They come from two
places — genuine mutual informations on co-existing systems, and a
conservation law for the temporal edges.

### 4.1 Genuine mutual informations

These are ordinary `I(X:Y) = S(X) + S(Y) - S(XY)` on systems that
co-exist in the one global state:

- `I(A:B)` — the input pair, before the interaction.
- `I(A':AB)`, `I(B':AB)`, `I(A':B')` — the output triple.
- `I(A:AB)`, `I(B:AB)` — the primary temporal quantities. `AB` is the
  joint state `ρ_AB`; `A'`, `B'` are its marginals. `I(A:AB)` is the
  genuine mutual information sitting in `ρ_AB` —
  `S(A') + S(B') − S(ρ_AB)` — how much the interaction correlated the
  two systems. It is an ordinary MI on one concrete object.

### 4.2 The conservation law

The remaining temporal edges close by conservation — information in =
information out — on the six-edge interaction structure
`A→A'`, `A→AB`, `B→B'`, `B→AB`:

$$S(A) = I(A{:}A') + I(A{:}AB), \qquad S(B) = I(B{:}B') + I(B{:}AB).$$

`I(A:AB)` and `I(B:AB)` are the *primary* quantities (genuine MIs,
§4.1); `I(A:A')` and `I(B:B')` are the **residuals**:

$$I(A{:}A') = S(A) - I(A{:}AB), \qquad I(B{:}B') = S(B) - I(B{:}B).$$

No co-existence of `A` with `A'` is ever required — every input is a
single-system entropy or a genuine MI on a state that exists. This is
what dissolves the no-cloning knot: there is no freeze, no process
tensor, no propagator snapshot.

### 4.3 Edge lengths

Every edge length is `d = -log I` (van Raamsdonk distance), normalised
so `d ≥ 0`, with an `ε_I` floor. Same-slice edges are spacelike,
cross-slice edges timelike, in the CDT sense.

---

## 5. The Regge action and the partition function

The geometric Regge action on the MI-lengthed complex,

$$S[C] = \sum_{h \in \text{hinges}} A_h\, \varepsilon_h,$$

with `A_h` the Heron hinge area and `ε_h = 2π - Σ θ` the deficit angle —
evaluated, not solved, through `ReggeSolver`'s `hingeArea` /
`deficitAngle` primitives. No worldline matter term: the matter is in
the MIs (the geometry is built from the entanglement), so a separate
`S_matter` would double-count.

The partition function is over interaction histories reachable from the
initial layer:

$$Z = \sum_{C} \frac{1}{C_C}\, e^{-\beta S[C]},$$

with `1/C_C` the symmetry factor. `β` is the inverse-temperature
coupling — varying `β` maps the phase structure, and the search is for
the `β` where `D_S → 4`.

---

## 6. The Monte Carlo

The equilibrium ensemble is sampled with two moves:

- **`interact{X,Y}`** — pick a uniformly-random eligible frontier
  spatial edge (`X`, `Y` both on the frontier — no out-edges), attach
  the `(2,3)` cell, spawn `AB`.
- **`unInteract`** — pick a uniformly-random *leaf* cell (all three
  products still on the frontier), remove it.

A system may interact only while it has no out-edges; `unInteract`
removes only leaf cells. So each system interacts at most once, the
moves are cleanly reversible, and `N₊` (frontier spatial edges) and
`N₋` (leaf cells) are well-defined incremental tables.

Metropolis–Hastings acceptance:

$$A(C \to C') = \min\!\left\{1,\; \frac{N_+}{N_-}\cdot\frac{C_C}{C_{C'}}\cdot e^{-\beta\,\Delta S}\right\}.$$

`ΔS` is local — the new cell's hinge contributions — read off a
per-hinge action table. The volume is controlled by capping the
interaction count (the `T`-cap). The lifecycle is **tune, then
thermalize**, the same order as `CDTSimulation`.

---

## 7. Implementation

One C++ class, `InteractionSimulation`, in `tessera::quantum`, shaped
like `CDTSimulation`: constructed with the couplings and the initial
layer, exposes the move primitives and `propose*` counterparts,
`sweep` / `thermalize` / `tune`, and `computeAction` /
`getAcceptanceRates` / the observable getters. The global quantum state
lives inside the class and never crosses the language boundary.

The global state is mixed (the initial systems are mixed), so it is
carried as a **purification** — an MPS on the system+ancilla doubled
lattice. Single-system entropies and the genuine MIs of §4.1 are
reduced-density-matrix computations on that MPS; the spike
`test_mps_site_insertion.cpp` has validated the site-insertion and
3-site-gate mechanics the interaction event needs.

| piece | status |
| --- | --- |
| `InteractionSimulation` scaffold (header) | done |
| `interact` / `unInteract` + frontier bookkeeping (simplicial side) | done |
| MPS site-insertion + 3-site-gate mechanics | spiked, validated |
| randomized mixed-state initial layer + purification | to build |
| KAK decomposition + Cartan-core Choi state | to build |
| conservation-law edge bookkeeping | to build |
| incremental Regge `ΔS` + Metropolis loop | to build |
| observables + Python bindings | to build |
| `D_S = 4` search experiment + writeup | to build |

---

## 8. Design decisions

The design is closed. The choices that took discussion to settle:

1. **`AB` is the genuine joint state** `ρ_AB`, a single node — not a
   Choi-isomorphism construct. It connects through the ordinary edges
   `A→AB`, `B→AB`, `A'–AB`, `B'–AB`; `A→A'`, `B→B'` are `A`'s and `B`'s
   other out-edges. No internal multi-leg structure.
2. **`I(A:AB)` is the genuine mutual information of `ρ_AB`** —
   `S(A') + S(B') − S(ρ_AB)` — and `I(A:A') = S(A) − I(A:AB)` is the
   residual. Every quantity is a reduced-density-matrix computation on
   a state that concretely exists; no freeze, no Choi, no co-existence
   puzzle.
3. **Lifecycle: tune, then thermalize** — the `CDTSimulation` order.

---

## References

- Van Raamsdonk, *Building up spacetime with quantum entanglement*,
  [1005.3035](https://arxiv.org/abs/1005.3035) — the `d ∝ -log I`
  relation.
- Kraus, Cirac, *Optimal creation of entanglement using a two-qubit
  gate*, [quant-ph/0011050](https://arxiv.org/abs/quant-ph/0011050) —
  the KAK / Cartan decomposition and entangling power.
- Choi, *Completely positive linear maps on complex matrices*, Linear
  Algebra Appl. 10 (1975); Jamiołkowski, Rep. Math. Phys. 3 (1972) —
  the operator↔state isomorphism for the Cartan core.
- Ambjorn, Jurkiewicz, Loll, *Reconstructing the Universe*,
  [hep-th/0505154](https://arxiv.org/abs/hep-th/0505154) — the Regge
  action and the Metropolis machinery `CDTSimulation` mirrors.
