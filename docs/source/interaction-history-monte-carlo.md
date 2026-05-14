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

When `A` and `B` interact, the interaction is a two-system unitary `U`.
Its **KAK (Cartan) decomposition** is exact:

$$U = (k_A \otimes k_B)\; e^{i(c_x XX + c_y YY + c_z ZZ)}\; (k'_A \otimes k'_B).$$

The single-qubit factors carry no entanglement; they rotate `A → A'` and
`B → B'`. The non-local content is the **Cartan core**
`e^{i(c_x XX + c_y YY + c_z ZZ)}`, and the Cartan coordinates
`\vec c = (c_x, c_y, c_z)` are the gate's entangling power.

### 3.1 The Cartan core as a state

The interaction product `AB` is the Cartan core, written as a state via
the Choi–Jamiołkowski isomorphism (operator on `H` ↔ state on
`H ⊗ H`). The core is diagonal in the Bell basis, so its Choi state is
clean:

$$|AB\rangle = \tfrac12 \sum_{\mu} e^{i\varphi_\mu}\, |B_\mu\rangle_{\text{out}}\, |B_\mu\rangle_{\text{in}},$$

with `|B_μ⟩ ∈ {|Φ⁺⟩, |Φ⁻⟩, |Ψ⁺⟩, |Ψ⁻⟩}` and the four phases the sign
combinations

$$\varphi_\mu \in \{\,c_x - c_y + c_z,\; -c_x + c_y + c_z,\; c_x + c_y - c_z,\; -c_x - c_y - c_z\,\}.$$

`|AB⟩` is a pure 4-leg state on `(out_A, out_B, in_A, in_B)`, fully
explicit in the computational basis. `AB` is a **genuine subsystem** of
the global state — a new worldline created by the event — and its four
legs are what the cell's edges to `A`, `B`, `A'`, `B'` attach to.

The entangling power shows up as the entanglement of `|AB⟩` across the
**A-side | B-side cut** `(out_A, in_A) | (out_B, in_B)`: for `\vec c = 0`
the core is the identity and `|AB⟩` factorizes across that cut (a local
interaction couples nothing); for `\vec c ≠ 0` it is entangled, by a
definite function of `\vec c`. That cut-entanglement is the honest
single-number measure of how much the interaction couples the two
systems.

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
- `I(A:AB)` — `ρ_A` linked into `|AB⟩`'s `in_A` leg; the mutual
  information of that linked state. This is "how much of `A` fed the
  core" — concretely the A-side entanglement of the core, weighted by
  `ρ_A`. Similarly `I(B:AB)` via the `in_B` leg.

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
interaction count (the `T`-cap). The lifecycle is **thermalize, then
tune**.

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

## 8. Open items to confirm

These are the implementation-level choices the design does not yet
fully pin:

1. **`AB`'s representation in the MPS.** `|AB⟩` is a 4-leg Choi state.
   Is `AB` a single node carrying a 4-qubit subsystem, or do its four
   legs (`in_A`, `out_A`, `in_B`, `out_B`) attach directly as the four
   edges to `A`, `A'`, `B`, `B'` (a link object with no independent
   qubit)? The conservation law in §4.2 needs whichever choice makes
   `S(AB)` and `I(·:AB)` consistently defined.
2. **Linking `ρ_A` into `|AB⟩`.** "`ρ_A` linked into the `in_A` leg" —
   the link-product mechanics (project the `in_A` leg onto `ρ_A`, or
   feed `ρ_A` as the input register) — needs to be pinned so `I(A:AB)`
   is unambiguous.
3. **Lifecycle order.** "Thermalize, then tune" — confirm this is the
   intended order (it is the reverse of the `CDTSimulation` lifecycle,
   which tunes the volume first).

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
