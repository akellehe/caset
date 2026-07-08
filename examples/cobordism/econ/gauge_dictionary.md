# The gauge dictionary — theory note for the economic register

> Part of the exploratory spike tessera#602. Every construct below is
> implemented in `econ_register.py` / `leak_experiment.py`; the numbers
> quoted are from the 2017 national build unless noted. Nothing here
> asserts that the economy is a quantum system; Stages 0–1 use only
> real-valued Hodge theory, and the discipline of the cobordism paper
> (correspondence on finite complexes, no continuum claims) applies.

## 1. The complex and the flow cochain

Vertices are the 71 BEA summary industries plus four closure sectors —
households (HH), the capital account (CAP), government (GOV), and rest of
world (ROW). An oriented edge carries the **net bilateral money flow** in
the direction the dollars move (buyer pays seller); pairwise offsetting
(the removal of 2-cycles) is applied at construction, since mutual
obligations that cancel are bookkeeping, not economics. A triangle is
**filled** when its three net flows form a directed 3-cycle whose minimum
flow exceeds the netting threshold τ: around such a triad a bookkeeper
can cancel a circulation with no economic substance. The observed flows
form a real 1-cochain `f` on this complex.

## 2. The gauge group is additive ℝ, not compact U(1)

A lattice gauge field assigns a group element to each oriented edge. Here
the group is the additive reals: flows compose by addition along paths,
and reversal negates. The compact circle group U(1) is deliberately
excluded at this stage: it would identify circulations mod 2π and alias
large circuits into small ones. The noncompact additive theory is exactly
weighted Hodge theory — every statement below is standard linear algebra,
used without approximation. (tessera's `HERMITIAN_WEIGHTED` U(1)-phase
machinery becomes relevant only if Stage 2 reaches transition
amplitudes.)

## 3. Gauge freedom is bookkeeping

Two re-descriptions of the books change the cochain without changing the
economics:

1. **Vertex re-potentialing** (gradient shifts, `f → f + d₀φ`): re-basing
   the hierarchical component — the Helmholtz "potential" of
   Kichikawa–Iyetomi hierarchy analyses. The Gauss law pins it from the
   margins; it is not discarded, it is *determined*.
2. **Netting on filled faces** (`f → f + ∂₂ψ`): subtracting a raw
   circulation around any filled triangle.

The invariant content under both is the **harmonic register**: the
quotient of conserved flows by netting moves. Its dimension is the first
Betti number b₁ of the complex (2017 national build: b₁ = 134 at τ = 0;
2005: b₁ = 77). The **periods** — signed loop sums around the independent
unfilled cycles — are the Wilson loops of this abelian theory, and they
are the coordinates in which the paper's register theorems operate.

## 4. The Gauss law is the flow-of-funds identity

The divergence of the money cochain at a vertex is inflow minus outflow —
**net accumulation**. Charge is therefore **net lending position**:

- Industry vertices carry zero charge by the accounting identities
  (verified on the data: T018 = T005 + VABAS and the supply-side
  identities hold to ≈2×10⁻⁶ relative; the assembled network closes with
  max divergence 1.3×10⁻⁷ of total flow).
- Closure sectors carry the sectoral balances as explicit net-lending
  edges into CAP (2017: HH −2.86 T$, GOV −2.09 T$, ROW +0.54 T$ — the
  last is the current-account deficit, a sanity anchor).
- Global charge is zero up to the statistical discrepancy (−4.8 M$ on a
  59 T$ economy).

"Every purchase is a sale" is the lattice Gauss law `∂₁f = 0`; "net
financial claims sum to zero" is total charge zero on the closed complex.
Note that conservation is **metric-free**: it constrains raw dollar
flows, independent of any weighting choice.

## 5. Energy, and the value→metric knob

The energy of a flow configuration is the weighted quadratic form

    E[f] = Σ_edges R_e f_e²   (+ the face terms killed by netting)

— the abelian Kogut–Susskind form: an electric term on links, a magnetic
term on plaquettes. Harmonic flows are its zero modes among conserved
configurations. The diagonal metric `R` is the **single free function of
the model**, the value→metric map:

- `conductance` — R = 1/gross flow: thick trading relationships are easy
  directions; the energy of a perturbation is Σ δ²/w, a χ² statistic.
- `length` — R = gross flow: the length ∝ value convention.
- `unit` — R = 1.

The metric never changes b₁ or which flows are conserved; it decides
orthogonality — which part of a change counts as harmonic — and therefore
every projected number. It is treated as a scanned knob, not an
assumption. Numerical note: R spans many orders of magnitude on real
tables; the harmonic solver row-normalizes its constraint matrix (a
null-space-preserving operation) to keep rank detection honest.

## 6. The IPF null is minimum-energy relaxation

Iterative proportional fitting of the earlier year's network to the later
year's margins is the minimum-Kullback–Leibler completion of the pinned
Gauss-law data — and min-KL agrees with the min-χ² (minimum electric
energy, conductance metric) completion to second order. The null model of
the experiment is therefore not an arbitrary baseline: it is *the ground
state of the model's own action* given the observed sizes. The **period
leak** — the R-norm of the harmonic component of the observed change —
is by construction the part of the transition that no energy relaxation
on the fixed geometry can absorb.

## 7. The held-fixed protocol (the tessera pattern, verbatim)

The paper's machine pins boundary data, relaxes the interior under an
action, and reads a stuck positive residual as a topological obstruction.
The economic transplant, implemented in `leak_experiment.py`:

- **Pinned geometry**: the year-t complex (edges, fills, metric, harmonic
  basis) is frozen.
- **Tested structure**: the later flows (year t+1, an IPF null, a planted
  break, or the MRIO-implied structure) expressed on that geometry.
- **Leak**: harmonic component of the change — the certified obstruction.
- **Off-complex mass**: flow on vertex pairs absent from the fixed edge
  set — demand for topology change the geometry cannot carry at all
  (≈0 at the 71-industry grain, where the graph is complete; expected to
  be the dominant signal at firm grain).

## 8. What the control pair established (Gate 2)

On the frozen 2005 geometry:

- A **margin-preserving re-sourcing rewire** (half of every buyer's
  purchases from credit intermediation re-sourced, then rebalanced) moves
  the flow cochain by 0.84 of its R-norm yet leaks 1.2×10⁻⁸: the
  certificate is silent on changes the geometry can carry. This is the
  machinery agreeing with Leontief economics, not failing.
- An **injected irreducible circulation** of size ε = 0.05 (a harmonic
  cochain: divergence-free, so margins cannot see it; orthogonal to every
  netting move) is measured as 0.05000, the IPF null absorbs none of it,
  and per-sector attribution recovers the injected mode's sectors 5/5.

So at the industry grain, "certified structural break" means precisely:
**the economy's irreducible circulation pattern moved more than its size
recomposition explains** — and the statistic detecting it is calibrated,
exact on positive controls, and silent on negative ones.

## 9. Higher-degree structure lives in the bulk, not the boundary

A sharp combinatorial fact constrains any attempt to run L₂/L₃ analysis
directly on the economic 1-complex: after pairwise netting, the net
flows orient the complete graph into a **tournament**, and a 4-vertex
tournament contains at most two cyclic triangles (out-degree counting:
the cyclic-triangle count is C(4,3) − Σᵢ C(dᵢ,2) with Σdᵢ = 6, so at
least two of the four faces are transitive). Since our 2-cells are
directed 3-cycles, **no tetrahedron can ever have all four faces
filled**: the boundary complex admits no 3-cells, verified empirically
(zero tetrahedra in every year at every threshold, as the theorem
requires). Consequently L₂ on the boundary degenerates to the co-closed
condition alone and L₃ has empty domain.

The higher Laplacians therefore live where the cobordism paper puts them
— in the **bulk**: the triangulated 3-/4-dimensional geometry
interpolating between year-boundary states, where tessera's builds
(Proton, ProtonIngredients, MultiCobordism) create genuine 3- and
4-cells. Interrogating the built bulk with HodgeLaplacian at k = 2, 3 is
the correct transplant of "use L₂ and L₃."

## 10. Correspondence table (paper ↔ economy)

| cobordism paper | economic register |
|---|---|
| carrier complex of a state | year-t money-flow complex |
| harmonic flows, ker L₁ | irreducible circuits of money |
| periods, conservation p_a+p_b+p_c = 0 | Wilson loops; sectoral balances / global charge zero |
| pinned boundary + relaxed interior | pinned margins + IPF (minimum-energy) relaxation |
| residual floor | period leak in excess of the IPF null |
| surgery with pinned boundary | severed/created trading relationships, changed netting structure |
| Gram matrix / Gram-defect law | Gram of the raw cycle basis (computed); needed only when amplitudes enter (Stage 2) |
