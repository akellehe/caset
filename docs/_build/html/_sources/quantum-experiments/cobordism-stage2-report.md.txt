# Cobordism experiment — Stage 2 (topological layer): results

> What the Stage-2 ($n=3$, simplicial) topological layer built, and what it
> shows. Companion to the runnable oracle
> `examples/cobordism/topological_correspondence.py` and to the Stage-1 plan
> `cobordism-plan.md`; the experiment charter is `cobordism.md` §5.

## Verdict

The State–Operation–Cobordism correspondence is **supported** at the topological
layer. The $\mathbb{Z}_2$ Dijkgraaf–Witten state-sum $Z(W)$ is a well-defined,
triangulation-invariant cobordism functor whose behaviour matches the hypothesis:
the falsifiable predictions **P1, P2, P3 all hold** (checks T1–T3), with T4, T5 and
the §5.6 Lorentzian variant as structural corroboration. The oracle runs clean and
deterministic:

```
State-Operation-Cobordism correspondence -- Stage 2 (topological oracle)

  check                            result   residual  detail
  ----------------------------------------------------------------------------------
  T1  cylinder = identity          PASS     6.28e-16  map(T²×I)=id₄; ⟨ψ_A|Z|ψ_B⟩=⟨ψ_A|ψ_B⟩
  T2  Pachner invariance           PASS     0.00e+00  Z(S²×S¹) drift 0 over 18 interior moves
  T3  sign is an invariant         PASS     0.00e+00  Z_Sign(RP³)=0 ≠ Z_Triv(RP³)=1; negs S²×S¹,T³
  T4  cross-layer holonomy         PASS     0.00e+00  holonomy == Stage-1 flux Φ_γ (mod 2π)
  T5  composition / functoriality  PASS     0.00e+00  map(glue(W₁,W₂)) == map(W₂)·map(W₁) == id₄
  §5.6  Lorentzian null-harmonic   PASS     3.36e-15  ⟨h,h⟩_W=(2-α)/3: +below, null at, −above α=2
  ----------------------------------------------------------------------------------

  P1, P2, P3 supported: YES   (P4, P5 carried from Stage 1)
```

## What was built

The Stage-2 machinery extends the Stage-1 seam (mesh/topology ↔ operator/state) from
degree 0 to degree $k\ge 1$ and adds the discrete TQFT partition function. All of it
is class-organized in the existing namespaces; no parallel hierarchies.

- **Metric Hodge Laplacian, $k\ge 1$** (`cobordism::HodgeLaplacian`). Assembles
  $L_k=\partial_k^\ast\partial_k+\partial_{k+1}\partial_{k+1}^\ast$ from `ChainComplex`
  boundary matrices with simplex-volume weights; $\ker L_k\cong H_k$ gives the qubit
  $\ker L_1(T^2)=\mathbb{C}^2$. A combinatorial fast path ($W_k=I$) is kept as a
  cross-check.
- **Lorentzian d'Alembertian** (§5.6). With signed simplex volumes the inner product
  goes indefinite, $L_k$ becomes non-self-adjoint (a general eigensolver), and
  $\ker L_k\cong H_k$ degrades to a pseudo-Hodge decomposition with null-norm modes.
- **Foundations.** `IntegerLinalg::gf2Nullspace` (flat $\mathbb{Z}_2$ connections),
  `ChainComplex::orientedTopSimplices`/`fundamentalClass`/`kSimplexVertices` (the
  orientation signs $\epsilon_t$ and the C₁ ordering), and signature-aware
  `Simplex::volume`/`cayleyMengerMatrix` (honest signed geometry; Wick rotation made
  explicit so CDT/Regge are unchanged).
- **The state-sum** (`cobordism::DijkgraafWitten`). Closed $Z(W)=\frac{1}{2^{|V|}}
  \sum_{\text{flat }g}\prod_t\omega(g)^{\epsilon_t}$ for both cocycle classes
  ($\omega\equiv1$ and $\omega(a,b,c)=(-1)^{abc}$), and the boundary reading
  $Z(W):Z(\Sigma_B)\to Z(\Sigma_A)$ (fix $g|_{\partial W}$, sum interior).
- **Fixtures.** `SphereCircleProduct` ($S^2\times S^1$), the Walkup 11-vertex
  `RealProjectiveSpace` ($\mathbb{RP}^3$), and `StellarSubdivision` (retriangulations).
- **Moves & gluing.** The existing `PachnerMove` family generalized to pre-geometric,
  boundary-fixed interior moves; `Cobordism::glue`/`selfGlue` for composition.
- **Cross-layer probe.** `WilsonLoop`'s `U1_CONNECTION` mode (holonomy of the
  `Edge::phase` connection).

## The checks

- **T1 — cylinder = identity (P1).** For $W=\Sigma\times[0,T]$, $Z(W)$ is the identity
  on $Z(\partial\Sigma)$: `map(T²×I)` is exactly $\mathrm{id}_4$, and
  $\langle\psi_A|Z(W)|\psi_B\rangle=\langle\psi_A|\psi_B\rangle$ for harmonic-1-form
  boundary states $\psi\in\ker L_1(T^2)$.
- **T2 — triangulation independence (P2), the make-or-break.** $Z(W)$ is invariant to
  machine precision across interior Pachner sequences (including $1\!\to\!4$ moves
  that change $|V|$) and across distinct triangulations of $S^2\times S^1$, for both
  cocycle classes.
- **T3 — the sign carries an invariant (P3).** $Z_{\text{sign}}(\mathbb{RP}^3)=0\ne
  1=Z_{\text{triv}}(\mathbb{RP}^3)$, while the two coincide on the negative controls.
- **T4 — cross-layer consistency.** The bulk $\mathbb{Z}_2$ holonomy equals the
  Stage-1 cycle flux restricted to $\{0,\pi\}$.
- **T5 — composition / functoriality.** $Z(W_2\cup_{\Sigma_C}W_1)=Z(W_2)\,Z(W_1)$
  (matrix product), with $\operatorname{Tr}(\text{map})=Z_{\text{closed}}$ on the
  self-glue.

## Key findings

1. **The sign cocycle does *not* distinguish $T^3$.** The twist is the mod-2 cup-cube
   $(-1)^{\langle g^3,[W]\rangle}$. In $H^*(T^3;\mathbb{Z}_2)=\Lambda(x,y,z)$ every
   degree-1 class has $g^2=0$, hence $g^3=0$ — so $T^3$ (and $S^2\times S^1$) are
   **negative** controls. The positive control requires a 1-class with $g^3\ne0$, i.e.
   $\mathbb{RP}^3$ ($H^*=\mathbb{Z}_2[t]/t^4$, $t^3\ne0$). The 2-torsion in
   $H_1(\mathbb{RP}^3)=\mathbb{Z}_2$ is exactly what separates it from $S^2\times S^1$
   (same $\mathbb{Z}_2$ Betti numbers, free $H_1$). The untwisted normalization is
   $Z_{\text{triv}}(W)=2^{b_1(\mathbb{Z}_2)-1}$.
2. **Lorentzian harmonics go null at $\alpha=2$.** On the 3-cycle with one timelike
   edge ($l^2=-\alpha^2$) the spectrum is exactly $\{0,3,1-2/\alpha\}$ (non-PSD for
   $\alpha<2$); the harmonic's indefinite norm $(2-\alpha)/3$ crosses zero at
   $\alpha=2$ — a concrete realization of the §5.6 "harmonic representative becomes
   null."
3. **A `ChainComplex` homology bug, found and fixed.** `fromSpacetime` traversed *all*
   registered simplices, but the mesh leaves orphaned lazy facets after a move,
   corrupting the chain groups (negative Betti) on mutated complexes; fixed by seeding
   from top cells only — required for the state-sum to survive Pachner moves.
4. **The brute-force state-sum bounds fixture size.** Flat-connection enumeration is
   $2^{\dim Z^1}$, capped at $\dim Z^1\le 24$. So the DW partition function runs on
   small manifolds ($S^3$, $S^2\times S^1$, $\mathbb{RP}^3$); $T^3$ (27 vertices,
   $\dim Z^1=29$) is used only for the combinatorial/homology (T2) checks, and its
   $g^3=0$ status is read off its torsion-free $H_1$.

## Conventions

- Operators are flat row-major; vec is the row-major flatten (Stage 1).
- The $k=0$ Laplacian keeps the **magnitude** degree convention $D_{ii}=\sum_j|A_{ij}|$
  (Hermitian, unitary evolution); the $k\ge1$ metric weights $W_k$ are honest signed
  simplex volumes (Euclidean = $|l^2|$; Lorentzian preserves the sign).
- Wick rotation is explicit/opt-in in `Simplex` geometry, so the CDT/Regge path is
  byte-unchanged.

## Reproduce

```
pip install -e ".[dev]"
python examples/cobordism/topological_correspondence.py   # the table above
python -m pytest tests/cobordism                                      # the per-check tests
```

Parameter sweeps (flux, Pachner depth, $\alpha$) and figures are written to
`/tmp/cobordism/`; they are not committed.

## Beyond Stage 2

Not built here, the natural continuations:

- **Boundary-state synthesis (§4b)** — the inverse eigenvector problem: the simplest
  complex whose $L_k$ has a target state as an eigenvector, grown by topology-preserving
  coning / Pachner-add.
- **Realizability search (§5.0)** — given an operation $U$, *find* a bulk $W$ with
  $Z(W)=U$ or prove none exists, and characterize the realizable set $\{Z(W)\}$ (for
  $\mathbb{Z}_2$-DW on $T^2$, $Z(T^2)=\mathbb{C}^4$ generated by the modular $S,T$).
- **More fixtures** — lens-space families $L(p,q)$, mapping tori, genus-$g$ surfaces;
  and a non-brute-force state-sum to lift the $\dim Z^1\le 24$ ceiling.
