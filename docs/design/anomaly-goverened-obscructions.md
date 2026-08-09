# Experimental Design: Anomaly-Governed Obstructions in Bulk Synthesis for Kähler–Dirac Fermions

**Handoff document. Assume zero prior context.** Everything needed to implement, run,
and report this experiment is defined below. Do not consult external sources; the
specification is self-contained. Python 3 with `numpy`, `scipy`, `matplotlib` only.
Target total runtime: under 10 minutes on one CPU. All randomness seeded.

---

## 0. Mission summary

We test whether an optimizer that "synthesizes a bulk" between two boundaries can
decouple one boundary's fermion content from the other, and whether the obstruction to
doing so is governed by a topological invariant (an anomaly). The experiment has four
stages:

- **Stage G (gates):** build discrete differential operators on small simplicial
  complexes and verify six exact mathematical identities to machine precision.
  Everything downstream is invalid if any gate fails.
- **Stage A (linear synthesis):** an optimizer shapes a transverse "mass profile" to
  bind prescribed zero modes to one wall of a slab while suppressing them at the other
  wall. Prediction: the *linear* problem has **no** topological floor — the optimizer
  succeeds for all tested topologies, discovering an exponentially localized
  (domain-wall) profile.
- **Stage B (anomaly bookkeeping):** compute the exact integer invariant (index =
  Euler characteristic) that the obstruction in Stage C is conjectured to track, and
  verify its invariance under randomization of all continuous parameters.
- **Stage C (interacting zero-mode problem):** the real test. Reduce to the protected
  zero-mode subspace, enumerate symmetry-allowed interactions, and determine by exact
  diagonalization for which flavor numbers `N` and topologies the zero modes can be
  gapped **symmetrically** (unique ground state, gap, no symmetry-breaking order
  parameter). Compare the resulting table against three pre-registered hypotheses.

Deliverables (Section 8): `RESULTS.md`, `results.json`, `figures/`, and the code.

---

## 1. Mathematical definitions (implement exactly these)

### 1.1 Simplicial complexes and boundary matrices

A 2-dimensional oriented simplicial complex is given by:

- vertices `V = {0, 1, ..., n0-1}`,
- edges `E`: pairs `(i, j)` with `i < j`,
- triangles `F`: triples `(i, j, k)` with `i < j < k`.

Orientation convention: every simplex is stored with vertices in ascending order; its
orientation is that ordering.

Boundary matrices (real, sparse or dense — sizes here are tiny, dense is fine):

- `B1` of shape `(n0, n1)`: column for edge `(i, j)` has `-1` at row `i`, `+1` at row
  `j`.
- `B2` of shape `(n1, n2)`: column for triangle `(i, j, k)` has entries on its three
  edges: `+1` on `(j, k)`, `-1` on `(i, k)`, `+1` on `(i, j)`.

**Identity to verify:** `B1 @ B2 == 0` exactly.

Coboundary operators: `d0 = B1.T` (shape `(n1, n0)`), `d1 = B2.T` (shape `(n2, n1)`).
Codifferentials (with identity inner products on all grades): `delta1 = d0.T`,
`delta2 = d1.T`.

### 1.2 Hodge Laplacians, Betti numbers, Euler characteristic

- `L0 = d0.T @ d0`
- `L1 = d0 @ d0.T + d1.T @ d1`
- `L2 = d1 @ d1.T`

Betti numbers: `b_p = dim ker(L_p)`, computed as the number of eigenvalues below
`1e-10` (all matrices are symmetric positive semidefinite; use `numpy.linalg.eigh`).

Euler characteristic two ways (must agree):

- `chi_f = n0 - n1 + n2` (f-vector),
- `chi_b = b0 - b1 + b2` (Betti alternating sum).

### 1.3 The Kähler–Dirac operator and the grading

Work on the total cochain space `C = C0 ⊕ C1 ⊕ C2` of dimension `n0 + n1 + n2`. Define
the block operators:

```
K = [[ 0,      d0.T,   0    ],        # K is real symmetric
     [ d0,     0,      d1.T ],
     [ 0,      d1,     0    ]]

G = diag( +I_{n0}, -I_{n1}, +I_{n2} )   # grading operator, "Gamma": (-1)^p per grade
```

`K` is the (Hermitian form of the) Kähler–Dirac operator; `G` is the grade involution.

**Identities to verify:**

1. `K @ G + G @ K == 0` (anticommutation), to `1e-12`.
2. `K @ K == blockdiag(L0, L1, L2)`, to `1e-12`.
3. Spectral pairing: the multiset of **nonzero** eigenvalues of `L0 ⊕ L2` equals that
   of `L1`, to `1e-8` after sorting (this is supersymmetric pairing: `d` and `delta`
   map between even and odd grades).
4. `dim ker K = b0 + b1 + b2` and the **index**
   `index = trace of G restricted to ker K = b0 - b1 + b2 = chi`.
   Compute the restriction by projecting `G` with an orthonormal kernel basis
   `Q` (columns spanning `ker K`): `index = round(trace(Q.T @ G @ Q))`.
5. `trace(G) = n0 - n1 + n2 = chi` on the full space (no kernel projection).

### 1.4 Weighted robustness (used in Stage B)

Weighted versions: draw positive weights `w0 (n0,), w1 (n1,), w2 (n2,)` i.i.d. from
`Uniform[0.5, 2.0]`. Define diagonal matrices `W_p = diag(w_p)` and the weighted
codifferential `delta_p = W_{p-1}^{-1} d_{p-1}.T W_p`. Build the weighted `K_w` (no
longer symmetric in the plain inner product; symmetrize by the similarity transform
`K_sym = S K_w S^{-1}` with `S = blockdiag(W0^{1/2}, W1^{1/2}, W2^{1/2})`, or
equivalently build `K_sym` directly from `d̃_p = W_p^{1/2} d_p W_{p-1}^{-1/2}`).
Verify that `b_p`, `chi`, and the index are **unchanged** for 20 independent weight
draws (they are topological; this is the invariance the whole experiment leans on).

### 1.5 The complex zoo (construct all of these)

1. **TET** — boundary of the tetrahedron (minimal sphere `S^2`):
   `V = {0,1,2,3}`; `E` = all 6 pairs; `F` = all 4 triples.
   Expected: `n = (4, 6, 4)`, `b = (1, 0, 1)`, `chi = 2`.
2. **OCT** — boundary of the octahedron (`S^2` again, larger):
   vertices `0..5` with antipodal pairs `(0,5), (1,3), (2,4)`; faces = all 8 triples
   containing no antipodal pair; edges = all 12 pairs that are not antipodal.
   Expected: `n = (6, 12, 8)`, `b = (1, 0, 1)`, `chi = 2`.
3. **TOR** — triangulated torus `T^2`, grid size `m x m`, `m = 4`:
   vertices `(i, j)`, `i, j in 0..m-1`, index `v = m*i + j`; wrap arithmetic mod `m`.
   Edges: `(v, right(v))`, `(v, down(v))`, `(v, downright(v))` for every `v`
   (three edges per vertex; `right = (i, j+1)`, `down = (i+1, j)`,
   `downright = (i+1, j+1)`).
   Faces: for every `v`, triangles `(v, right(v), downright(v))` and
   `(v, downright(v), down(v))` (store each with vertices sorted ascending; the sign
   conventions of Section 1.1 handle orientation automatically **only if** `B1 B2 = 0`
   still holds — verify it; if it fails, fix face orientation by inspection until it
   passes).
   Expected: `n = (16, 48, 32)`, `b = (1, 2, 1)`, `chi = 0`.
4. **TET2 = TET ⊔ TET** — disjoint union (relabel the second copy's vertices `4..7`).
   Expected: `b = (2, 0, 2)`, `chi = 4`.
5. Optional robustness copy: **OCT-R** — one refinement of OCT (subdivide every face
   into 4 by adding edge midpoints). Only used to confirm invariants don't change.

Flavors: `N` copies of the Kähler–Dirac field on the same complex means the operator
`K_N = I_N ⊗ K` and grading `G_N = I_N ⊗ G`; the index becomes `N * chi`.

---

## 2. Stage G — validation gates

For each complex in the zoo, run and tabulate the following, with the stated
tolerances. **If any gate fails, stop, and write the failure into `RESULTS.md`; do not
proceed.**

| Gate | Statement | Tolerance |
|---|---|---|
| G1 | `B1 @ B2 = 0` | exact integers |
| G2 | `{K, G} = 0` | `1e-12` |
| G3 | `K^2 = L0 ⊕ L1 ⊕ L2` | `1e-12` |
| G4 | nonzero spec(`L0⊕L2`) = nonzero spec(`L1`) | `1e-8` |
| G5 | `b_p` match expectations; `chi_f = chi_b` | exact |
| G6 | index (kernel-projected `trace G`) `= chi`; full `trace G = chi` | exact |
| G7 | G5–G6 invariant under 20 random weightings (Section 1.4) | exact |

---

## 3. Stage A — linear bulk synthesis (the optimizer rediscovers the domain wall)

### 3.1 The slab operator

The bulk is a slab: the surface complex `Sigma` extruded over `n_s` transverse sites
(`n_s in {8, 16, 32}`). Do **not** build a 3-complex; use the standard algebraic
product:

```
K_bulk(theta) = K_Sigma ⊗ I_{n_s}  +  G_Sigma ⊗ T(theta)
```

where `T(theta)` is a real symmetric tridiagonal `n_s x n_s` matrix with diagonal
`(m_1, ..., m_{n_s})` and off-diagonal `(w_1, ..., w_{n_s-1})`; the optimizer's
parameter vector is `theta = (m_1..m_{n_s}, w_1..w_{n_s-1})`, initialized with
`m_k = 0`, `w_k = 1`, plus `Normal(0, 0.01)` seeded noise per restart. Because
`{K, G} = 0`, one has `K_bulk^2 = K_Sigma^2 ⊗ I + I ⊗ T^2`; a zero mode of the slab
factorizes as `(harmonic mode of Sigma) ⊗ (null vector of T)`.

### 3.2 The synthesis residual

Let `{h_a}, a = 1..A` be an orthonormal basis of `ker K_Sigma`
(`A = b0 + b1 + b2` per flavor; run `N = 1` here — Stage A is flavor-trivial).
Let `P_near` and `P_far` project a slab field onto transverse sites `s = 1` and
`s = n_s` respectively. For fields `Psi_a` on the slab define

```
r(theta) = sum_a  min_{Psi_a} [  ||K_bulk(theta) Psi_a||^2
                               + lam_b * || P_near Psi_a - h_a ||^2
                               + lam_f * || P_far  Psi_a ||^2      ]
```

with `lam_b = lam_f = 1`. The inner minimization is an ordinary linear least-squares
problem per `a` (assemble the stacked matrix and use `numpy.linalg.lstsq`, or solve
the normal equations with a `1e-10` ridge for conditioning). The outer minimization
over `theta` uses `scipy.optimize.minimize(method="L-BFGS-B")` with numerical
gradients or analytic ones if convenient; 20 restarts (seeds 0..19); report the best.

### 3.3 Measurements and pre-registered predictions

For each `Sigma in {TET, TOR, TET2}` and each `n_s`:

- **floor** = best `r` over restarts (report all restarts' final values too);
- transverse profile `p_a(s) = || Psi_a restricted to site s ||` for the optimal
  solution; fit `log p_a(s)` on the far half to a line; report slope and `R^2`;
- **leakage** `= || P_far Psi_a ||^2` at the optimum, versus `n_s` (semilog plot);
- the learned mass profile `m_k` (plot).

Predictions (mark each PASS/FAIL in `RESULTS.md`):

- **A1.** `floor < 1e-8` for every `(Sigma, n_s)` — i.e., **no topological floor in
  the linear problem**, for `chi = 0, 2, 4` alike. (Interpretation for the report: a
  quadratic/linear optimizer can always *separate* the modes; the anomaly cannot
  obstruct at this level. If A1 fails after all restarts, report gradient norms and
  the spread across restarts, and flag prominently: a genuine linear floor would be a
  surprising and important result.)
- **A2.** The learned `m_k` is (up to overall sign) a monotone kink-like profile:
  report `sum_k |sign changes of m|` (predict `<= 1`).
- **A3.** Leakage decreases exponentially with `n_s`: linear fit of `log(leakage)`
  vs `n_s` has `R^2 > 0.99` and negative slope.

---

## 4. Stage B — exact anomaly bookkeeping

Tabulate, for every complex in the zoo and `N = 1..4`:

- `chi`, index (must equal `chi` and `N*chi` for `N` flavors — verify by explicit
  kernel projection at `N = 1, 2`),
- the anomaly classes `A2 = (N*chi) mod 2` and `A4 = (N*chi) mod 4`,
- invariance of all of the above under the 20 random weightings (G7 already covers
  `N = 1`; spot-check `N = 2` on TET).

This table defines the "anomaly" column against which Stage C's outcomes are
compared. No dynamics here; this stage is exact bookkeeping and must have zero
numerical ambiguity.

---

## 5. Stage C — symmetric gapping of the protected zero modes (the discrimination)

### 5.1 Rationale (one paragraph, for the report)

Stage A shows a quadratic optimizer can always localize modes away from the far wall;
what it can never do is *remove* the protected zero-mode content, whose count is fixed
by topology (Stage B). The physics question is whether **interactions** can gap those
zero modes without breaking the symmetry — "symmetric mass generation." The
conjecture under test: this is possible **iff** the anomaly class vanishes. We test
it in the smallest honest arena: the zero-mode subspace itself, second-quantized,
with all symmetry-allowed interactions.

### 5.2 Construction

For `Sigma in {TET, TET2}` (i.e., `chi = 2` and `chi = 4`) and flavor numbers
`N in {1, 2}` (and `N = 3, 4` for TET if runtime permits; the Hilbert spaces below
are tiny, so it will):

1. Compute an orthonormal basis `{u_1..u_Z}` of `ker K_Sigma` (`Z = 2` for TET,
   `Z = 4` for TET2). Record each basis vector's grade content: the expectation
   `g_z = u_z.T @ G @ u_z` (for these complexes the kernel vectors can be chosen
   grade-pure; report `g_z` — they should all be `+1` here, grades 0 and 2).
2. Second-quantize: one complex fermion mode per `(z, flavor)` pair — `M = Z * N`
   modes, Fock dimension `2^M` (`<= 2^8 = 256` at the largest planned case; exact
   diagonalization is trivial).
3. Symmetry group `S` whose invariance defines "symmetric":
   - `U(1)_V`: `c_{z,f} -> e^{i a} c_{z,f}` (total fermion number),
   - `U(1)_A` (the Kähler–Dirac chiral rotation): `c_{z,f} -> e^{i a g_z} c_{z,f}`
     — with all `g_z = +1` this coincides with `U(1)_V` on the kernel; **derive and
     record this coincidence explicitly**; the discrete remnant relevant to the
     anomaly is the `Z_4` subgroup `a in {0, pi/2, pi, 3pi/2}` acting jointly with
     fermion parity,
   - flavor symmetry: `U(N)` rotating `f`,
   - complex conjugation `C`: `c <-> c^dagger` composed with the grade map (implement
     as the antiunitary particle-hole transformation on the Fock space; a Hamiltonian
     "respects C" iff `C H C^{-1} = H`).
4. Enumerate candidate Hamiltonians in the zero-mode space:
   - all quadratic terms `c^dag_i c_j + h.c.` and pairing terms
     `c^dag_i c^dag_j + h.c.`;
   - all quartic terms `c^dag_i c^dag_j c_k c_l` (+ h.c.);
   classify each monomial as ALLOWED or FORBIDDEN under `S` (check invariance under:
   a generic `U(1)_V` phase, the `Z_4` chiral element `a = pi/2`, one generic `U(N)`
   rotation, and `C`). Report the counts.
5. For the ALLOWED set, form `H(g) = sum_alpha g_alpha O_alpha` and search over
   couplings (`g_alpha in [-2, 2]`, 200 random draws per case, seeds 100..299, plus
   the all-equal point) for a Hamiltonian with:
   - **unique ground state** (degeneracy 1 within `1e-8`),
   - **gap** `Delta = E_1 - E_0 >= 0.1 * max|g|`,
   - **no condensate**: for every FORBIDDEN quadratic bilinear `Q_beta`,
     `|<GS| Q_beta |GS>| < 1e-8`.
   Record the best case found (max gap subject to the constraints).

### 5.3 The pre-registered discrimination table

Define `SYMGAP(Sigma, N) = 1` if a Hamiltonian satisfying all three conditions in
5.2.5 exists in the search, else `0`. Fill:

| case | `N*chi` | `A4` | `A2` | H-mod4 predicts | H-mod2 predicts | H-never predicts | measured |
|---|---|---|---|---|---|---|---|
| TET,  N=1 | 2 | 2 | 0 | 0 | 1 | 0 | ? |
| TET,  N=2 | 4 | 0 | 0 | 1 | 1 | 0 | ? |
| TET2, N=1 | 4 | 0 | 0 | 1 | 1 | 0 | ? |
| TET,  N=3 | 6 | 2 | 0 | 0 | 1 | 0 | ? |
| TET,  N=4 | 8 | 0 | 0 | 1 | 1 | 0 | ? |

The three hypotheses: **H-mod4** (`SYMGAP = 1` iff `N*chi ≡ 0 mod 4`), **H-mod2**
(iff `≡ 0 mod 2`), **H-never** (protected zero modes can never be symmetrically
gapped). The measured column decides. The discriminating rows are (TET, N=1) and
(TET, N=3) — the only rows where the hypotheses disagree.

Caveats to implement honestly:

- A `SYMGAP = 0` result is a statement about the searched family; increase the draw
  count to 1000 for any `0` row and state the final search size in the report.
- If the symmetry derivation in 5.2.3 admits a second reasonable definition of the
  chiral action on the kernel (e.g., if some `g_z = -1` appears on another complex),
  run the classification under **both** definitions and report both tables, clearly
  labeled. Do not silently pick one.

---

## 6. Optional Stage D — integration with the `tessera` repository

Only if the repository `github.com/akellehe/tessera` is available in the environment.
Read `examples/cobordism/emergent_proton.py` and the documentation under `docs/` to
discover the cobordism API. Then: (i) reproduce Stage A's synthesis with the
repository's own optimizer and connection-Laplacian machinery on a `Cylinder`
topology, pinning only the near boundary; (ii) locality probe: perturb one
near-boundary link phase by `0.1`, re-synthesize, and record the change in the bulk
solution as a function of transverse distance; fit exponential vs power-law decay.
If the API cannot be matched to this specification in reasonable time, skip Stage D
and say so; Stages G–C stand alone.

---

## 7. Numerical policy

- Seeds: fixed and logged for every stochastic step (weights: 0..19; Stage A
  restarts: 0..19; Stage C draws: 100..299, extended 300..1099 for zero rows).
- Tolerances: exact-integer checks by `numpy.array_equal`; float gates as tabulated;
  eigen-decompositions via `numpy.linalg.eigh` only (symmetric matrices throughout).
- Sizes: nothing in this design exceeds a few hundred dimensions (largest dense
  matrix: slab over TOR at `n_s = 32`: `(16+48+32)*32 = 3072` — still fine dense).
- Runtime budget: abort any single optimization at 60 s and report it.

## 8. Required outputs (exactly these)

1. **`RESULTS.md`** containing, in order:
   - environment (versions), total runtime;
   - Stage G gate table (PASS/FAIL, max abs deviation per gate per complex);
   - Stage A table (`Sigma, n_s, floor, leakage, slope, R^2, kink sign changes`) and
     PASS/FAIL for A1–A3, with one sentence of interpretation each;
   - Stage B anomaly table;
   - Stage C: the derived symmetry action on zero modes (the `g_z` values and the
     recorded coincidence or non-coincidence of `U(1)_A` with `U(1)_V` on the
     kernel), ALLOWED/FORBIDDEN operator counts, and the filled discrimination table
     with the winning hypothesis (or "inconclusive", with which rows blocked it);
   - a "Surprises and deviations" section: anything that did not match this
     specification, however small;
   - explicit statement of the single most load-bearing numerical result.
2. **`results.json`**: every number in the tables, machine-readable.
3. **`figures/`**: (a) Stage G spectral-pairing scatter; (b) Stage A transverse
   profiles (log scale) and learned `m_k` for one representative case per `Sigma`;
   (c) Stage A leakage vs `n_s` semilog; (d) Stage C gap and ground-state degeneracy
   vs coupling scale for the best Hamiltonian in each row.
4. The code, runnable end-to-end by a single entry point (`python run_all.py`).

## 9. What the requester will do with this

The filled discrimination table (Stage C) against the three hypotheses, the A1
verdict (linear problem floor-free), and the G/B exactness results will be pasted
back for joint interpretation. Prioritize correctness of the gates and honesty of
the Stage C search reporting over breadth; a smaller, airtight table beats a larger
ambiguous one.
