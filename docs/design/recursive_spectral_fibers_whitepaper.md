# Recursive Spectral Fibers on Simplicial Cobordisms

## A geometric program for quarks, color, fermion statistics, Fock space, and baryons

Implementation is tracked by GitHub epic
[#763](https://github.com/akellehe/tessera/issues/763).

Standalone editions: [LaTeX source](recursive_spectral_fibers_whitepaper.tex) and
[compiled PDF](recursive_spectral_fibers_whitepaper.pdf).

## Abstract

This paper proposes a single geometric formulation for the particle content already
suggested by Tessera's cobordism experiments. A connected, high-modularity
simplicial component is treated as one effective vertex at the next resolution.
The component's selected localized Hodge eigenspace is its fiber, and the couplings
between components induce transport between those fibers. Repeating this operation
produces a nested, potentially fractal hierarchy of complexes. No independent gauge
field, particle label, or auxiliary lattice is introduced: every state, connection,
exchange sign, and observable is read from oriented simplices, complex edge data,
edge qubits, and the Hodge/Regge operators already present in the construction.

The proposal has a substantial exact core. Schur reduction proves when a component
may be replaced by a vertex without changing boundary quadratic energies. The
graded tensor product of chain complexes gives the correct Hodge operator for
interacting systems. Three oriented edge qubits form the exact exterior algebra
`Λ* C^3 = 1 ⊕ 3 ⊕ 3̄ ⊕ 1`; the one-occupation sector is a color qutrit, its bilinears
close `su(3)`, the three-occupation sector is a color singlet, and the grading gives
the fermionic exchange sign and canonical anticommutation relations. A connection
is derived from overlap of neighboring spectral frames, and Wilson loops are then
gauge-invariant observables rather than new degrees of freedom. Successive
cobordism interactions generate the finite stages of an inductive-limit Fock space.

The physical identification remains a hypothesis to be tested. A quark is proposed
to be a persistent, odd-parity, rank-three spectral fiber localized on a
high-modularity component; a proton is three such components bound into one
persistent supercomponent, occupying a normalized color wedge, carrying baryon
number `+1`, electric charge `+1`, and a total-space spin-`1/2` holonomy. The paper
separates exact identities, results already supported by Tessera, and new
falsifiable conjectures. It also states an implementation program in which analytic
or structure-exact operations are used before iterative numerical work, so the
model can scale without turning its defining observables into uncontrolled
approximations.

## 1. Epistemic status and design constraint

Three kinds of statement are deliberately distinguished:

1. **Exact identity** — follows algebraically from the stated finite complex,
   orientation, and inner product.
2. **Existing Tessera evidence** — measured by an existing experiment, with the
   scope and residual reported in the repository.
3. **Proposed physical identification** — a new hypothesis with an explicit
   falsification test.

The governing constraint is parsimony. The ontology is limited to:

- an oriented simplicial complex and its cobordisms;
- a complex squared length and a normalized qubit on each edge;
- incidence, Hodge, and Regge operators derived from that data; and
- tensor composition when systems interact.

Spectral fibers, color frames, connections, Wilson loops, particle sectors, and
coarse vertices are *derived views* of that same data. They are not separately
sampled fields. This is important both scientifically and computationally: adding a
new independent field could fit a desired answer, while deriving every readout from
one complex leaves the construction falsifiable.

## 2. Present evidence in Tessera

Two existing results motivate the construction.

First, the state-operation-cobordism experiments show that the Hodge-carried
register is an isometry to machine precision and that its spectral value reproduces
the quantum transition amplitude for every operation that the tested geometry
actually carries. Generic fixed-complexity operations can remain obstructed, and
the obstruction is visible both as a residual floor and as leakage from the carried
subspace. The claim is therefore not that every finite complex realizes every gate;
it is that a realized, isometrically embedded register computes the corresponding
amplitude. See
[`cobordism-results.md`](../source/quantum-experiments/state-operation-cobordism/cobordism-results.md).

Second, interaction-history complexes exhibit a stable near-four-dimensional
spectral regime. The strongest reported measurements approach, but do not yet prove,
an exact spectral dimension of four. The current status and finite-size caveats are
recorded in
[`h_ds4_status.md`](../source/quantum-experiments/overview/h_ds4_status.md).

The current proton animation adds a third, more preliminary observation. It starts
with the phase pattern `{1, ω, ω²}` and evaluates its singlet diagnostics while a
joint Regge-Hodge stationarity objective changes the complex. It no longer forces
register holes to appear, and holes have not re-emerged in the current construction.
That negative result is useful: the proposed quark should therefore not be defined
as a hole. It will be sought as a persistent modular spectral cluster, while Betti
numbers remain independent topological observables.

## 3. The microscopic geometric state

Let `K` be a finite oriented simplicial complex. For every edge `e`, store

$$
x_e=(z_e,q_e),\qquad
z_e=\rho_e e^{i\theta_e}\in\mathbb C,\qquad
q_e=\alpha_e|0\rangle+\beta_e|1\rangle,\quad
|\alpha_e|^2+|\beta_e|^2=1.
$$

The complex phase is not a second link field; it is part of the existing complex
edge geometry. The qubit is the occupation/amplitude carrier. An implementation may
choose a convention in which the relative qubit phase is locked to `θ_e`, but the
mathematics only requires that the convention be explicit and invariant under edge
reorientation.

Let

$$
\partial_k:C_k(K)\longrightarrow C_{k-1}(K),\qquad
\partial_{k-1}\partial_k=0
$$

be the oriented boundary maps and let `W_k(z)` be the metric weight on `k`-chains.
With the weighted adjoint

$$
\partial_k^*=W_k^{-1}\partial_k^\dagger W_{k-1},
$$

the degree-`k` Hodge Laplacian is

$$
L_k=\partial_{k+1}\partial_{k+1}^*+\partial_k^*\partial_k.
$$

In the positive metric regime this is self-adjoint in the `W_k` inner product. In
the signed Lorentzian regime it can be non-normal; then left and right spectral
frames and their biorthogonal condition numbers must be reported rather than
silently treating `L_k` as Hermitian.

The geometry evolves toward joint stationary points of the existing Regge and Hodge
functionals. In emergence mode, particle-specific observables below are read after
optimization and are not inserted as target terms. Controlled synthesis mode may
pin a carrier to test realizability, but that is a separate experiment.

## 4. A component is exactly an effective vertex

Partition the `k`-cells of a connected component into interface cells `B` and
interior cells `I`, and block its Hodge operator as

$$
L=
\begin{pmatrix}
L_{BB}&L_{BI}\\
L_{IB}&L_{II}
\end{pmatrix}.
$$

After projecting out incompatible interior zero modes, minimization over the
interior has the exact solution

$$
x_I^*=-L_{II}^{+}L_{IB}x_B,
$$

and the exact effective boundary operator

$$
\boxed{L_{\mathrm{eff}}
=L_{BB}-L_{BI}L_{II}^{+}L_{IB}}.
$$

Here `+` denotes the Moore-Penrose inverse on the supported interior subspace. For
every compatible boundary value,

$$
\min_{x_I}
\begin{pmatrix}x_B\\x_I\end{pmatrix}^{\!\dagger}
L
\begin{pmatrix}x_B\\x_I\end{pmatrix}
=x_B^\dagger L_{\mathrm{eff}}x_B.
$$

This is the precise sense in which a connected component can be replaced by a
coarse vertex: all external quadratic response is preserved. The effective blocks
between two coarse components become the coarse edges. A harmonic interior mode is
not discarded; it becomes an explicit fiber coordinate attached to the coarse
vertex.

This reduction is exact for the retained interface. Approximation enters only if
one additionally truncates the interface or the fiber, and that approximation has a
measurable residual.

## 5. Recursive spectral fibers

Let `P_ℓ={C_v^ℓ}` be an intrinsic partition of the complex at scale `ℓ` into
persistent, high-modularity connected components. Define the next complex by

$$
K_{\ell+1}=K_\ell/P_\ell,
$$

with one vertex for each component and effective couplings supplied by Schur
reduction. Within component `C`, choose an isolated localized spectral band and a
weighted orthonormal frame

$$
\Phi_C=(\phi_1,\ldots,\phi_r),\qquad
\Phi_C^\dagger W_C\Phi_C=I_r.
$$

The derived fiber is

$$
E_C=\operatorname{Ran}\Phi_C.
$$

It need not be a harmonic space and therefore need not be supported by a hole. What
it does require is a spectral gap, localization, and persistence. A candidate
component is accepted only if all of the following remain stable across a stated
range of scales:

- high modularity and low conductance relative to neighboring cuts;
- a localized spectral projector with stable rank;
- a nonzero band gap separating it from discarded modes;
- overlap with its predecessor and successor components;
- lifetime across multiple cobordism frames; and
- small external transport leakage.

This gives a self-similar hierarchy

$$
\cdots\longrightarrow K_2\longrightarrow K_1\longrightarrow K_0
$$

in which a vertex at one level resolves into a connected complex at the next. A
fractal-like pattern is permitted but not required: the measured scaling of module
count, volume, boundary size, and spectral gap decides whether the hierarchy is
self-similar.

## 6. Interactions and the expanding Hilbert space

For chain complexes `A` and `B`, the graded tensor differential is the exact rule

$$
d_{A\widehat\otimes B}(a\otimes b)
=d_Aa\otimes b+(-1)^{\deg a}a\otimes d_Bb.
$$

For a noninteracting product with product metric,

$$
L_{A\widehat\otimes B}=L_A\otimes I+I\otimes L_B,
$$

so eigenvalues add and eigenvectors tensor. Connecting simplices contributed by a
cobordism add a sparse interaction operator `V_AB`:

$$
L_{AB}=L_A\otimes I+I\otimes L_B+V_{AB}.
$$

At the fiber level, an interaction grows the carried space as

$$
\mathcal H_{AB}=E_A\widehat\otimes E_B,
$$

and a later interaction appends another factor. If `J_C` embeds an abstract state
into the geometric carrier, exact amplitude preservation requires

$$
J_C^\dagger W_CJ_C=I.
$$

Tensor products preserve isometry exactly. If `G=J^\dagger WJ` has Gram defect
`ε=||G-I||`, then

$$
|a^\dagger Gb-a^\dagger b|
\leq \varepsilon\,\|a\|\,\|b\|,
$$

and two tensor factors obey

$$
\varepsilon_{AB}
\leq\varepsilon_A+\varepsilon_B+\varepsilon_A\varepsilon_B.
$$

Thus the amplitude claim has an explicit, composable error budget.

## 7. A triangle carries the exact color algebra

Consider three oriented edge qubits around a triangle and interpret `|1>` as an
occupied edge mode. Their graded tensor product is canonically the exterior algebra

$$
(\mathbb C^2)^{\widehat\otimes 3}
\cong\Lambda^\bullet\mathbb C^3
=\mathbf1\oplus\mathbf3\oplus\overline{\mathbf3}\oplus\mathbf1.
$$

The sectors have occupation number `N=0,1,2,3`:

| Sector | Basis dimension | Color interpretation | Fermion parity |
|---|---:|---|---:|
| `Λ^0 C^3` | 1 | vacuum | even |
| `Λ^1 C^3` | 3 | fundamental color triplet | odd |
| `Λ^2 C^3` | 3 | antisymmetric anti-triplet | even |
| `Λ^3 C^3` | 1 | color singlet | odd |

Let `a_i^†,a_i` be the exterior creation and contraction operators. They satisfy the
canonical anticommutation relations exactly:

$$
\{a_i,a_j\}=0,\qquad
\{a_i^\dagger,a_j^\dagger\}=0,\qquad
\{a_i,a_j^\dagger\}=\delta_{ij}.
$$

On the one-occupation sector, the bilinears

$$
E_{ij}=a_i^\dagger a_j
$$

satisfy

$$
[E_{ij},E_{k\ell}]=\delta_{jk}E_{i\ell}-\delta_{i\ell}E_{kj}.
$$

The six Hermitian off-diagonal combinations together with

$$
H_1=E_{11}-E_{22},\qquad
H_2=\frac{E_{11}+E_{22}-2E_{33}}{\sqrt3}
$$

are the eight generators of `su(3)`. Thus the triangle does not merely hold three
phases: its one-particle edge sector carries the fundamental representation, its
two-particle sector carries the dual representation, and its traceless bilinears
carry the adjoint octet.

### 7.1 Geometric normalization

For triangle edge magnitudes `ℓ_i` and phases `θ_i`, define

$$
c_i=\frac{\ell_i e^{i\theta_i}}
{\sqrt{\ell_1^2+\ell_2^2+\ell_3^2}},\qquad
|c\rangle=\sum_{i=1}^3c_i|i\rangle.
$$

Then `⟨c|c⟩=1`. Constraining the perimeter to one is a valid geometric scale gauge,
but it is an `L^1` condition and does not replace the `L^2` Hilbert normalization.
Normalized pure color states form `CP^2`; `SU(3)` is the transformation group, not
the surface of the triangle itself.

### 7.2 The existing omega phase pattern

Let `ω=e^{2πi/3}`. The exact Fourier frame

$$
F_3=\frac1{\sqrt3}
\begin{pmatrix}
1&1&1\\
1&\omega&\omega^2\\
1&\omega^2&\omega
\end{pmatrix}
$$

is unitary. The existing pattern `(1,ω,ω²)/√3` is therefore one color basis vector,
not by itself the whole color fiber. Its cyclic orbit supplies an exact orthonormal
triad.

## 8. Color transport and Wilson loops without a new gauge field

Let `T_AB` be the chain-level transfer already induced by connecting simplices from
component `B` to component `A`. In local spectral frames, the raw fiber map is

$$
M_{AB}=\Phi_A^\dagger W_A T_{AB}\Phi_B.
$$

Its departure from an isometry is a physical leakage diagnostic:

$$
\eta_{AB}=\|M_{AB}^\dagger M_{AB}-I\|.
$$

When the selected band is isolated and `η_AB` is small, take the polar unitary and
remove its determinant phase:

$$
V_{AB}=M_{AB}(M_{AB}^\dagger M_{AB})^{-1/2},\qquad
U_{AB}=\frac{V_{AB}}{(\det V_{AB})^{1/3}}\in SU(3).
$$

Under a change of local spectral frame, `Φ_A→Φ_Ag_A` and `Φ_B→Φ_Bg_B`,

$$
U_{AB}\longmapsto g_A^\dagger U_{AB}g_B.
$$

Consequently the Wilson observable for a closed coarse loop `γ` is exactly
gauge-invariant:

$$
W(\gamma)=\frac13\operatorname{Tr}
\prod_{(AB)\in\gamma}U_{AB}.
$$

Polar normalization must never conceal a bad fiber assignment: every Wilson value
is reported together with `η_AB`, the band gap, and the frame condition number.

## 9. Quarks as modular clusters

A *quark candidate* is proposed to be a component `Q` satisfying all of the
following derived conditions:

1. `Q` is a persistent high-modularity cluster, not a prescribed region.
2. Its selected color fiber has stable rank three.
3. It occupies an odd exterior sector.
4. Its color transport has bounded leakage over its lifetime.
5. Its oriented world tube has one unit of quark-direction flux; reversing the
   tube yields the dual color representation and an antiquark.
6. Its total spectral fingerprint is stable under refinement and vertex relabeling.

The distinction between an antiquark and the `Λ^2 C^3` anti-triplet of two quarks is
made by oriented world-tube flux and total occupation, not color alone. Assigning
`B=+1/3` to a forward odd quark tube and `B=-1/3` to its orientation reverse makes
pair creation conserve baryon number exactly.

Flavor and electric charge are not assumed as hidden labels. The conservative
hypothesis is that two stable subclasses of the same cluster fiber provide an
isospin doublet. On such a doublet, the measured orientation flux supplies baryon
number and the standard relation

$$
Q=I_3+\frac{B}{2}
$$

gives `Q_u=+2/3` and `Q_d=-1/3`. This is a proposed identification, not yet a
derivation: it succeeds only if an unlabeled two-dimensional spectral band emerges,
is transported coherently, and its Gauss-flux readout agrees with those values.

## 10. Fermion statistics from simplicial orientation

The graded interchange law is

$$
\tau(a\widehat\otimes b)
=(-1)^{F_aF_b}b\widehat\otimes a.
$$

Two odd clusters therefore acquire a minus sign on exchange, while an even
composite does not. Parity adds modulo two:

| Object | Odd constituents | Composite parity |
|---|---:|---:|
| quark or antiquark | 1 | odd |
| meson `q q̄` | 2 | even |
| diquark `q q` | 2 | even |
| baryon `q q q` | 3 | odd |

Pauli exclusion is the exact exterior-algebra identity

$$
\|v_1\wedge\cdots\wedge v_n\|^2
=\det[\langle v_i,v_j\rangle].
$$

If two complete one-particle modes coincide, the determinant and the state vanish.
The “complete” qualifier prevents double-counting signs: color, spin, flavor,
space, and component support are wedged once as one mode. One must not multiply an
extra fermion sign by the sign already present in the color epsilon tensor.

### 10.1 A label-independent exchange experiment

Let `Φ_t` be an orthonormal frame for the isolated odd subspace at frame `t`. Define
the parallel transport

$$
R_t=\operatorname{polar}(\Phi_{t+1}^\dagger W_t\Phi_t),\qquad
U_\gamma=R_{T-1}\cdots R_0.
$$

The determinant line gives the exchange character

$$
\chi_F(\gamma)=\det U_\gamma.
$$

The proposed fermion test is

$$
\chi_F(\text{single exchange})=-1,\qquad
\chi_F(\text{double exchange})=+1.
$$

Because the determinant is invariant under conjugation, this test is independent of
the arbitrary frame chosen inside the isolated band. A separate `2π` rotation loop
must also return `-1` before the construction can claim a spin-statistics link.

## 11. Fock space as an inductive limit of interactions

For `M` oriented fermionic edge modes,

$$
\widehat\bigotimes_{m=1}^{M}\mathbb C^2
\cong\Lambda^\bullet\mathbb C^M
=\bigoplus_{n=0}^{M}\Lambda^n\mathbb C^M,
$$

and the dimension identity is exact:

$$
2^M=\sum_{n=0}^{M}{M\choose n}.
$$

Adding a new noninteracting mode uses the vacuum embedding

$$
\iota_M:\mathcal H_M\hookrightarrow\mathcal H_{M+1},\qquad
\iota_M(\psi)=\psi\widehat\otimes|0\rangle.
$$

The infinite Fock space is the direct limit

$$
\mathcal F=\varinjlim(\mathcal H_M,\iota_M).
$$

This makes the infinite expansion precise without ever allocating an infinite
array. At every finite simulation time only finitely many modes have interacted.
Consistency requires

$$
\|\iota_MU_M-U_{M+1}\iota_M\|\longrightarrow0
$$

over a refinement sequence.

Bosonic gauge excitations need not add a new local oscillator. The traceless even
bilinears

$$
a_i^\dagger a_j-\frac13\delta_{ij}N
$$

span the color octet in `3⊗3̄=1⊕8` and have even fermion parity. Arbitrarily many
such collective excitations are represented by adding more microscopic modes at
finer resolution. The finite edge qubit remains unchanged.

## 12. The proton as the maximally informative baryon

The proton is chosen because, beyond generic baryon structure, it demands a
nontrivial flavor pattern, electric charge, spin, and experimentally meaningful
form factors.

Let three persistent quark components `A,B,C` have color frames and normalized
color columns `c_A,c_B,c_C`. The invariant color volume is

$$
S_{ABC}=\epsilon_{ijk}c_A^ic_B^jc_C^k
=\det[c_A\ c_B\ c_C].
$$

Under a common `g∈SU(3)`, `S→det(g)S=S`. Its squared magnitude is the Gram
determinant

$$
|S_{ABC}|^2=\det(C^\dagger C)\in[0,1].
$$

The value one means the three color directions form an orthonormal frame. Their
normalized wedge is then the unique `Λ^3 C^3` singlet. The proposed proton
certificate is the conjunction:

- three persistent odd rank-three quark clusters;
- one persistent bound supercluster containing them;
- normalized color wedge with `|S_ABC|^2≈1` and vanishing net color flux;
- flavor spectrum with the `uud` occupation pattern;
- oriented baryon flux `B=1`;
- Gauss flux `Q=+1`;
- total-space spin holonomy `J^2=3/4` and `2π→-1`;
- finite radius and stable spectral mass/form-factor readouts; and
- stability of every dimensionless certificate under refinement.

None of these conditions should be included as an emergence target. The proton is
found only if the base geometric optimization produces a component satisfying them.
Targeted runs remain valuable as existence and obstruction experiments, but must be
labeled as synthesis rather than emergence.

## 13. The master recursive construction

At every scale:

$$
\boxed{
\begin{aligned}
K_{\ell+1}&=K_\ell/P_\ell,\\
E_v^{\ell+1}&=\text{isolated localized spectral subspace of }C_v^\ell,\\
L_{\ell+1}&=\operatorname{Schur}(L_\ell;P_\ell),\\
U_{vw}^{\ell+1}&=\operatorname{SU3Polar}
\big((\Phi_v^\ell)^\dagger W T_{vw}\Phi_w^\ell\big),\\
\mathcal H_{\ell+1}&=\widehat\bigotimes_v E_v^{\ell+1}.
\end{aligned}}
$$

This one recursion supplies the effective geometry, the internal fiber, the derived
connection, and the expanding state space. A high-level vertex can itself contain
the same pattern at the next finer level.

## 14. Exactness and performance principles

The simulation should prefer an exact structural identity over a general dense
numerical operation whenever both compute the same object:

- use sparse Schur solves, not explicit dense inverses;
- use Künneth sums for uncoupled product spectra, not diagonalization of the full
  Kronecker matrix;
- use exterior bit parity for exchange signs, not sampled phases;
- use exact `3×3` determinants and the fixed `F_3` frame for color certificates;
- use analytic Regge/Hodge derivatives and Wirtinger gradients, not finite
  differences;
- use Smith normal form for integer homology and a spectral threshold only as a
  cross-check;
- use matrix-determinant/Woodbury updates for local cobordism changes;
- cache component factorizations and invalidate only affected stars;
- keep tensor products lazy and block-sparse by occupation/parity; and
- attach residual, gap, leakage, and condition-number certificates to every
  iterative eigensolve.

The exact route is not only faster. It prevents a numerical tolerance from becoming
an undocumented physical postulate.

## 15. Falsification program

The formulation fails, or must be narrowed, if any of the following persists under
refinement and tighter numerical certification:

1. **No persistent rank-three clusters.** High-modularity components appear, but
   their localized fiber rank or spectral gap is unstable.
2. **No faithful coarse response.** Schur-reduced components fail to reproduce
   measured external amplitudes within their stated residual.
3. **No derived gauge covariance.** Wilson values depend on the local spectral
   frame after leakage is controlled.
4. **No fermion holonomy.** A single cluster exchange does not give `-1`, or the
   sign changes under relabeling.
5. **No spinor rotation.** Exchange works but a `2π` physical rotation does not
   give `-1`.
6. **No inductive compatibility.** Adding vacuum modes changes already-computed
   amplitudes by a nonvanishing amount.
7. **No unforced baryon.** Targeted synthesis can build the certificates, but the
   stationary geometric ensemble never produces them without a proton-specific
   term.
8. **No continuum stability.** Dimensionless color, parity, charge, spin, and
   amplitude certificates drift rather than converge with refinement.

Holes may re-emerge and may correlate with some phases, but no claim in this paper
depends on them doing so.

## 16. Conclusion

The proposed geometry is economical: a quark is a modular spectral component, its
color is a three-edge exterior sector, its gauge transport is the overlap of local
spectral frames, its fermionic sign is the grading forced by orientation, and its
Fock space is the limit of the tensor factors created by interactions. Three such
components form a baryon through their normalized color wedge; the proton is the
sharpest test because it also demands the correct charge, flavor, and spin.

Much of the algebra works out exactly. What remains genuinely open is the important
part: whether Tessera's unforced Regge-Hodge dynamics actually produces the required
persistent clusters and holonomies. That question is now a finite sequence of
measurable implementation and validation tasks rather than a metaphor.

## References and repository evidence

- F. Dörfler and F. Bullo, “Kron Reduction of Graphs with Applications to
  Electrical Networks,” [`arXiv:1102.2950`](https://arxiv.org/abs/1102.2950).
- Tessera amplitude and obstruction results,
  [`cobordism-results.md`](../source/quantum-experiments/state-operation-cobordism/cobordism-results.md).
- Tessera spectral-dimension status,
  [`h_ds4_status.md`](../source/quantum-experiments/overview/h_ds4_status.md).
- Tessera interaction-history construction,
  [`interaction-history-monte-carlo.md`](../source/interaction-history-monte-carlo.md).
- Existing total-space spin obstruction,
  [`joint_proton_spin_findings.md`](joint_proton_spin_findings.md).
- Existing fixed-partition modularity implementation,
  [`ModularityOptimizer.h`](../../include/observables/ModularityOptimizer.h).
- Current visualization and joint-stationarity experiment,
  [`proton_animation.py`](../../examples/cobordism/proton_animation.py).
