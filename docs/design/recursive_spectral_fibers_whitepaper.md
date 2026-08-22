<div class="center">

Implementation is tracked by GitHub epic
[\#763](https://github.com/akellehe/tessera/issues/763).

</div>

# Epistemic status and design constraint

Three kinds of statement are deliberately distinguished:

1.  **Exact identity** — follows algebraically from the stated finite
    complex, orientation, and inner product.

2.  **Existing Tessera evidence** — measured by an existing experiment,
    with the scope and residual reported in the repository.

3.  **Proposed physical identification** — a new hypothesis with an
    explicit falsification test.

The governing constraint is parsimony. The ontology is limited to:

- an oriented simplicial complex and its cobordisms;

- a complex squared length and one two-level occupation mode on each
  edge;

- incidence, Hodge, and Regge operators derived from that data;

- a generally entangled boundary/Fock state on those modes; and

- simplicial gluing followed by fermionic second quantization.

Spectral fibers, color frames, connections, Wilson loops, particle
sectors, and coarse vertices are *derived views* of that same data. They
are not separately sampled fields. This is important both scientifically
and computationally: adding a new independent field could fit a desired
answer, while deriving every readout from one complex leaves the
construction falsifiable.

<figure id="fig:concept-map">

<figcaption>Concept map for the recursive complex construction. Colors
encode epistemic status, not physical sectors: blue is established or
exact machinery, green is a derived observable, and amber is a proposed
physical identification.</figcaption>
</figure>

# Present evidence in Tessera

Two existing results motivate the construction.

First, the state-operation-cobordism experiments show that the
Hodge-carried register is an isometry to machine precision and that its
spectral value reproduces the quantum transition amplitude for every
operation that the tested geometry actually carries. Generic
fixed-complexity operations can remain obstructed, and the obstruction
is visible both as a residual floor and as leakage from the carried
subspace. The claim is therefore not that every finite complex realizes
every gate; it is that a realized, isometrically embedded register
computes the corresponding amplitude. See
[`cobordism-results.md`](../source/quantum-experiments/state-operation-cobordism/cobordism-results.md).

Second, interaction-history complexes exhibit a stable
near-four-dimensional spectral regime. The strongest reported
measurements approach, but do not yet prove, an exact spectral dimension
of four. The current status and finite-size caveats are recorded in
[`h_ds4_status.md`](../source/quantum-experiments/overview/h_ds4_status.md).
Diffusion-based spectral dimension on simplicial quantum geometries has
important precedent in causal dynamical triangulations (Ambjørn,
Jurkiewicz, and Loll 2005); the Tessera evidence is an independent
result for a different construction and should be compared at the level
of the return-probability estimator and its finite-size window.

The current proton animation adds a third, more preliminary observation.
It starts with the phase pattern $\{1,\omega,\omega^2\}$ and evaluates
its singlet diagnostics while a joint Regge-Hodge stationarity objective
changes the complex. It no longer forces register holes to appear, and
holes have not re-emerged in the current construction. That negative
result is useful: the proposed quark should therefore not be defined as
a hole. It will be sought as a persistent modular spectral cluster,
while Betti numbers remain independent topological observables.

# The microscopic geometric state

Let `K` be a finite oriented simplicial complex. For every edge $e$,
store the complex squared length

$$z_e=\rho_e e^{i\theta_e}\in\mathbb C$$

and attach the two-level occupation factor
$\mathcal H_e=\operatorname{span}\{|0\rangle_e,|1\rangle_e\}$. The phase
of $z_e$ is not a second link field; it is part of the existing complex
edge geometry. Saying that an edge “carries a qubit” means that it
carries this local mode algebra, not that the global state is forced to
be a product of normalized vectors $q_e$.

Writing $\mathfrak h_K=\operatorname{span}\{|e\rangle:e\in K_1\}$ for
the one-particle edge space, the microscopic quantum carrier is

$$\mathcal H_K=\mathcal F_-(\mathfrak h_K)
=\Lambda^\bullet\mathfrak h_K
\cong\widehat\bigotimes_{e\in K_1}\mathcal H_e,$$

and a boundary state is a vector or density operator on $\mathcal H_K$.
It may be entangled. A one-particle color state
$a_\phi^\dagger|0\rangle$ and the nonseparable proton-spin sectors are
therefore native states, not exceptions to the ontology. For an isolated
occupied band with projector $P$, the corresponding quasi-free reference
state has covariance

$$\Gamma_{ef}=\langle a_f^\dagger a_e\rangle=P_{ef},\qquad
\langle n_e\rangle=P_{ee}.$$

Thus a per-edge Bloch vector or occupation is a derived
marginal/readout. The quasi-free state is a useful analytic baseline,
not a restriction: finitely many interacting non-Gaussian sectors are
represented explicitly by the lazy Fock construction of
Section <a href="#fock-space-as-an-inductive-limit-of-interactions"
data-reference-type="ref"
data-reference="fock-space-as-an-inductive-limit-of-interactions">11</a>.

Let

$$\partial_k:C_k(K)\longrightarrow C_{k-1}(K),\qquad
\partial_{k-1}\partial_k=0$$

be the oriented boundary maps and let `W_k(z)` be the metric weight on
`k`-chains. With the weighted adjoint

$$\partial_k^*=W_k^{-1}\partial_k^\dagger W_{k-1},$$

the degree-`k` Hodge Laplacian is

$$L_k=\partial_{k+1}\partial_{k+1}^*+\partial_k^*\partial_k.$$

In the positive metric regime this is self-adjoint in the `W_k` inner
product. In the signed Lorentzian regime it can be non-normal; then left
and right spectral frames and their biorthogonal condition numbers must
be reported rather than silently treating `L_k` as Hermitian.

The geometry evolves toward joint stationary points of the existing
Regge and Hodge functionals. In emergence mode, particle-specific
observables below are read after optimization and are not inserted as
target terms. Controlled synthesis mode may pin a carrier to test
realizability, but that is a separate experiment.

This operator stack sits on established foundations: Regge calculus
encodes piecewise-flat gravity in simplicial deficit angles (Regge
1961), discrete exterior calculus supplies metric-dependent
chain/cochain operators (Desbrun et al. 2005), and combinatorial Hodge
spectra on simplicial complexes have a developed spectral theory (Horak
and Jost 2013). Tessera’s proposal is not a replacement for those
constructions; it is a constrained use of them as the sole source of the
later particle readouts.

# A component is an exact static response vertex

Partition the `k`-cells of a connected component into interface cells
`B` and interior cells `I`, and block its Hodge operator as

$$L=
\begin{pmatrix}
L_{BB}&L_{BI}\\
L_{IB}&L_{II}
\end{pmatrix}.$$

In the positive self-adjoint regime, after projecting out incompatible
interior zero modes, minimization over the interior has the exact
solution

$$x_I^*=-L_{II}^{+}L_{IB}x_B,$$

and the exact effective boundary operator

$$\boxed{L_{\mathrm{eff}}
=L_{BB}-L_{BI}L_{II}^{+}L_{IB}}.$$

Here `+` denotes the Moore-Penrose inverse on the supported interior
subspace. For every compatible boundary value,

$$\min_{x_I}
\begin{pmatrix}x_B\\x_I\end{pmatrix}^{\!\dagger}
L
\begin{pmatrix}x_B\\x_I\end{pmatrix}
=x_B^\dagger L_{\mathrm{eff}}x_B.$$

This is the precise static, or zero-frequency, sense in which a
connected component can be replaced by a coarse response vertex. In a
Hermitian indefinite regime the same equation is a stationarity
condition, not a minimum. For a non-normal block it is simply block
elimination; solvability requires

$$L_{IB}x_B\perp\ker L_{II}^{\dagger}.$$

The plain Schur complement does *not* preserve the nonzero spectrum. For
a spectral parameter $\lambda$ such that $L_{II}-\lambda I$ is
invertible, define the exact Feshbach–Schur response

$$\boxed{F_B(\lambda)=L_{BB}-\lambda I
-L_{BI}(L_{II}-\lambda I)^{-1}L_{IB}}.$$

Then

$$\lambda\in\operatorname{spec}L
\quad\Longleftrightarrow\quad
0\in\operatorname{spec}F_B(\lambda),$$

with algebraic multiplicities in the supported finite-dimensional
sector. At an interior resonance the inverse is replaced only after
checking the compatibility condition
$L_{IB}x_B\perp\ker(L_{II}-\lambda I)^\dagger$ and retaining the
resonant interior modes explicitly. Thus harmonic response uses
$F_B(0)$, while a localized band centered at $\lambda_C$ uses
$F_B(\lambda)$ over a stated frequency window. A linear reduced
eigenproblem may instead retain interface constraint modes plus selected
fixed-interface modes using Craig–Bampton component-mode synthesis or
AMLS; that route is certified approximation whose error is controlled by
residuals and separation from discarded modes, not an exact spectral
identity (Craig and Bampton 1968; Bennighof and Lehoucq 2004; Bach et
al. 2003).

The effective blocks between coarse components become operator-valued
links. A harmonic or retained interior mode is not discarded; it becomes
an explicit stalk/fiber coordinate attached to the response vertex.

For graph Laplacians this is the classical Kron reduction by Schur
complement (Dörfler and Bullo 2013). Spectral graph reduction provides
related approximation guarantees when additional coarsening or
truncation is performed (Loukas 2019). The extension proposed here is to
apply static response reduction degree by degree to weighted Hodge
blocks, and shifted Feshbach or certified component-mode reduction to
nonzero bands, while retaining localized zero, resonant, and selected
interior modes as explicit fiber coordinates.

# Recursive spectral fibers

Let $P_\ell=\{C_v^\ell\}$ be an intrinsic partition at scale $\ell$ into
persistent connected components. At $\ell=0$ the object is the
microscopic simplicial complex $K_0$. After the first elimination the
honest coarse object is generally not another simplicial complex: it is
an operator-valued response network $\mathcal R_{\ell+1}$ whose vertices
carry vector spaces and whose links carry linear response blocks. A
cellular sheaf on the quotient graph is a natural realization when the
blocks admit compatible restriction-map factorization (Hansen and Ghrist
2019); otherwise Tessera retains the more general response network and
does not invent incidence maps that the reduction did not determine.

Within component `C`, choose an isolated localized spectral band and, in
the positive self-adjoint regime, a weighted orthonormal frame

$$\Phi_C=(\phi_1,\ldots,\phi_r),\qquad
\Phi_C^\dagger W_C\Phi_C=I_r.$$

The derived fiber is

$$E_C=\operatorname{Ran}\Phi_C.$$

In a Hermitian indefinite regime record the inertia of
$\Phi_C^\dagger W_C\Phi_C$ and normalize it to a signature matrix
$J_C=\operatorname{diag}(I_p,-I_q)$. Negative Krein signature is a
certificate, not an automatic identification with an antiparticle. In a
non-normal regime use matched right and left frames $\Phi_C,\Psi_C$ with
$\Psi_C^\dagger W_C\Phi_C=I$ and report both residuals and the frame
condition number.

It need not be a harmonic space and therefore need not be supported by a
hole. What it does require is a spectral gap, localization, and
persistence. A candidate component is accepted only if all of the
following remain stable across a stated range of scales:

- high modularity and low conductance relative to neighboring cuts;

- a localized spectral projector with stable rank;

- a nonzero band gap separating it from discarded modes;

- overlap with its predecessor and successor components;

- lifetime across multiple cobordism frames; and

- small external transport leakage.

This gives a type-stable hierarchy of response objects

$$\cdots\longrightarrow \mathcal R_2\longrightarrow
\mathcal R_1\longrightarrow K_0$$

in which a response vertex at one level resolves into a connected
microscopic component plus retained stalk coordinates at the next finer
level. “Self-similar” refers to closure of the response-network data
type, not to a claim that every reduced operator is a simplicial Hodge
Laplacian. A fractal-like pattern is permitted but not required:
measured scaling of module count, volume, boundary size, and spectral
gap decides whether the hierarchy is statistically self-similar.

Community objectives supply deterministic cluster candidates (Reichardt
and Bornholdt 2006), while network renormalization supplies tests for
genuine self-similarity rather than visual resemblance (Song, Havlin,
and Makse 2005). The partition is therefore a measured part of the
analysis: a recursively drawn pattern is not evidence of a fractal
unless its scaling observables survive a refinement window. The current
`ModularityOptimizer` uses Newman–Girvan modularity on a combinatorial
one-skeleton; it is a heuristic proposal generator that does not see
signed or complex Hodge weights and is subject to the modularity
resolution limit (Fortunato and Barthélemy 2007). Every accepted fiber
is therefore conditioned on independent, weight-aware gap, localization,
leakage, persistence, and refinement certificates.

<figure id="fig:recursive-step">

<figcaption>One recursive step. Persistent connected modules become
stalk-bearing vertices of an operator-valued response network. Static
response is preserved by the supported Schur complement; nonzero bands
use shifted Feshbach or certified component-mode reduction. Selected
internal modes remain attached as fibers, and a persistent supermodule
can be reduced again at the next scale.</figcaption>
</figure>

# Interactions and the expanding Hilbert space

Two operations must not be conflated. For the Cartesian product of chain
complexes `A` and `B`, the graded tensor differential is the exact rule

$$d_{A\widehat\otimes B}(a\otimes b)
=d_Aa\otimes b+(-1)^{\deg a}a\otimes d_Bb.$$

For a noninteracting product with product metric,

$$L_{A\widehat\otimes B}=L_A\otimes I+I\otimes L_B,$$

so one-particle eigenvalues add and eigenvectors tensor. This identity
is about a product complex, not about gluing two cobordisms.

Actual simplicial gluing is a pushout along a shared boundary. At the
one-particle level it produces a chain space assembled from direct sums
modulo boundary identifications (equivalently described by the relevant
Mayer–Vietoris sequence) and a block operator

$$L_{A\cup B}=
\begin{pmatrix}L_A&C_{AB}\\ C_{BA}&L_B\end{pmatrix}$$

in a basis adapted to the two interiors. The coupling blocks are induced
by the connecting simplices and shared-boundary constraints; they are
not a Kronecker interaction term.

The expanding Hilbert space follows after applying the fermionic Fock
functor to the one-particle space $\mathfrak h$. The exact identities
are

$$\mathcal F_-(\mathfrak h_A\oplus\mathfrak h_B)
\cong\mathcal F_-(\mathfrak h_A)\widehat\otimes
\mathcal F_-(\mathfrak h_B),$$

and

$$d\Gamma(L_A\oplus L_B)
=d\Gamma(L_A)\widehat\otimes I
+I\widehat\otimes d\Gamma(L_B).$$

For the coupling block,

$$d\Gamma(C_{AB}+C_{BA})
=\sum_{ij}(C_{AB})_{ij}a_{A,i}^\dagger a_{B,j}
+\mathrm{h.c.},$$

so geometric connections become hopping terms without adding a new
field. If the one-particle eigenvalues are $\lambda_1,\ldots,\lambda_M$,
then the free many-body spectrum is the set of occupation subset sums
$\sum_i n_i\lambda_i$, $n_i\in\{0,1\}$, rather than the one-particle
pairwise spectrum being relabeled as a Fock spectrum (Berezin 1966).

At the selected-fiber level, an interaction grows the carried space as

$$\mathcal H_{AB}=E_A\widehat\otimes E_B,$$

and a later interaction appends another factor. This is a statement
about state-space composition after second quantization, not the
topology of the glued chain complex. If $J_C$ embeds an abstract state
into the geometric carrier, exact amplitude preservation requires

$$J_C^\dagger W_CJ_C=I.$$

Tensor products preserve isometry exactly. If $G=J_C^\dagger W_CJ_C$ has
Gram defect $\varepsilon=\lVert G-I\rVert$, then

$$|a^\dagger Gb-a^\dagger b|
\leq \varepsilon\,\|a\|\,\|b\|,$$

and two tensor factors obey

$$\varepsilon_{AB}
\leq\varepsilon_A+\varepsilon_B+\varepsilon_A\varepsilon_B.$$

Thus the amplitude claim has an explicit, composable error budget.

Cobordism composition as a map between boundary state spaces is the
organizing idea of topological field theory (Atiyah 1988); the
general-boundary program makes the region/boundary assignment explicit
for quantum theory (Oeckl 2003), and categorical quantum mechanics
formalizes tensor composition and diagrammatic process semantics
(Abramsky and Coecke 2004). Tessera keeps only the parts that can be
realized by its finite simplicial carrier and tests the resulting map
numerically rather than assuming topological invariance.

# A triangle carries the exact color algebra

Consider the three edge-mode factors around an oriented triangle and
interpret `1>` as an occupied edge mode. Choosing an oriented ordering
$(e_1,e_2,e_3)$ identifies their graded tensor product with the exterior
algebra

$$(\mathbb C^2)^{\widehat\otimes 3}
\cong\Lambda^\bullet\mathbb C^3
=\mathbf1\oplus\mathbf3\oplus\overline{\mathbf3}\oplus\mathbf1.$$

The orientation of one triangle fixes the ordering up to a cyclic, hence
even, permutation, so the local wedge sign is unambiguous. Globally the
exterior algebra $\Lambda^\bullet\mathfrak h_K$ and the CAR are
intrinsic; only a compilation into tensor-product qubits or bitsets
needs a deterministic mode order and the corresponding permutation
parity. A Kasteleyn orientation is useful for two-dimensional
surface-dimer Pfaffians but is not required to define this abstract Fock
space (Cimasoni and Reshetikhin 2007). A genuine continuum spinor
interpretation is a separate question addressed by the rotation
certificate below.

The sectors have occupation number $N=0,1,2,3$:

| Sector                 | Basis dimension | Color interpretation       | Fermion parity |
|:-----------------------|----------------:|:---------------------------|---------------:|
| $\Lambda^0\mathbb C^3$ |               1 | vacuum                     |           even |
| $\Lambda^1\mathbb C^3$ |               3 | fundamental color triplet  |            odd |
| $\Lambda^2\mathbb C^3$ |               3 | antisymmetric anti-triplet |           even |
| $\Lambda^3\mathbb C^3$ |               1 | color singlet              |            odd |

Let $a_i^\dagger,a_i$ be the exterior creation and contraction
operators. They satisfy the canonical anticommutation relations exactly:

$$\{a_i,a_j\}=0,\qquad
\{a_i^\dagger,a_j^\dagger\}=0,\qquad
\{a_i,a_j^\dagger\}=\delta_{ij}.$$

On the one-occupation sector, the bilinears

$$E_{ij}=a_i^\dagger a_j$$

satisfy

$$[E_{ij},E_{k\ell}]=\delta_{jk}E_{i\ell}-\delta_{i\ell}E_{kj}.$$

The six Hermitian off-diagonal combinations together with

$$H_1=E_{11}-E_{22},\qquad
H_2=\frac{E_{11}+E_{22}-2E_{33}}{\sqrt3}$$

are the eight generators of `su(3)`. Thus the triangle does not merely
hold three phases: its one-particle edge sector carries the fundamental
representation, its two-particle sector carries the dual representation,
and its traceless bilinears carry the adjoint octet.

The triplet description of quarks and the three-quark construction of
baryons originate with the quark model (Gell-Mann 1964); the additional
color triplet was introduced to resolve the statistics and
state-counting problem (Han and Nambu 1965; Greenberg 1964). The claim
here is narrower and new: Tessera’s three oriented edge modes would
provide a geometric carrier of the same representation content, not a
derivation of QCD from the combinatorics alone.

<figure id="fig:color-exterior">

<figcaption>Exact representation content of three oriented edge-mode
factors. The exterior sectors and their parity are algebraic identities.
Interpreting the rank-three odd sector as quark color and the top wedge
as a baryon color singlet is the physical hypothesis to be
tested.</figcaption>
</figure>

<div id="geometric-normalization">

## Geometric normalization

</div>

For the stored complex squared lengths $z_i=\rho_i e^{i\theta_i}$ on the
three oriented edges, define

$$c_i=\frac{z_i}{\sqrt{|z_1|^2+|z_2|^2+|z_3|^2}},\qquad
|c\rangle=\sum_{i=1}^3c_i|i\rangle.$$

Then $\langle c\vert c\rangle=1$. Constraining the perimeter to one is a
valid geometric scale gauge, but it is an `L^1` condition and does not
replace the $L^2$ Hilbert normalization. Normalized pure color states
form $\mathbb{CP}^2$; $SU(3)$ is the transformation group, not the
surface of the triangle itself.

<div id="the-existing-omega-phase-pattern">

## The existing omega phase pattern

</div>

Let $\omega=e^{2\pi i/3}$. The exact Fourier frame

$$F_3=\frac1{\sqrt3}
\begin{pmatrix}
1&1&1\\
1&\omega&\omega^2\\
1&\omega^2&\omega
\end{pmatrix}$$

is unitary. The existing pattern $(1,\omega,\omega^2)/\sqrt3$ is
therefore one color basis vector, not by itself the whole color fiber.
Its cyclic orbit supplies an exact orthonormal triad.

# Color transport and Wilson loops without a new gauge field

Let $T_{AB}$ be the chain-level transfer already induced by connecting
simplices from component $B$ to component $A$. In positive self-adjoint
local spectral frames of common rank $r$, the raw fiber map is

$$M_{AB}=\Phi_A^\dagger W_A T_{AB}\Phi_B.$$

Its departure from an isometry is a physical leakage diagnostic:

$$\eta_{AB}=\|M_{AB}^\dagger M_{AB}-I\|.$$

When the selected band is isolated, $M_{AB}$ has full numerical rank,
and $\eta_{AB}$ is small, take the polar unitary

$$V_{AB}=M_{AB}(M_{AB}^\dagger M_{AB})^{-1/2}\in U(r).$$

Under a change of local spectral frame, $\Phi_A\mapsto\Phi_Ag_A$ and
$\Phi_B\mapsto\Phi_Bg_B$,

$$V_{AB}\longmapsto g_A^\dagger V_{AB}g_B.$$

Consequently the full closed holonomy and its normalized trace transform
by conjugation at the base point:

$$H(\gamma)=\prod_{(AB)\in\gamma}V_{AB},\qquad
W_{U(r)}(\gamma)=\frac1r\operatorname{Tr}H(\gamma).$$

At rank three there is no globally single-valued operation
$V\mapsto V/(\det V)^{1/3}$: the cube root is $\mathbb Z_3$-ambiguous.
The faithful derived datum is therefore retained as

$$V_{AB}\in U(3),\qquad
\delta_{AB}=\det V_{AB}\in U(1),\qquad
[V_{AB}]\in PU(3)\cong SU(3)/\mathbb Z_3.$$

A local $SU(3)$ lift $\widetilde U_{AB}=\delta_{AB}^{-1/3}V_{AB}$ may be
followed continuously along a path after fixing a base branch, but its
accumulated center sector must be reported. Fundamental Wilson traces
use the full $U(3)$ holonomy or an explicitly lifted path;
adjoint/projective Wilson loops are center-blind and require no branch.
This turns the former ambiguity into two measured sectors rather than
silently discarding one.

The determinant line also supplies a possible oriented flux readout. For
a closed, full-rank world-tube family $V(t)$,

$$\nu=\frac{1}{2\pi}\oint d\arg\det V(t)\in\mathbb Z$$

is homotopy-invariant while the gap and rank remain open, and changes
sign when the tube orientation is reversed. Identifying $B=\nu/3$ for an
accepted quark tube is a proposed physical interpretation, not a group
identity. Conservation under pair creation is exact only for a
continuous conjugate-pair homotopy with no determinant zero or boundary
flux.

For non-normal bands the correct overlap is biorthogonal,

$$M_{AB}=\Psi_A^\dagger W_A T_{AB}\Phi_B,
\qquad \Psi_C^\dagger W_C\Phi_C=I.$$

It is generally a $GL(r,\mathbb C)$ transport, not a unitary one; left
and right residuals, singular values, and frame condition numbers are
part of the observable. In an indefinite Hermitian sector the Krein
inertia is reported and a pseudo-unitary reduction is attempted only
when the two signatures agree. No $U(r)$ or $SU(3)$ Wilson value is
emitted by silently applying the positive-metric formula outside its
domain.

Polar normalization must never conceal a bad fiber assignment: every
accepted holonomy is reported together with $\eta_{AB}$, rank/singular
value thresholds, endpoint band gaps, and frame conditioning.

This construction is the discrete spectral-frame analogue of Berry
transport (Berry 1984) and its non-Abelian Wilczek-Zee generalization
(Wilczek and Zee 1984). Overlap matrices are also a standard route to
gauge-covariant lattice observables (Fukui, Hatsugai, and Suzuki 2005).
Wilson loops themselves are foundational lattice gauge observables
(Wilson 1974). What is specific here is that the link matrix is not
independently assigned: it is reconstructed from neighboring Hodge
frames and accompanied by a leakage certificate.

# Quarks as modular clusters

A *quark candidate* is proposed to be a component `Q` satisfying all of
the following derived conditions:

The abstract rank-three band must first be anchored to oriented
two-simplices. For an oriented triangle $\tau\subset Q$, let
$R_\tau:C_1(Q)\to\mathbb C^3$ restrict a one-chain to the three ordered
boundary edges, including their incidence signs. For a rank-three frame
$\Phi_Q$, define the gauge-invariant anchor score

$$a_Q^2=\sum_{\tau\subset Q}w_\tau
\left|\det(R_\tau\Phi_Q)\right|^2,
\qquad w_\tau\ge0,\quad\sum_\tau w_\tau=1.$$

The phases of the nonzero determinants give a local determinant-line
trivialization; their coherence on overlapping triangles is recorded
separately. Concentration on one triangle is a sufficient special case,
not a requirement: an extended fiber may be anchored by an atlas of
oriented faces. Because every 2-simplex has exactly three boundary
edges, this anchor is independent of the ambient spectral dimension.

1.  `Q` is a persistent high-modularity cluster, not a prescribed
    region.

2.  Its selected color fiber has stable rank three.

3.  Its triangle-anchor score and determinant-line coherence are stable.

4.  It occupies an odd exterior sector.

5.  Its color transport has bounded leakage over its lifetime.

6.  Its determinant line has an accepted oriented winding $\nu=\pm1$;
    reversing the tube yields the dual color representation and an
    antiquark.

7.  Its total spectral fingerprint is stable under refinement and vertex
    relabeling.

The distinction between an antiquark and the $\Lambda^2\mathbb C^3$
anti-triplet of two quarks is made by determinant-line orientation and
total occupation, not color alone. Assigning $B=\nu/3$ is accepted only
when the winding certificate above exists. A forward/reverse pair then
has zero total winding under a gap-preserving conjugate homotopy;
without that certificate baryon number remains unknown rather than being
inserted by definition.

Flavor and electric charge are not assumed as hidden labels. The
conservative hypothesis is that two stable subclasses of the same
cluster fiber provide an isospin doublet. On such a doublet, the
measured orientation flux supplies baryon number and the standard
relation

$$Q=I_3+\frac{B}{2}$$

gives $Q_u=+2/3$ and $Q_d=-1/3$. This is a proposed identification, not
yet a derivation: it succeeds only if an unlabeled two-dimensional
spectral band emerges, is transported coherently, and its Gauss-flux
readout agrees with those values.

# Fermion statistics from simplicial orientation

The graded interchange law is

$$\tau(a\widehat\otimes b)
=(-1)^{F_aF_b}b\widehat\otimes a.$$

Two odd clusters therefore acquire a minus sign on exchange, while an
even composite does not. Parity adds modulo two:

| Object             | Odd constituents | Composite parity |
|:-------------------|-----------------:|-----------------:|
| quark or antiquark |                1 |              odd |
| meson $q\bar q$    |                2 |             even |
| diquark $qq$       |                2 |             even |
| baryon $qqq$       |                3 |              odd |

Pauli exclusion is the exact exterior-algebra identity

$$\|v_1\wedge\cdots\wedge v_n\|^2
=\det[\langle v_i,v_j\rangle].$$

If two complete one-particle modes coincide, the determinant and the
state vanish. The “complete” qualifier prevents double-counting signs:
color, spin, flavor, space, and component support are wedged once as one
mode. One must not multiply an extra fermion sign by the sign already
present in the color epsilon tensor.

<div id="a-label-independent-exchange-experiment">

## A label-independent exchange experiment

</div>

Let $\Phi_t$ be an orthonormal frame for the isolated odd subspace at
frame $t$. Define the parallel transport

$$R_t=\operatorname{polar}(\Phi_{t+1}^\dagger W_t\Phi_t),\qquad
U_\gamma=R_{T-1}\cdots R_0.$$

The raw determinant line contains both permutation statistics and an
ordinary Abelian Berry phase:

$$\chi_{\mathrm{raw}}(\gamma)=\det U_\gamma.$$

It is therefore not expected to equal $\pm1$ on a generic geometric
loop. Let $\gamma_0$ be a non-exchanging reference motion with the same
geometric footprint, timing, and local frame convention. The
interferometric exchange character is

$$\widehat\chi_F(\gamma)=
\frac{\det U_\gamma}{\det U_{\gamma_0}}.$$

The proposed dynamical test is

$$\widehat\chi_F(\text{single exchange})=-1,\qquad
\widehat\chi_F(\text{double exchange})=+1.$$

As an independent structural cross-check, persistent component matching
extracts the permutation $P_\gamma$ of localized odd blocks and reports
$\operatorname{sgn}P_\gamma$ together with the norm of the residual
in-block motion after comparison with the reference loop. The algebraic
wedge sign is exact; the interferometric holonomy is the dynamical
certificate.

A physical $2\pi$ rotation uses the same reference normalization. If an
emergent manifold-like regime supplies tangent frames, a continuum
spinor claim additionally requires a lift of the frame holonomy from
$SO(d)$ to $\operatorname{Spin}(d)$; obstruction by the second
Stiefel–Whitney class is then a falsification certificate (Lawson and
Michelsohn 1989). This requirement concerns the physical spin lift, not
the existence of the abstract CAR/Fock algebra. The spin-statistics
comparison is cleanest as

$$\widehat\chi_F(\text{exchange})
\widehat\chi_F(2\pi\text{ rotation})^{-1}=+1,$$

with each factor separately near $-1$.

Configuration-space topology already explains how exchange classes can
carry quantum phases (Laidlaw and DeWitt 1971) and, in two dimensions,
more general statistics (Leinaas and Myrheim 1977). The Tessera proposal
uses this precedent only as a diagnostic template. The minus sign from
the graded exterior algebra is exact; the claim that an actual geometric
exchange cobordism realizes the corresponding determinant holonomy
remains an experiment.

# Fock space as an inductive limit of interactions

For $M$ oriented fermionic edge modes,

$$\widehat\bigotimes_{m=1}^{M}\mathbb C^2
\cong\Lambda^\bullet\mathbb C^M
=\bigoplus_{n=0}^{M}\Lambda^n\mathbb C^M,$$

and the dimension identity is exact:

$$2^M=\sum_{n=0}^{M}\binom{M}{n}.$$

The exterior algebra is canonical as a functor of the one-particle
space. Writing it as a literal ordered qubit tensor product, or
implementing creation operators by Jordan–Wigner/bitset strings,
requires a chosen mode order. Tessera derives a deterministic order from
oriented component lineage and applies the parity of every reordering;
all reported observables must be invariant under relabeling plus that
induced unitary.

Adding a new noninteracting mode uses the vacuum embedding

$$\iota_M:\mathcal H_M\hookrightarrow\mathcal H_{M+1},\qquad
\iota_M(\psi)=\psi\widehat\otimes|0\rangle.$$

The infinite Fock space is the direct limit

$$\mathcal F=\varinjlim(\mathcal H_M,\iota_M).$$

This makes the infinite expansion precise without ever allocating an
infinite array. At every finite simulation time only finitely many modes
have interacted. Consistency requires

$$\|\iota_MU_M-U_{M+1}\iota_M\|\longrightarrow0$$

over a refinement sequence.

Bosonic gauge excitations need not add a new local oscillator. The
traceless even bilinears

$$a_i^\dagger a_j-\frac13\delta_{ij}N$$

span the color octet in $\mathbf3\otimes\overline{\mathbf3}
=\mathbf1\oplus\mathbf8$ and have even fermion parity. Arbitrarily many
such collective excitations are represented by adding more microscopic
modes at finer resolution. Each finite edge-mode factor remains
two-dimensional.

This scale-by-scale state growth is adjacent to entanglement
renormalization, where local Hilbert data are reorganized across layers
before truncation (Vidal 2007). The distinction is material: Tessera
uses static/shifted response reduction plus an inductive vacuum
embedding, and it must certify compatibility between successive finite
spaces rather than assume a fixed bond dimension.

## Occupation exterior algebra is not automatically Kähler–Dirac

The exterior algebra above is over the one-particle *mode space*; its
degree is occupation number. A Kähler–Dirac field instead lives on the
inhomogeneous differential-form/cochain space $\bigoplus_k C^k(K)$ and
is acted on by $d-d^*$ (or a related Dirac square root). These
constructions share exterior-algebra notation but are not the same
operator or grading. Consequently the present model does not inherit
lattice taste multiplicity merely from using
$\Lambda^\bullet\mathfrak h_K$.

If a later Tessera model promotes its one-particle field to all cochain
degrees and uses the Kähler–Dirac operator, the known flat
four-dimensional decomposition into four Dirac spinors becomes an
expected spectrum diagnostic, not an unexplained bug (Becher and Joos
1982; Butt et al. 2021). Any observed near-fourfold cluster in the
present model is reported as an empirical degeneracy until that stronger
operator identification is made.

# The proton as the maximally informative baryon

The proton is chosen because, beyond generic baryon structure, it
demands a nontrivial flavor pattern, electric charge, spin, and
experimentally meaningful form factors.

Let three persistent quark components $A,B,C$ have color frames and
normalized color columns $c_A,c_B,c_C$. The invariant color volume is

$$S_{ABC}=\epsilon_{ijk}c_A^ic_B^jc_C^k
=\det[c_A\ c_B\ c_C].$$

Under a common $g\in SU(3)$, $S\mapsto\det(g)S=S$. Its squared magnitude
is the Gram determinant

$$|S_{ABC}|^2=\det(C^\dagger C)\in[0,1].$$

The value one means the three color directions form an orthonormal
frame. Their normalized wedge is then the unique $\Lambda^3\mathbb C^3$
singlet. The proposed proton certificate is the conjunction:

- three persistent odd rank-three quark clusters with accepted
  triangle-anchor certificates;

- one persistent bound supercluster containing them;

- normalized color wedge with $|S_{ABC}|^2\approx1$ and vanishing net
  color flux;

- flavor spectrum with the `uud` occupation pattern;

- oriented baryon flux $B=1$;

- Gauss flux $Q=+1$;

- total-space spin holonomy $J^2=3/4$, a reference-normalized
  $2\pi\mapsto-1$, and, where applicable, an accepted spin lift;

- finite radius and stable spectral mass/form-factor readouts; and

- stability of every dimensionless certificate under refinement.

None of these conditions should be included as an emergence target. The
proton is found only if the base geometric optimization produces a
component satisfying them. Targeted runs remain valuable as existence
and obstruction experiments, but must be labeled as synthesis rather
than emergence.

<figure id="fig:emergence-firewall">

<figcaption>No-feedback emergence protocol. Only the base geometric
objective drives optimization. Cluster, fiber, color, exchange, and
baryon observables are computed from accepted snapshots and cannot feed
back into the objective in an emergence run. Targeted synthesis is a
separate, explicitly labeled mode.</figcaption>
</figure>

# The master recursive construction

Let $\mathcal R_0(\lambda)=L_0-\lambda I$ denote the microscopic
one-particle response pencil. At every scale:

$$\boxed{
\begin{aligned}
P_\ell&=\operatorname{PersistentPartition}(\mathcal R_\ell),\\
E_v^{\ell+1}&=\text{certified isolated subspace of }C_v^\ell,\\
\mathcal R_{\ell+1}(\lambda)
&=\operatorname{Feshbach}_{P_\ell}(\mathcal R_\ell(\lambda)),\\
V_{vw}^{\ell+1}&=\operatorname{Polar}_{r_v}
\big((\Phi_v^\ell)^\dagger W T_{vw}\Phi_w^\ell\big),\\
\mathcal H_{\ell+1}&=\mathcal F_-
\!\left(\bigoplus_v E_v^{\ell+1}\right).
\end{aligned}}$$

At $\lambda=0$ the response step is the exact supported static Schur
complement. For a nonzero band it is the exact energy-dependent pencil;
a cached linear $\mathcal R_{\ell+1}$ is an AMLS/component-mode
surrogate with a declared frequency window and residual. The transport
rank is generic; only an anchored accepted rank-three fiber receives the
color interpretation. This recursion supplies the response network,
retained stalk, derived transport, and expanding state space without
claiming that every coarse level is literally a new simplicial complex.

# Exactness and performance principles

The simulation should prefer an exact structural identity over a general
dense numerical operation whenever both compute the same object:

- use sparse static and shifted Schur/Feshbach solves, not explicit
  dense inverses, and use AMLS when a reusable linear band surrogate is
  needed;

- use Künneth sums only for actual product complexes, and occupation
  subset sums for $d\Gamma(L)$, not diagonalization of an eager Fock
  matrix;

- use exterior bit parity for exchange signs, not sampled phases;

- use exact $3\times3$ determinants and the fixed $F_3$ frame only after
  a rank-three band passes its triangle-anchor certificate;

- use analytic Regge/Hodge derivatives and Wirtinger gradients, not
  finite differences;

- use Smith normal form for integer homology and a spectral threshold
  only as a cross-check;

- use matrix-determinant/Woodbury updates for local cobordism changes;

- cache component factorizations and invalidate only affected stars;

- keep tensor products lazy and block-sparse by occupation/parity;

- use $U(r)$ polar transport at generic rank, retaining determinant-line
  and projective/center data at $r=3$; and

- attach frequency window, residual, gap, leakage, signature, and
  condition-number certificates to every iterative eigensolve or
  reduction.

The exact route is not only faster. It prevents a numerical tolerance
from becoming an undocumented physical postulate.

# Prior art and boundary of novelty

No single cited work establishes the full recursive spectral-fiber
proposal. The construction is a synthesis of several mature ideas, and
its novelty should be evaluated at the joins.
Table <a href="#tab:prior-art" data-reference-type="ref"
data-reference="tab:prior-art">1</a> states the boundary explicitly.

<div id="tab:prior-art">

| Topic                                 | Established prior art                                                                                                                                                                                                                                                                                                                                                           | Additional claim made here                                                                                                                                                                                                                                            |
|:--------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Topic                                 | Established prior art                                                                                                                                                                                                                                                                                                                                                           | Additional claim made here                                                                                                                                                                                                                                            |
| Simplicial geometry and Hodge spectra | Regge curvature, discrete exterior calculus, and combinatorial Laplace spectra (Regge 1961; Desbrun et al. 2005; Horak and Jost 2013).                                                                                                                                                                                                                                          | Use one jointly optimized Regge-Hodge complex as the only microscopic carrier for geometry and quantum readouts.                                                                                                                                                      |
| Coarse response and recursive modules | Kron/Schur reduction, Feshbach maps, component-mode synthesis/AMLS, cellular sheaf Laplacians, modular communities, and self-similar network renormalization (Dörfler and Bullo 2013; Bach et al. 2003; Craig and Bampton 1968; Bennighof and Lehoucq 2004; Hansen and Ghrist 2019; Reichardt and Bornholdt 2006; Fortunato and Barthélemy 2007; Song, Havlin, and Makse 2005). | Treat a persistent component as a static response vertex; retain nonzero bands through shifted or certified component-mode reduction; recurse in operator-valued response networks, using a sheaf realization only when its factorization is certified.               |
| Geometric gauge transport             | Berry and Wilczek-Zee holonomy, overlap-based lattice links, connection Laplacians/vector diffusion, and Wilson loops (Berry 1984; Wilczek and Zee 1984; Fukui, Hatsugai, and Suzuki 2005; Singer and Wu 2012; Wilson 1974).                                                                                                                                                    | Derive $U(r)$ transport from component frames; at anchored rank three retain the determinant line, projective $SU(3)/\mathbb Z_3$ class, and any chosen center lift, assigning no independent gauge link variable.                                                    |
| Color and fermion structure           | Quark/color triplets, exterior Fock/second quantization, topological exchange phases, and spin structures (Gell-Mann 1964; Han and Nambu 1965; Greenberg 1964; Berezin 1966; Laidlaw and DeWitt 1971; Leinaas and Myrheim 1977; Lawson and Michelsohn 1989).                                                                                                                    | Realize $\mathbf1\oplus\mathbf3\oplus\overline{\mathbf3}\oplus\mathbf1$ on three oriented edge modes, anchor abstract rank-three fibers to oriented faces, and test exchange by a Berry-cancelled determinant-line interferometer plus structural permutation parity. |
| Scale composition and boundaries      | TQFT cobordisms, general-boundary state assignments, categorical tensor composition, second quantization, and entanglement renormalization (Atiyah 1988; Oeckl 2003; Abramsky and Coecke 2004; Berezin 1966; Vidal 2007).                                                                                                                                                       | Keep simplicial gluing at the one-particle level, then build finite Fock stages functorially and require vacuum-embedding compatibility under refinement.                                                                                                             |
| Kähler–Dirac boundary                 | Differential-form fermions and their taste structure (Becher and Joos 1982; Butt et al. 2021).                                                                                                                                                                                                                                                                                  | Do not infer Kähler–Dirac tastes from occupation exterior algebra; test for them only if the one-particle field is promoted to inhomogeneous cochains with a Kähler–Dirac operator.                                                                                   |
| Spectral spacetime                    | Diffusion spectral dimension on ensembles of simplicial geometries (Ambjørn, Jurkiewicz, and Loll 2005).                                                                                                                                                                                                                                                                        | Test whether many interacting Tessera cobordisms yield a stable four-dimensional spectral window while simultaneously supporting the particle certificates.                                                                                                           |

Established ingredients and the additional Tessera claim.

</div>

<figure id="fig:prior-art-map">

<figcaption>Relationship to prior art. The blue inputs are established
research programs; the green center is the proposed Tessera synthesis;
the amber outputs are new physical identifications and must be validated
independently. An arrow denotes conceptual inheritance, not a proof of
the downstream claim.</figcaption>
</figure>

# Falsification program

The formulation fails, or must be narrowed, if any of the following
persists under refinement and tighter numerical certification:

1.  **No persistent rank-three clusters.** High-modularity components
    appear, but their localized fiber rank or spectral gap is unstable.

2.  **No oriented color anchor.** A rank-three band appears, but its
    projected alternating volume on oriented triangles vanishes or
    drifts.

3.  **No faithful coarse response.** Schur-reduced components fail to
    reproduce static response, or shifted/AMLS reduction fails over its
    declared frequency band, within the stated residual.

4.  **No derived gauge covariance.** Wilson values depend on the local
    spectral frame after leakage is controlled, or the determinant and
    $\mathbb Z_3$ center sectors cannot be made path-consistent.

5.  **No fermion holonomy.** The Berry-cancelled exchange ratio or
    structural permutation sign does not give `-1`, or the verdict
    changes under relabeling.

6.  **No spinor rotation.** Exchange works but the reference-normalized
    $2\pi$ physical rotation does not give `-1`; in a manifold-like
    continuum claim, failure of a consistent spin lift is also decisive.

7.  **No inductive compatibility.** Adding vacuum modes changes
    already-computed amplitudes by a nonvanishing amount.

8.  **No unforced baryon.** Targeted synthesis can build the
    certificates, but the stationary geometric ensemble never produces
    them without a proton-specific term.

9.  **No continuum stability.** Dimensionless color, parity, charge,
    spin, and amplitude certificates drift rather than converge with
    refinement.

10. **Unexpected multiplicity.** A robust flavor/taste degeneracy is
    neither predicted by the stated one-particle operator nor stable
    enough to be promoted to an emergent flavor mechanism.

Holes may re-emerge and may correlate with some phases, but no claim in
this paper depends on them doing so.

# Conclusion

The revised geometry is economical but more precise: an edge carries a
two-level mode, not an independently stored pure state; a quark
candidate is a modular spectral component whose rank-three band is
anchored to oriented faces; its transport is a certified $U(3)$ overlap
with retained determinant-line and projective color data; and its
fermionic sign is the exterior grading, checked dynamically only after
cancelling ordinary Berry phase. Simplicial gluing constructs the
one-particle operator and second quantization constructs the expanding
Fock state. Three accepted components form a baryon through their
normalized color wedge; the proton is the sharpest test because it also
demands the correct determinant-line flux, charge, flavor, and
reference-normalized spin response.

The exact claims are now limited to their proper domains: static Schur
response, energy-dependent Feshbach isospectrality, exterior/CAR
algebra, second-quantized direct-sum composition, and gauge covariance
of accepted transport. What remains genuinely open is whether Tessera’s
unforced Regge-Hodge dynamics produces the required anchored clusters,
low-leakage holonomies, determinant windings, and spin lift. A tempting
Hellmann–Feynman/envelope argument does not by itself make the first
variation of transport Gram defect vanish at a Regge–Hodge stationary
point, because the defect is not the optimized functional. Tessera will
therefore measure that correlation as a conjectural scaling law rather
than cite stationarity as a theorem.

# Repository evidence

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

- External-review disposition and exact-claim ledger,
  [`referee response`](recursive_spectral_fibers_referee_response.md).

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-abramsky2004categorical" class="csl-entry">

Abramsky, Samson, and Bob Coecke. 2004. “A Categorical Semantics of
Quantum Protocols.” In *Proceedings of the 19th Annual IEEE Symposium on
Logic in Computer Science*, 415–25.
<https://doi.org/10.1109/LICS.2004.1319636>.

</div>

<div id="ref-ambjorn2005spectral" class="csl-entry">

Ambjørn, Jan, Jerzy Jurkiewicz, and Renate Loll. 2005. “Spectral
Dimension of the Universe.” *Physical Review Letters* 95: 171301.
<https://doi.org/10.1103/PhysRevLett.95.171301>.

</div>

<div id="ref-atiyah1988tqft" class="csl-entry">

Atiyah, Michael F. 1988. “Topological Quantum Field Theories.”
*Publications Mathématiques de l’IHÉS* 68: 175–86.
<https://doi.org/10.1007/BF02698547>.

</div>

<div id="ref-bach2003feshbach" class="csl-entry">

Bach, Volker, Thomas Chen, Jürg Fröhlich, and Israel Michael Sigal.
2003. “Smooth Feshbach Map and Operator-Theoretic Renormalization Group
Methods.” *Journal of Functional Analysis* 203 (1): 44–92.
<https://doi.org/10.1016/S0022-1236(03)00057-0>.

</div>

<div id="ref-becher1982dirackahler" class="csl-entry">

Becher, Peter, and Hans Joos. 1982. “The Dirac–kähler Equation and
Fermions on the Lattice.” *Zeitschrift für Physik C* 15: 343–65.
<https://doi.org/10.1007/BF01614426>.

</div>

<div id="ref-bennighof2004amls" class="csl-entry">

Bennighof, Jeffrey K., and Richard B. Lehoucq. 2004. “An Automated
Multilevel Substructuring Method for Eigenspace Computation in Linear
Elastodynamics.” *SIAM Journal on Scientific Computing* 25 (6):
2084–2106.

</div>

<div id="ref-berezin1966second" class="csl-entry">

Berezin, Felix A. 1966. *The Method of Second Quantization*. New York:
Academic Press.

</div>

<div id="ref-berry1984phase" class="csl-entry">

Berry, Michael V. 1984. “Quantal Phase Factors Accompanying Adiabatic
Changes.” *Proceedings of the Royal Society of London A* 392 (1802):
45–57. <https://doi.org/10.1098/rspa.1984.0023>.

</div>

<div id="ref-butt2022kahler" class="csl-entry">

Butt, Nouman, Simon Catterall, Arnab Pradhan, and Goksu Can Toga. 2021.
“Anomalies and Symmetric Mass Generation for kähler–Dirac Fermions.”
*Physical Review D* 104: 094504.
<https://doi.org/10.1103/PhysRevD.104.094504>.

</div>

<div id="ref-cimasoni2007dimers" class="csl-entry">

Cimasoni, David, and Nicolai Reshetikhin. 2007. “Dimers on Surface
Graphs and Spin Structures. i.” *Communications in Mathematical Physics*
275: 187–208. <https://doi.org/10.1007/s00220-007-0302-7>.

</div>

<div id="ref-craig1968coupling" class="csl-entry">

Craig, Jr., Roy R., and Mervyn C. C. Bampton. 1968. “Coupling of
Substructures for Dynamic Analyses.” *AIAA Journal* 6 (7): 1313–19.
<https://doi.org/10.2514/3.4741>.

</div>

<div id="ref-desbrun2005dec" class="csl-entry">

Desbrun, Mathieu, Anil N. Hirani, Melvin Leok, and Jerrold E. Marsden.
2005. “Discrete Exterior Calculus.”
<https://arxiv.org/abs/math/0508341>.

</div>

<div id="ref-dorfler2013kron" class="csl-entry">

Dörfler, Florian, and Francesco Bullo. 2013. “Kron Reduction of Graphs
with Applications to Electrical Networks.” *IEEE Transactions on
Circuits and Systems I: Regular Papers* 60 (1): 150–63.
<https://doi.org/10.1109/TCSI.2012.2215780>.

</div>

<div id="ref-fortunato2007resolution" class="csl-entry">

Fortunato, Santo, and Marc Barthélemy. 2007. “Resolution Limit in
Community Detection.” *Proceedings of the National Academy of Sciences*
104 (1): 36–41. <https://doi.org/10.1073/pnas.0605965104>.

</div>

<div id="ref-fukui2005chern" class="csl-entry">

Fukui, Takahiro, Yasuhiro Hatsugai, and Hiroshi Suzuki. 2005. “Chern
Numbers in Discretized Brillouin Zone: Efficient Method of Computing
(Spin) Hall Conductances.” *Journal of the Physical Society of Japan* 74
(6): 1674–77. <https://doi.org/10.1143/JPSJ.74.1674>.

</div>

<div id="ref-gellmann1964quark" class="csl-entry">

Gell-Mann, Murray. 1964. “A Schematic Model of Baryons and Mesons.”
*Physics Letters* 8 (3): 214–15.
<https://doi.org/10.1016/S0031-9163(64)92001-3>.

</div>

<div id="ref-greenberg1964paraquark" class="csl-entry">

Greenberg, O. W. 1964. “Spin and Unitary-Spin Independence in a
Paraquark Model of Baryons and Mesons.” *Physical Review Letters* 13:
598–602. <https://doi.org/10.1103/PhysRevLett.13.598>.

</div>

<div id="ref-han1965color" class="csl-entry">

Han, Moo-Young, and Yoichiro Nambu. 1965. “Three-Triplet Model with
Double SU(3) Symmetry.” *Physical Review* 139: B1006–10.
<https://doi.org/10.1103/PhysRev.139.B1006>.

</div>

<div id="ref-hansen2019sheaves" class="csl-entry">

Hansen, Jakob, and Robert Ghrist. 2019. “Toward a Spectral Theory of
Cellular Sheaves.” *Journal of Applied and Computational Topology* 3:
315–58. <https://doi.org/10.1007/s41468-019-00038-7>.

</div>

<div id="ref-horak2013spectra" class="csl-entry">

Horak, Danijela, and Jürgen Jost. 2013. “Spectra of Combinatorial
Laplace Operators on Simplicial Complexes.” *Advances in Mathematics*
244: 303–36. <https://doi.org/10.1016/j.aim.2013.05.005>.

</div>

<div id="ref-laidlaw1971indistinguishable" class="csl-entry">

Laidlaw, Michael G. G., and Cécile Morette DeWitt. 1971. “Feynman
Functional Integrals for Systems of Indistinguishable Particles.”
*Physical Review D* 3: 1375–78.
<https://doi.org/10.1103/PhysRevD.3.1375>.

</div>

<div id="ref-lawson1989spin" class="csl-entry">

Lawson, Jr., H. Blaine, and Marie-Louise Michelsohn. 1989. *Spin
Geometry*. Princeton: Princeton University Press.

</div>

<div id="ref-leinaas1977identical" class="csl-entry">

Leinaas, Jon M., and Jan Myrheim. 1977. “On the Theory of Identical
Particles.” *Il Nuovo Cimento B* 37: 1–23.
<https://doi.org/10.1007/BF02727953>.

</div>

<div id="ref-loukas2019graph" class="csl-entry">

Loukas, Andreas. 2019. “Graph Reduction with Spectral and Cut
Guarantees.” *Journal of Machine Learning Research* 20 (116): 1–42.
<https://jmlr.org/papers/v20/18-680.html>.

</div>

<div id="ref-oeckl2003boundary" class="csl-entry">

Oeckl, Robert. 2003. “A ‘General Boundary’ Formulation for Quantum
Mechanics and Quantum Gravity.” *Physics Letters B* 575: 318–24.
<https://doi.org/10.1016/j.physletb.2003.08.043>.

</div>

<div id="ref-regge1961" class="csl-entry">

Regge, Tullio. 1961. “General Relativity Without Coordinates.” *Il Nuovo
Cimento* 19: 558–71. <https://doi.org/10.1007/BF02733251>.

</div>

<div id="ref-reichardt2006community" class="csl-entry">

Reichardt, Jörg, and Stefan Bornholdt. 2006. “Statistical Mechanics of
Community Detection.” *Physical Review E* 74: 016110.
<https://doi.org/10.1103/PhysRevE.74.016110>.

</div>

<div id="ref-singer2012vector" class="csl-entry">

Singer, Amit, and Hau-Tieng Wu. 2012. “Vector Diffusion Maps and the
Connection Laplacian.” *Communications on Pure and Applied Mathematics*
65 (8): 1067–1144. <https://doi.org/10.1002/cpa.21395>.

</div>

<div id="ref-song2005selfsimilar" class="csl-entry">

Song, Chaoming, Shlomo Havlin, and Hernán A. Makse. 2005.
“Self-Similarity of Complex Networks.” *Nature* 433: 392–95.
<https://doi.org/10.1038/nature03248>.

</div>

<div id="ref-vidal2007entanglement" class="csl-entry">

Vidal, Guifré. 2007. “Entanglement Renormalization.” *Physical Review
Letters* 99: 220405. <https://doi.org/10.1103/PhysRevLett.99.220405>.

</div>

<div id="ref-wilczekzee1984gauge" class="csl-entry">

Wilczek, Frank, and A. Zee. 1984. “Appearance of Gauge Structure in
Simple Dynamical Systems.” *Physical Review Letters* 52: 2111–14.
<https://doi.org/10.1103/PhysRevLett.52.2111>.

</div>

<div id="ref-wilson1974confinement" class="csl-entry">

Wilson, Kenneth G. 1974. “Confinement of Quarks.” *Physical Review D*
10: 2445–59. <https://doi.org/10.1103/PhysRevD.10.2445>.

</div>

</div>
# Recursive Spectral Fibers on Simplicial Cobordisms

## A geometric program for quarks, color, fermion statistics, Fock space, and baryons

> This Markdown edition is generated from the revised LaTeX source. See the
> [compiled PDF](recursive_spectral_fibers_whitepaper.pdf) for the illustrated
> diagrams and the [referee response](recursive_spectral_fibers_referee_response.md)
> for the disposition of every external-review finding.

## Abstract

This paper proposes a single geometric formulation in which persistent components
become operator-valued response vertices, isolated Hodge bands become fibers, and
frame overlaps induce certified transport. The revision distinguishes static Schur
response from nonzero-frequency Feshbach/AMLS reduction; separates simplicial gluing
from fermionic second quantization; represents edge qubits as local occupation-mode
factors of a generally entangled Fock state; retains determinant-line and projective
center data at rank three; cancels Berry phase in exchange and rotation tests; and
anchors abstract color fibers to oriented triangles. The exact algebraic core and
the proposed particle identifications are labeled separately and tested by explicit
falsification certificates.
