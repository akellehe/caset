# Recursive Spectral Fibers on Simplicial Cobordisms

*A geometric program for quarks, color, fermion statistics, Fock space, and baryons*

Tessera — cobordism programme

Implementation is tracked by GitHub epic [#763](https://github.com/akellehe/tessera/issues/763).

## Abstract

This paper proposes a single geometric formulation for the particle content already suggested by Tessera’s cobordism experiments. A persistent, spectrally certified connected simplicial component is treated as one effective vertex at the next resolution. The component’s selected localized Hodge eigenspace is its fiber, and the couplings between components induce transport between those fibers. Repeating this operation produces a nested, potentially fractal hierarchy of complexes. No independent gauge field, particle label, or auxiliary lattice is introduced. Each edge supplies one two-level occupation mode and complex geometric data, while the generally entangled state lives on the exterior Fock space of all active edge modes. Connections, exchange signs, and observables are derived from that state and the Hodge/Regge operators already present in the construction.

The proposal has a substantial exact core. Static Schur reduction proves when a component may be replaced by a response vertex without changing supported boundary quadratic energies. Nonzero spectral bands instead use the energy-dependent Feshbach–Schur map, or a certified Craig–Bampton/AMLS linear surrogate. Simplicial gluing acts on the one-particle chain space; fermionic second quantization then turns direct sums into graded tensor products and coupling blocks into hopping terms. Every generator so obtained is quadratic, so the dynamics is exactly quasi-free: the state is carried without loss by a covariance matrix, Wick’s theorem evaluates every polynomial certificate, and mean-field geometry backreaction provably stays Gaussian. Three oriented edge-mode factors form the exact exterior algebra $\Lambda^{\bullet}\mathbb{C}^{3}=\mathbf{1}\oplus
\mathbf{3}\oplus\overline{\mathbf{3}}\oplus\mathbf{1}$; the one-occupation sector is a color qutrit, its bilinears close $\mathfrak{su}(3)$, the three-occupation sector is a color singlet, and the grading gives the fermionic exchange sign and canonical anticommutation relations. A rank-$r$ connection is derived from overlap of neighboring spectral frames. At rank three its determinant line and projective $SU(3)/\mathbb{Z}_{3}$ transport are retained rather than choosing an unrecorded cube-root branch. Closed holonomies are gauge-invariant observables rather than new degrees of freedom. Successive cobordism interactions generate the finite stages of an inductive-limit Fock space.

The physical identification remains a hypothesis to be tested. A quark is proposed to be a persistent, odd-parity, rank-three spectral fiber anchored to oriented faces of a certified component; a proton is three such components bound into one persistent supercomponent, occupying a normalized color wedge, carrying baryon number `+1`, electric charge `+1`, and a sharpness-certified total-space spin-`1/2` readout. The paper separates exact identities, results already supported by Tessera, and new falsifiable conjectures, and it states the sharpest open question as a dichotomy: either an exact covariance-only proton exists, or a genuinely non-Gaussian, geometry-mediated interaction is required. An implementation program of structure-exact operations before iterative numerics keeps the defining observables from becoming uncontrolled approximations.

> The rendered vector diagrams are preserved in the LaTeX/PDF edition; this Markdown edition is the searchable text companion.

# Epistemic status and design constraint

Three kinds of statement are deliberately distinguished:

1.  **Exact identity** — follows algebraically from the stated finite complex, orientation, and inner product.

2.  **Existing Tessera evidence** — measured by an existing experiment, with the scope and residual reported in the repository.

3.  **Proposed physical identification** — a new hypothesis with an explicit falsification test.

The governing constraint is parsimony. The ontology is limited to:

- an oriented simplicial complex and its cobordisms;

- a complex squared length and one two-level occupation mode on each edge;

- incidence, Hodge, and Regge operators derived from that data;

- a generally entangled boundary/Fock state on those modes; and

- simplicial gluing followed by fermionic second quantization.

Spectral fibers, color frames, connections, Wilson loops, particle sectors, and coarse vertices are *derived views* of that same data. They are not separately sampled fields. This is important both scientifically and computationally: adding a new independent field could fit a desired answer, while deriving every readout from one complex leaves the construction falsifiable. One consequence is recorded in Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>: every generator this ontology currently supplies is quadratic after second quantization, so the reachable states are exactly the quasi-free class together with whatever non-Gaussian data is fed at the boundary.

<figure id="fig:concept">

<figcaption>Concept map for the recursive complex construction. Colors encode epistemic status, not physical sectors: blue is established or exact machinery, green is a derived observable, and amber is a proposed physical identification.</figcaption>
</figure>

# Present evidence in Tessera

Two existing results motivate the construction.

First, the state-operation-cobordism experiments show that the Hodge-carried register is an isometry to machine precision and that its spectral value reproduces the quantum transition amplitude for every operation that the tested geometry actually carries. Generic fixed-complexity operations can remain obstructed, and the obstruction is visible both as a residual floor and as leakage from the carried subspace. The claim is therefore not that every finite complex realizes every gate; it is that a realized, isometrically embedded register computes the corresponding amplitude. See `cobordism-results.md`.

Second, interaction-history complexes exhibit a stable near-four-dimensional spectral regime. The strongest reported measurements approach, but do not yet prove, an exact spectral dimension of four. The current status and finite-size caveats are recorded in `h_ds4_status.md`. Diffusion-based spectral dimension on simplicial quantum geometries has important precedent in causal dynamical triangulations (Ambjørn, Jurkiewicz, and Loll 2005); the Tessera evidence is an independent result for a different construction and should be compared at the level of the return-probability estimator and its finite-size window.

The current proton animation adds a third, more preliminary observation. It starts with the phase pattern $\{1,\omega,\omega^{2}\}$ and evaluates its singlet diagnostics while a joint Regge–Hodge stationarity objective changes the complex. It no longer forces register holes to appear, and holes have not re-emerged in the current construction. That negative result is useful: the proposed quark should therefore not be defined as a hole. It will be sought as a persistent modular spectral cluster, while Betti numbers remain independent topological observables.

# The microscopic geometric state

Let `K` be a finite oriented simplicial complex. For every edge $e$, store the complex squared length $$z_{e}=\rho_{e}e^{i\theta_{e}}\in\mathbb{C}$$ and attach the two-level occupation factor $\mathcal{H}_{e}=\operatorname{span}\{\lvert 0\rangle_{e},\lvert 1\rangle_{e}\}$. The phase of $z_{e}$ is not a second link field; it is part of the existing complex edge geometry. Saying that an edge “carries a qubit” means that it carries this local mode algebra, not that the global state is forced to be a product of normalized vectors $q_{e}$.

Writing $\mathfrak{h}_{K}=\operatorname{span}\{\lvert e\rangle : e\in K_{1}\}$ for the one-particle edge space, the microscopic quantum carrier is $$\mathcal{H}_{K}=\mathcal{F}_{-}(\mathfrak{h}_{K})=\Lambda^{\bullet}\mathfrak{h}_{K}
  \;\cong\;\widehat{\bigotimes}_{e\in K_{1}}\mathcal{H}_{e},$$ and a boundary state is a vector or density operator on $\mathcal{H}_{K}$. It may be entangled. A one-particle color state $a^{\dagger}_{\phi}\lvert 0\rangle$ and the nonseparable proton-spin sectors are therefore native states, not exceptions to the ontology. For an isolated occupied band with projector $P$, the corresponding quasi-free reference state has covariance $$\Gamma_{ef}=\langle a^{\dagger}_{f}a_{e}\rangle=P_{ef},
  \qquad \langle n_{e}\rangle=P_{ee}.$$ Thus a per-edge Bloch vector or occupation is a derived marginal/readout. The quasi-free state is a useful analytic baseline, not a restriction of the state space: the lazy Fock construction of Section <a href="#sec:fock" data-reference-type="ref" data-reference="sec:fock">12</a> can represent explicitly non-Gaussian sectors. Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a> records, however, that no generator currently present in the model produces such sectors from Gaussian data; until one of the mechanisms listed there is adopted, non-Gaussianity can enter only as boundary data.

Let $$\partial_{k}:C_{k}(K)\longrightarrow C_{k-1}(K),
  \qquad \partial_{k-1}\partial_{k}=0$$ be the oriented boundary maps and let `W_k(z)` be the metric weight on `k`-chains. With the weighted adjoint $$\partial^{*}_{k}=W_{k}^{-1}\partial^{\dagger}_{k}W_{k-1},$$ the degree-`k` Hodge Laplacian is $$L_{k}=\partial_{k+1}\partial^{*}_{k+1}+\partial^{*}_{k}\partial_{k}.$$ In the positive metric regime this is self-adjoint in the `W_k` inner product. In the signed Lorentzian regime it can be non-normal; then left and right spectral frames and their biorthogonal condition numbers must be reported rather than silently treating `L_k` as Hermitian.

The geometry evolves toward joint stationary points of the existing Regge and Hodge functionals. In emergence mode, particle-specific observables below are read after optimization and are not inserted as target terms. Controlled synthesis mode may pin a carrier to test realizability, but that is a separate experiment. Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a> refines the emergence protocol into two labeled modes — strict no-backreaction, and certificates-blind mean-field backreaction — and records that both remain inside the quasi-free class.

This operator stack sits on established foundations: Regge calculus encodes piecewise-flat gravity in simplicial deficit angles (Regge 1961), discrete exterior calculus supplies metric-dependent chain/cochain operators (Desbrun et al. 2005), and combinatorial Hodge spectra on simplicial complexes have a developed spectral theory (Horak and Jost 2013). Tessera’s proposal is not a replacement for those constructions; it is a constrained use of them as the sole source of the later particle readouts.

# A component is an exact static response vertex

Partition the `k`-cells of a connected component into interface cells `B` and interior cells `I`, and block its Hodge operator as $$L=\begin{pmatrix} L_{BB} & L_{BI}\\ L_{IB} & L_{II}\end{pmatrix}.$$ In the positive self-adjoint regime, after projecting out incompatible interior zero modes, minimization over the interior has the exact solution $$x_{I}^{*}=-L_{II}^{+}L_{IB}x_{B},$$ and the exact effective boundary operator $$\boxed{\,L_{\mathrm{eff}}=L_{BB}-L_{BI}L_{II}^{+}L_{IB}\,}.$$ Here `+` denotes the Moore–Penrose inverse on the supported interior subspace. For every compatible boundary value, $$\min_{x_{I}}
  \begin{pmatrix}x_{B}\\ x_{I}\end{pmatrix}^{\dagger}
  L
  \begin{pmatrix}x_{B}\\ x_{I}\end{pmatrix}
  = x_{B}^{\dagger}L_{\mathrm{eff}}x_{B}.$$

This is the precise static, or zero-frequency, sense in which a connected component can be replaced by a coarse response vertex. In a Hermitian indefinite regime the same equation is a stationarity condition, not a minimum. For a non-normal block it is simply block elimination; solvability requires $$L_{IB}x_{B}\perp\ker L_{II}^{\dagger}.$$

The plain Schur complement does *not* preserve the nonzero spectrum. For a spectral parameter $\lambda$ such that $L_{II}-\lambda I$ is invertible, define the exact Feshbach–Schur response $$\boxed{\,F_{B}(\lambda)=L_{BB}-\lambda I
    - L_{BI}(L_{II}-\lambda I)^{-1}L_{IB}\,}.$$ Then, for $\lambda$ outside $\operatorname{spec}L_{II}$, the exact determinant factorization $$\det(L-\lambda I)=\det(L_{II}-\lambda I)\,\det F_{B}(\lambda)$$ holds, so $$\lambda\in\operatorname{spec}L \iff 0\in\operatorname{spec}F_{B}(\lambda).$$ The order of the zero of $\det F_{B}(\cdot)$ at $\lambda$ equals the algebraic multiplicity of $\lambda$ in $L$, while $\dim\ker F_{B}(\lambda)$ equals its geometric multiplicity; the two agree in the self-adjoint or otherwise semisimple setting but not in general. At an interior resonance the inverse is replaced only after checking the compatibility condition $L_{IB}x_{B}\perp\ker(L_{II}-\lambda I)^{\dagger}$ and retaining the resonant interior modes explicitly. Thus harmonic response uses $F_{B}(0)$, while a localized band centered at $\lambda_{C}$ uses $F_{B}(\lambda)$ over a stated frequency window. A linear reduced eigenproblem may instead retain interface constraint modes plus selected fixed-interface modes using Craig–Bampton component-mode synthesis or AMLS; that route is certified approximation whose error is controlled by residuals and separation from discarded modes, not an exact spectral identity (Craig and Bampton 1968; Bennighof and Lehoucq 2004; Bach et al. 2003).

The effective blocks between coarse components become operator-valued links. A harmonic or retained interior mode is not discarded; it becomes an explicit stalk/fiber coordinate attached to the response vertex.

For graph Laplacians this is the classical Kron reduction by Schur complement (Dörfler and Bullo 2013). Spectral graph reduction provides related approximation guarantees when additional coarsening or truncation is performed (Loukas 2019). The extension proposed here is to apply static response reduction degree by degree to weighted Hodge blocks, and shifted Feshbach or certified component-mode reduction to nonzero bands, while retaining localized zero, resonant, and selected interior modes as explicit fiber coordinates.

# Recursive spectral fibers

Let $P_{\ell}=\{C_{v}^{\ell}\}$ be an intrinsic partition at scale $\ell$ into persistent connected components. At $\ell=0$ the object is the microscopic simplicial complex $K_{0}$. After the first elimination the honest coarse object is generally not another simplicial complex: it is an operator-valued response network $\mathcal{R}_{\ell+1}$ whose vertices carry vector spaces and whose links carry linear response blocks. A cellular sheaf on the quotient graph is a natural realization when the blocks admit compatible restriction-map factorization (Hansen and Ghrist 2019); otherwise Tessera retains the more general response network and does not invent incidence maps that the reduction did not determine.

Within component `C`, choose an isolated localized spectral band and, in the positive self-adjoint regime, a weighted orthonormal frame $$\Phi_{C}=(\phi_{1},\dots,\phi_{r}),
  \qquad \Phi_{C}^{\dagger}W_{C}\Phi_{C}=I_{r}.$$ The derived fiber is $$E_{C}=\operatorname{Ran}\Phi_{C}.$$

In a Hermitian indefinite regime record the inertia of $\Phi_{C}^{\dagger}W_{C}\Phi_{C}$ and normalize it to a signature matrix $J_{C}=\operatorname{diag}(I_{p},-I_{q})$. Negative Krein signature is a certificate, not an automatic identification with an antiparticle. Existing Tessera pair-creation experiments do, however, exhibit an opposite-signature selection rule with conserved real part under conjugate-pair formation; that measured behavior is second-tier evidence for the particle/antiparticle reading, while the identification itself remains a third-tier proposed interpretation. In a non-normal regime use matched right and left frames $\Phi_{C},\Psi_{C}$ with $\Psi_{C}^{\dagger}W_{C}\Phi_{C}=I$ and report both residuals and the frame condition number.

It need not be a harmonic space and therefore need not be supported by a hole. What it does require is a spectral gap, localization, and persistence. A candidate component is accepted only if all of the following remain stable across a stated range of scales:

- a persistent connected cluster support, however proposed;

- a localized spectral projector with stable rank;

- a nonzero band gap separating it from discarded modes;

- overlap with its predecessor and successor components;

- lifetime across multiple cobordism frames; and

- small external transport leakage.

Community objectives supply deterministic cluster candidates (Reichardt and Bornholdt 2006), while network renormalization supplies tests for genuine self-similarity rather than visual resemblance (Song, Havlin, and Makse 2005). The partition is therefore a measured part of the analysis: a recursively drawn pattern is not evidence of a fractal unless its scaling observables survive a refinement window. The current `ModularityOptimizer` uses Newman–Girvan modularity on a combinatorial one-skeleton; it is a heuristic proposal generator that does not see signed or complex Hodge weights and is subject to the modularity resolution limit (Fortunato and Barthélemy 2007). Modularity may therefore propose candidate supports, but it may not veto an otherwise certified fiber: acceptance is conditioned only on the independent, weight-aware gap, localization, leakage, persistence, and refinement certificates above, together with the anchoring certificate of Section <a href="#sec:quarks" data-reference-type="ref" data-reference="sec:quarks">10</a> whenever a color interpretation is claimed.

This gives a type-stable hierarchy of response objects $$\cdots\longrightarrow\mathcal{R}_{2}\longrightarrow\mathcal{R}_{1}\longrightarrow K_{0}$$ in which a response vertex at one level resolves into a connected microscopic component plus retained stalk coordinates at the next finer level. “Self-similar” refers to closure of the response-network data type, not to a claim that every reduced operator is a simplicial Hodge Laplacian. A fractal-like pattern is permitted but not required: measured scaling of module count, volume, boundary size, and spectral gap decides whether the hierarchy is statistically self-similar.

<figure id="fig:recursion">

<figcaption>One recursive step. Persistent connected modules become stalk-bearing vertices of an operator-valued response network. Static response is preserved by the supported Schur complement; nonzero bands use shifted Feshbach or certified component-mode reduction. Selected internal modes remain attached as fibers, and a persistent supermodule can be reduced again at the next scale.</figcaption>
</figure>

# Interactions and the expanding Hilbert space

Two operations must not be conflated. For the Cartesian product of chain complexes `A` and `B`, the graded tensor differential is the exact rule $$d_{A\mathbin{\widehat{\otimes}}B}(a\otimes b)=d_{A}a\otimes b+(-1)^{\deg a}a\otimes d_{B}b.$$ For a noninteracting product with product metric, $$L_{A\mathbin{\widehat{\otimes}}B}=L_{A}\otimes I+I\otimes L_{B},$$ so one-particle eigenvalues add and eigenvectors tensor. This identity is about a product complex, not about gluing two cobordisms.

Actual simplicial gluing is a pushout along a shared boundary. At the one-particle level it produces a chain space assembled from direct sums modulo boundary identifications (equivalently described by the relevant Mayer–Vietoris sequence) and a block operator $$L_{A\cup B}=\begin{pmatrix} L_{A} & C_{AB}\\ C_{BA} & L_{B}\end{pmatrix}$$ in a basis adapted to the two interiors. The coupling blocks are induced by the connecting simplices and shared-boundary constraints; they are not a Kronecker interaction term.

The expanding Hilbert space follows after applying the fermionic Fock functor to the one-particle space $\mathfrak{h}$. The exact identities are $$\mathcal{F}_{-}(\mathfrak{h}_{A}\oplus\mathfrak{h}_{B})\cong\mathcal{F}_{-}(\mathfrak{h}_{A})\mathbin{\widehat{\otimes}}\mathcal{F}_{-}(\mathfrak{h}_{B}),$$ and $$d\Gamma(L_{A}\oplus L_{B})=d\Gamma(L_{A})\mathbin{\widehat{\otimes}}I+I\mathbin{\widehat{\otimes}}d\Gamma(L_{B}).$$ For the coupling block, $$d\Gamma(C_{AB}+C_{BA})
  =\sum_{ij}(C_{AB})_{ij}\,a^{\dagger}_{A,i}a_{B,j}+\text{h.c.},$$ so geometric connections become hopping terms without adding a new field. If the one-particle eigenvalues are $\lambda_{1},\dots,\lambda_{M}$, then the free many-body spectrum is the set of occupation subset sums $\sum_{i}n_{i}\lambda_{i}$, $n_{i}\in\{0,1\}$, rather than the one-particle pairwise spectrum being relabeled as a Fock spectrum (Berezin 1966).

At the selected-fiber level, an interaction grows the carried space as $$\mathcal{H}_{AB}=E_{A}\mathbin{\widehat{\otimes}}E_{B},$$ and a later interaction appends another factor. This is a statement about state-space composition after second quantization, not the topology of the glued chain complex. When carried subspaces of adjacent components overlap on interface cells, the composite is built on the abstract labeled sum with an explicit embedding Gram matrix; Section <a href="#sec:master" data-reference-type="ref" data-reference="sec:master">14</a> states the exact rule. If $J_{C}$ embeds an abstract state into the geometric carrier, exact amplitude preservation requires $$J_{C}^{\dagger}W_{C}J_{C}=I.$$ Tensor products preserve isometry exactly. If $G=J_{C}^{\dagger}W_{C}J_{C}$ has Gram defect $\varepsilon=\lVert G-I\rVert$, then $$\lvert a^{\dagger}Gb-a^{\dagger}b\rvert
  \le\varepsilon\,\lVert a\rVert\,\lVert b\rVert,$$ and two tensor factors obey $$\varepsilon_{AB}\le\varepsilon_{A}+\varepsilon_{B}
    +\varepsilon_{A}\varepsilon_{B}.$$ Thus the amplitude claim has an explicit, composable error budget.

Cobordism composition as a map between boundary state spaces is the organizing idea of topological field theory (Atiyah 1988); the general-boundary program makes the region/boundary assignment explicit for quantum theory (Oeckl 2003), and categorical quantum mechanics formalizes tensor composition and diagrammatic process semantics (Abramsky and Coecke 2004). Tessera keeps only the parts that can be realized by its finite simplicial carrier and tests the resulting map numerically rather than assuming topological invariance.

# Quasi-free dynamics and the covariance layer

Every many-body generator exhibited in this paper is quadratic. Free propagation is $d\Gamma(L)$, gluing contributes $d\Gamma$ of a coupling block, and every derived transport is the second quantization of a one-particle map. The exact consequence is closure of the quasi-free class: if the Hamiltonian is always of the form $$H(t)=d\Gamma\bigl(h(t)\bigr)=\sum_{ij}h_{ij}(t)\,a^{\dagger}_{i}a_{j},$$ then Gaussian/quasi-free states remain Gaussian.

The closure survives self-consistency. Let the one-particle operator depend on the covariance and on the classical geometry, $$h=h\bigl(\Gamma(t),g(t)\bigr),$$ with the geometry in turn relaxed against the state’s energy density. That is nonlinear mean-field dynamics of generalized Hartree–Fock type (Bach, Lieb, and Solovej 1994); it can localize and it can produce self-bound solutions, but it does not leave the Gaussian manifold. Classical or mean-field geometry backreaction alone therefore does not generate genuinely non-Gaussian correlations.

The emergence protocol accordingly splits into two labeled modes, both Gaussian-closed: *strict emergence*, in which the state does not act back on the geometry at all, and *certificates-blind mean-field backreaction*, in which the carried state’s energy density enters the joint stationarity objective while every particle certificate remains firewalled from it. The certificate firewall of Section <a href="#sec:proton" data-reference-type="ref" data-reference="sec:proton">13</a> applies to both modes.

Genuinely non-Gaussian correlations would require at least one of the following, none of which is currently part of the model:

1.  a genuine quartic effective interaction, $$H_{\mathrm{int}}
          =\sum_{ijkl}V_{ijkl}\,
            a^{\dagger}_{i}a^{\dagger}_{j}a_{k}a_{l};$$

2.  quantized geometry that becomes entangled with the fermions;

3.  integrating out dynamical geometry beyond the mean-field approximation, producing a retarded or quartic effective interaction;

4.  a cobordism map that is not the second quantization of a one-particle map; or

5.  measurement or postselection capable of taking Gaussian states outside the Gaussian class.

Adopting one of these is an explicit scope decision with its own certificates, not a background assumption. Until then, the statement that non-Gaussian sectors are representable (Section <a href="#sec:state" data-reference-type="ref" data-reference="sec:state">3</a>) must not be read as a statement that they are produced.

The quasi-free formulation is extremely attractive on its own terms. In the number-conserving case the entire state is the covariance matrix $$\Gamma_{ij}=\langle a^{\dagger}_{j}a_{i}\rangle,
  \qquad i\dot{\Gamma}=[h,\Gamma],$$ with $\Gamma^{2}=\Gamma$ exactly for a pure Slater state; a pairing sector would extend $\Gamma$ to the full Nambu covariance without changing the closure statement. Wick’s theorem then computes every polynomial observable exactly: occupations and parities, the Pauli/Gram determinants, the color wedge $\lvert S_{ABC}\rvert^{2}$, and both $\langle J^{2}\rangle$ and its variance. There is no reason to construct an exponential Fock vector except for oracle tests or for explicitly non-Gaussian boundary data.

The programme order follows. First test the strongest possible covariance-only theory. Treat failure of the sharp proton certificate of Section <a href="#sec:proton" data-reference-type="ref" data-reference="sec:proton">13</a> as a meaningful structural result rather than a numerical nuisance. Introduce a non-Gaussian interaction only if the geometry supplies one naturally, through one of the mechanisms above. The question that decides the next stage of the programme is stated exactly:

<div class="center">

</div>
Either answer is informative. A covariance-only proton would make the entire particle layer polynomially computable and exactly certifiable; a demonstrated obstruction would be the first internal evidence that the geometry must supply a true interaction term.

# A triangle carries the exact color algebra

Consider the three edge-mode factors around an oriented triangle and interpret `|1>` as an occupied edge mode. Choosing an oriented ordering $(e_{1},e_{2},e_{3})$ identifies their graded tensor product with the exterior algebra $$(\mathbb{C}^{2})^{\mathbin{\widehat{\otimes}}3}\cong\Lambda^{\bullet}\mathbb{C}^{3}
  =\mathbf{1}\oplus\mathbf{3}\oplus\overline{\mathbf{3}}\oplus\mathbf{1}.$$ The orientation of one triangle fixes the ordering up to a cyclic, hence even, permutation, so the local wedge sign is unambiguous. Globally the exterior algebra $\Lambda^{\bullet}\mathfrak{h}_{K}$ and the CAR are intrinsic; only a compilation into tensor-product qubits or bitsets needs a deterministic mode order and the corresponding permutation parity. A Kasteleyn orientation is useful for two-dimensional surface-dimer Pfaffians but is not required to define this abstract Fock space (Cimasoni and Reshetikhin 2007). A genuine continuum spinor interpretation is a separate question addressed by the rotation certificate below.

The sectors have occupation number $N=0,1,2,3$:

<div id="tab:sectors">

| Sector                      | Basis dimension | Color interpretation       | Fermion parity |
|:----------------------------|:---------------:|:---------------------------|:--------------:|
| $\Lambda^{0}\mathbb{C}^{3}$ |        1        | vacuum                     |      even      |
| $\Lambda^{1}\mathbb{C}^{3}$ |        3        | fundamental color triplet  |      odd       |
| $\Lambda^{2}\mathbb{C}^{3}$ |        3        | antisymmetric anti-triplet |      even      |
| $\Lambda^{3}\mathbb{C}^{3}$ |        1        | color singlet              |      odd       |

Exterior sectors of three oriented edge modes.

</div>

Let $a^{\dagger}_{i},a_{i}$ be the exterior creation and contraction operators. They satisfy the canonical anticommutation relations exactly: $$\{a_{i},a_{j}\}=0,\qquad
  \{a^{\dagger}_{i},a^{\dagger}_{j}\}=0,\qquad
  \{a_{i},a^{\dagger}_{j}\}=\delta_{ij}.$$ On the one-occupation sector, the bilinears $$E_{ij}=a^{\dagger}_{i}a_{j}$$ satisfy $$=\delta_{jk}E_{i\ell}-\delta_{i\ell}E_{kj}.$$ The six Hermitian off-diagonal combinations together with $$H_{1}=E_{11}-E_{22},\qquad
  H_{2}=\frac{E_{11}+E_{22}-2E_{33}}{\sqrt{3}}$$ are the eight generators of $\mathfrak{su}(3)$. Thus the triangle does not merely hold three phases: its one-particle edge sector carries the fundamental representation, its two-particle sector carries the dual representation, and its traceless bilinears carry the adjoint octet.

The triplet description of quarks and the three-quark construction of baryons originate with the quark model (Gell-Mann 1964); the additional color triplet was introduced to resolve the statistics and state-counting problem (Han and Nambu 1965; Greenberg 1964). The claim here is narrower and new: Tessera’s three oriented edge modes would provide a geometric carrier of the same representation content, not a derivation of QCD from the combinatorics alone.

<figure id="fig:triangle">

<figcaption>Exact representation content of three oriented edge-mode factors. The exterior sectors and their parity are algebraic identities. Interpreting the rank-three odd sector as quark color and the top wedge as a baryon color singlet is the physical hypothesis to be tested.</figcaption>
</figure>

## Geometric normalization

For the stored complex squared lengths $z_{i}=\rho_{i}e^{i\theta_{i}}$ on the three oriented edges, define $$c_{i}=\frac{z_{i}}
    {\sqrt{\lvert z_{1}\rvert^{2}+\lvert z_{2}\rvert^{2}
      +\lvert z_{3}\rvert^{2}}},
  \qquad
  \lvert c\rangle=\sum_{i=1}^{3}c_{i}\lvert i\rangle.$$ Then $\langle c|c\rangle=1$. Constraining the perimeter to one is a valid geometric scale gauge, but it is an `L^1` condition and does not replace the $L^{2}$ Hilbert normalization. Normalized pure color states form $\mathbb{CP}^{2}$; $SU(3)$ is the transformation group, not the surface of the triangle itself.

## The existing omega phase pattern

Let $\omega=e^{2\pi i/3}$. The exact Fourier frame $$F_{3}=\frac{1}{\sqrt{3}}
  \begin{pmatrix}
    1 & 1 & 1\\
    1 & \omega & \omega^{2}\\
    1 & \omega^{2} & \omega
  \end{pmatrix}$$ is unitary. The existing pattern $(1,\omega,\omega^{2})/\sqrt{3}$ is therefore one color basis vector, not by itself the whole color fiber. Its cyclic orbit supplies an exact orthonormal triad.

# Color transport and Wilson loops without a new gauge field

Let $T_{AB}$ be the chain-level transfer already induced by connecting simplices from component $B$ to component $A$. In positive self-adjoint local spectral frames of common rank $r$, the raw fiber map is $$M_{AB}=\Phi_{A}^{\dagger}W_{A}T_{AB}\Phi_{B}.$$ Its departure from an isometry is a physical leakage diagnostic: $$\eta_{AB}=\lVert M_{AB}^{\dagger}M_{AB}-I\rVert.$$ When the selected band is isolated, $M_{AB}$ has full numerical rank, and $\eta_{AB}$ is small, take the polar unitary $$V_{AB}=M_{AB}(M_{AB}^{\dagger}M_{AB})^{-1/2}\in U(r).$$ Under a change of local spectral frame, $\Phi_{A}\mapsto\Phi_{A}g_{A}$ and $\Phi_{B}\mapsto\Phi_{B}g_{B}$, $$V_{AB}\longmapsto g_{A}^{\dagger}V_{AB}g_{B}.$$ Consequently the full closed holonomy and its normalized trace transform by conjugation at the base point: $$H(\gamma)=\prod_{(AB)\in\gamma}V_{AB},
  \qquad
  W_{U(r)}(\gamma)=\frac{1}{r}\operatorname{Tr}H(\gamma).$$

At rank three there is no globally single-valued operation $V\mapsto V/(\det V)^{1/3}$: the cube root is $\mathbb{Z}_{3}$-ambiguous. The faithful derived datum is therefore retained as $$V_{AB}\in U(3),\qquad
  \delta_{AB}=\det V_{AB}\in U(1),\qquad
  [V_{AB}]\in PU(3)\cong SU(3)/\mathbb{Z}_{3}.$$ A local $SU(3)$ lift $\widetilde{U}_{AB}=\delta_{AB}^{-1/3}V_{AB}$ may be followed continuously along a path after fixing a base branch, but its accumulated center sector must be reported. Fundamental Wilson traces use the full $U(3)$ holonomy or an explicitly lifted path; adjoint/projective Wilson loops are center-blind and require no branch. This turns the former ambiguity into two measured sectors rather than silently discarding one.

The determinant line also supplies a possible oriented flux readout. For a closed, full-rank world-tube family $V(t)$, $$\nu=\frac{1}{2\pi}\oint d\,\arg\det V(t)\in\mathbb{Z}$$ is homotopy-invariant while the gap and rank remain open, and changes sign when the tube orientation is reversed. The integer character of $\nu$ requires the closed loop. A quark or proton world tube on a cobordism segment is an interval, and its raw endpoint change of $\arg\det V$ is a phase difference, not an invariant. The open-segment definition is therefore relative: either compose the physical transport with the inverse of a matched reference transport — the same non-exchanging reference construction used in Section <a href="#sec:exchange" data-reference-type="ref" data-reference="sec:exchange">11.1</a> — so that the composite closes, or fix endpoint trivializations supplied by the boundary registers. The reported $\nu$ is the integer winding of that closed composite, together with its reference specification. Without such closure $\nu$ is merely an endpoint phase change and is not certified. Identifying $B=\nu/3$ for an accepted quark tube is a proposed physical interpretation, not a group identity. Conservation under pair creation is exact only for a continuous conjugate-pair homotopy with no determinant zero or boundary flux.

For non-normal bands the correct overlap is biorthogonal, $$M_{AB}=\Psi_{A}^{\dagger}W_{A}T_{AB}\Phi_{B},
  \qquad
  \Psi_{C}^{\dagger}W_{C}\Phi_{C}=I.$$ It is generally a $GL(r,\mathbb{C})$ transport, not a unitary one; left and right residuals, singular values, and frame condition numbers are part of the observable. In an indefinite Hermitian sector the Krein inertia is reported and a pseudo-unitary reduction is attempted only when the two signatures agree. No $U(r)$ or $SU(3)$ Wilson value is emitted by silently applying the positive-metric formula outside its domain.

Polar normalization must never conceal a bad fiber assignment: every accepted holonomy is reported together with $\eta_{AB}$, rank/singular value thresholds, endpoint band gaps, and frame conditioning.

This construction is the discrete spectral-frame analogue of Berry transport (Berry 1984) and its non-Abelian Wilczek–Zee generalization (Wilczek and Zee 1984). Overlap matrices are also a standard route to gauge-covariant lattice observables (Fukui, Hatsugai, and Suzuki 2005), as are connection Laplacians and vector diffusion maps (Singer and Wu 2012). Wilson loops themselves are foundational lattice gauge observables (Wilson 1974). What is specific here is that the link matrix is not independently assigned: it is reconstructed from neighboring Hodge frames and accompanied by a leakage certificate.

# Quarks as modular clusters

A *quark candidate* is proposed to be a component `Q` satisfying all of the following derived conditions.

The abstract rank-three band must first be anchored to oriented two-simplices. For an oriented triangle $\tau\subset Q$, let $R_{\tau}:C_{1}(Q)\to\mathbb{C}^{3}$ restrict a one-chain to the three ordered boundary edges, including their incidence signs, and let $\lvert W_{\tau}\rvert$ be the modulus of the edge-weight block on those three edges. Define the weighted restriction $$A_{\tau}=\lvert W_{\tau}\rvert^{1/2}R_{\tau}\Phi_{Q}.$$ For a rank-three frame $\Phi_{Q}$, define the gauge-invariant anchor score $$a_{Q}^{2}=\sum_{\tau\subset Q}w_{\tau}\,
    \bigl\lvert\det A_{\tau}\bigr\rvert^{2},
  \qquad w_{\tau}\ge 0,\qquad \sum_{\tau}w_{\tau}=1,$$ with the convex weighting $\{w_{\tau}\}$ declared before the data are examined. In the positive regime $R_{\tau}^{\dagger}\lvert W_{\tau}\rvert
R_{\tau}\preceq W$, so each $\lvert\det A_{\tau}\rvert^{2}=\det(A_{\tau}^{\dagger}A_{\tau})\le 1$ and the score is calibrated: $a_{Q}^{2}\in[0,1]$, with value one exactly at full concentration on the weighted edge span of the anchoring faces. In signed sectors the restriction still uses $\lvert W_{\tau}\rvert^{1/2}$ and the Krein signature of the restricted block is reported separately.

The phases of the nonzero determinants give a local determinant-line trivialization; their coherence on overlapping triangles is recorded separately. The reported anchor datum is the profile, not only the score: the maximal term $\max_{\tau}\lvert\det A_{\tau}\rvert^{2}$, the participation ratio of the distribution $\{\lvert\det A_{\tau}\rvert^{2}\}$, and the determinant-phase dispersion accompany $a_{Q}^{2}$. Concentration on one triangle is a sufficient special case, not a requirement: an extended fiber may be anchored by an atlas of oriented faces. Because every 2-simplex has exactly three boundary edges, this anchor is independent of the ambient spectral dimension.

1.  `Q` is a persistent cluster certified as in Section <a href="#sec:fibers" data-reference-type="ref" data-reference="sec:fibers">5</a>, however its support was proposed.

2.  Its selected color fiber has stable rank three.

3.  Its calibrated triangle-anchor profile and determinant-line coherence are stable.

4.  It occupies an odd exterior sector.

5.  Its color transport has bounded leakage over its lifetime.

6.  Its determinant line has an accepted relative winding $\nu=\pm 1$ in the closed-composite sense of Section <a href="#sec:transport" data-reference-type="ref" data-reference="sec:transport">9</a>; reversing the tube yields the dual color representation and an antiquark.

7.  Its total spectral fingerprint is stable under refinement and vertex relabeling.

The distinction between an antiquark and the $\Lambda^{2}\mathbb{C}^{3}$ anti-triplet of two quarks is made by determinant-line orientation and total occupation, not color alone. Assigning $B=\nu/3$ is accepted only when the winding certificate above exists. A forward/reverse pair then has zero total winding under a gap-preserving conjugate homotopy; without that certificate baryon number remains unknown rather than being inserted by definition.

Flavor and electric charge are not assumed as hidden labels. The conservative hypothesis is that two stable subclasses of the same cluster fiber provide an isospin doublet. On such a doublet, the measured orientation flux supplies baryon number and the standard relation $$Q=I_{3}+\frac{B}{2}$$ gives $Q_{u}=+2/3$ and $Q_{d}=-1/3$. This is a proposed identification, not yet a derivation: it succeeds only if an unlabeled two-dimensional spectral band emerges, is transported coherently, and its Gauss-flux readout agrees with those values.

# Fermion statistics from simplicial orientation

The graded interchange law is $$\tau(a\mathbin{\widehat{\otimes}}b)=(-1)^{F_{a}F_{b}}\,b\mathbin{\widehat{\otimes}}a.$$ Two odd clusters therefore acquire a minus sign on exchange, while an even composite does not. Parity adds modulo two:

<div id="tab:parity">

| Object             | Odd constituents | Composite parity |
|:-------------------|:----------------:|:----------------:|
| quark or antiquark |        1         |       odd        |
| meson $q\bar{q}$   |        2         |       even       |
| diquark $qq$       |        2         |       even       |
| baryon $qqq$       |        3         |       odd        |

Composite exchange parity from constituent counting.

</div>

Pauli exclusion is the exact exterior-algebra identity $$\lVert v_{1}\wedge\cdots\wedge v_{n}\rVert^{2}
  =\det[\langle v_{i},v_{j}\rangle].$$ If two complete one-particle modes coincide, the determinant and the state vanish. The “complete” qualifier prevents double-counting signs: color, spin, flavor, space, and component support are wedged once as one mode. One must not multiply an extra fermion sign by the sign already present in the color epsilon tensor.

## A label-independent exchange experiment

Let $\Phi_{t}$ be an orthonormal frame for the isolated odd subspace at frame $t$. Define the parallel transport $$R_{t}=\operatorname{polar}(\Phi_{t+1}^{\dagger}W_{t}\Phi_{t}),
  \qquad
  U_{\gamma}=R_{T-1}\cdots R_{0}.$$ The raw determinant line contains both permutation statistics and an ordinary Abelian Berry phase: $$\chi_{\mathrm{raw}}(\gamma)=\det U_{\gamma}.$$ It is therefore not expected to equal $\pm 1$ on a generic geometric loop. Let $\gamma_{0}$ be a non-exchanging reference motion with the same geometric footprint, timing, and local frame convention. The interferometric exchange character is $$\widehat{\chi}_{F}(\gamma)=\frac{\det U_{\gamma}}{\det U_{\gamma_{0}}}.$$ The proposed dynamical test is $$\widehat{\chi}_{F}(\text{single exchange})=-1,
  \qquad
  \widehat{\chi}_{F}(\text{double exchange})=+1.$$

As an independent structural cross-check, persistent component matching extracts the permutation $P_{\gamma}$ of localized odd blocks and reports $\operatorname{sgn}P_{\gamma}$ together with the norm of the residual in-block motion after comparison with the reference loop. The algebraic wedge sign is exact; the interferometric holonomy is the dynamical certificate.

A physical $2\pi$ rotation uses the same reference normalization, and the rotation path is not left abstract: it is the documented total-space spin holonomy cycle already used for the $J^{2}$ readout (`joint_proton_spin_findings.md`), executed as a closed loop and normalized against its own co-moving, non-rotating reference. If an emergent manifold-like regime supplies tangent frames, a continuum spinor claim additionally requires a lift of the frame holonomy from $SO(d)$ to $\mathrm{Spin}(d)$; obstruction by the second Stiefel–Whitney class is then a falsification certificate (Lawson and Michelsohn 1989). This requirement concerns the physical spin lift, not the existence of the abstract CAR/Fock algebra. The spin-statistics comparison is cleanest as $$\widehat{\chi}_{F}(\text{exchange})\,
  \widehat{\chi}_{F}(2\pi\ \text{rotation})^{-1}=+1,$$ with each factor separately near $-1$.

Configuration-space topology already explains how exchange classes can carry quantum phases (Laidlaw and DeWitt 1971) and, in two dimensions, more general statistics (Leinaas and Myrheim 1977). The Tessera proposal uses this precedent only as a diagnostic template. The minus sign from the graded exterior algebra is exact; the claim that an actual geometric exchange cobordism realizes the corresponding determinant holonomy remains an experiment.

# Fock space as an inductive limit of interactions

For $M$ oriented fermionic edge modes, $$\widehat{\bigotimes}_{m=1}^{M}\mathbb{C}^{2}
  \cong\Lambda^{\bullet}\mathbb{C}^{M}
  =\bigoplus_{n=0}^{M}\Lambda^{n}\mathbb{C}^{M},$$ and the dimension identity is exact: $$2^{M}=\sum_{n=0}^{M}\binom{M}{n}.$$

The exterior algebra is canonical as a functor of the one-particle space. Writing it as a literal ordered qubit tensor product, or implementing creation operators by Jordan–Wigner/bitset strings, requires a chosen mode order. Tessera derives a deterministic order from oriented component lineage and applies the parity of every reordering; all reported observables must be invariant under relabeling plus that induced unitary.

Adding a new noninteracting mode uses the vacuum embedding $$\iota_{M}:\mathcal{H}_{M}\hookrightarrow\mathcal{H}_{M+1},
  \qquad
  \iota_{M}(\psi)=\psi\mathbin{\widehat{\otimes}}\lvert 0\rangle.$$ The infinite Fock space is the direct limit $$\mathcal{F}=\varinjlim(\mathcal{H}_{M},\iota_{M}).$$ This makes the infinite expansion precise without ever allocating an infinite array. At every finite simulation time only finitely many modes have interacted. Consistency requires $$\lVert\iota_{M}U_{M}-U_{M+1}\iota_{M}\rVert\longrightarrow 0$$ over a refinement sequence.

Bosonic gauge excitations need not add a new local oscillator. The traceless even bilinears $$a^{\dagger}_{i}a_{j}-\frac{1}{3}\delta_{ij}N$$ span the color octet in $\mathbf{3}\otimes\overline{\mathbf{3}}=\mathbf{1}\oplus\mathbf{8}$ and have even fermion parity. Arbitrarily many such collective excitations are represented by adding more microscopic modes at finer resolution. Each finite edge-mode factor remains two-dimensional.

This scale-by-scale state growth is adjacent to entanglement renormalization, where local Hilbert data are reorganized across layers before truncation (Vidal 2007). The distinction is material: Tessera uses static/shifted response reduction plus an inductive vacuum embedding, and it must certify compatibility between successive finite spaces rather than assume a fixed bond dimension.

## Occupation exterior algebra is not automatically Kähler–Dirac

The exterior algebra above is over the one-particle *mode space*; its degree is occupation number. A Kähler–Dirac field instead lives on the inhomogeneous differential-form/cochain space $\bigoplus_{k}C^{k}(K)$ and is acted on by $d-d^{*}$ (or a related Dirac square root). These constructions share exterior-algebra notation but are not the same operator or grading. Consequently the present model does not inherit lattice taste multiplicity merely from using $\Lambda^{\bullet}\mathfrak{h}_{K}$.

If a later Tessera model promotes its one-particle field to all cochain degrees and uses the Kähler–Dirac operator, the known flat four-dimensional decomposition into four Dirac spinors becomes an expected spectrum diagnostic, not an unexplained bug (Becher and Joos 1982; Butt et al. 2021). Any observed near-fourfold cluster in the present model is reported as an empirical degeneracy until that stronger operator identification is made.

# The proton as the maximally informative baryon

The proton is chosen because, beyond generic baryon structure, it demands a nontrivial flavor pattern, electric charge, spin, and experimentally meaningful form factors.

Let three persistent quark components $A,B,C$ have color frames and normalized color columns $c_{A},c_{B},c_{C}$. The invariant color volume is $$S_{ABC}=\epsilon_{ijk}c_{A}^{i}c_{B}^{j}c_{C}^{k}
  =\det[c_{A}\ c_{B}\ c_{C}].$$ Under a common $g\in SU(3)$, $S\mapsto\det(g)S=S$. Its squared magnitude is the Gram determinant $$\lvert S_{ABC}\rvert^{2}=\det(C^{\dagger}C)\in[0,1].$$ The value one means the three color directions form an orthonormal frame. Their normalized wedge is then the unique $\Lambda^{3}\mathbb{C}^{3}$ singlet. The proposed proton certificate is the conjunction:

- three persistent odd rank-three quark clusters with accepted triangle-anchor certificates;

- one persistent bound supercluster containing them;

- normalized color wedge with $\lvert S_{ABC}\rvert^{2}\approx 1$ and vanishing net color flux;

- flavor spectrum with the `uud` occupation pattern, in the sense of the still-hypothetical isospin-doublet construction of Section <a href="#sec:quarks" data-reference-type="ref" data-reference="sec:quarks">10</a>;

- oriented baryon flux $B=1$ in the relative-winding sense of Section <a href="#sec:transport" data-reference-type="ref" data-reference="sec:transport">9</a>;

- Gauss flux $Q=+1$;

- a sharp total-space spin readout: $\langle J^{2}\rangle=3/4$ with $\operatorname{Var}(J^{2})
        =\langle (J^{2})^{2}\rangle-\langle J^{2}\rangle^{2}\approx 0$, a reference-normalized $2\pi\mapsto -1$, and, where applicable, an accepted spin lift;

- finite radius and stable spectral mass/form-factor readouts; and

- stability of every dimensionless certificate under refinement.

The variance condition is essential and is exactly computable: $J^{2}$ is quartic and $(J^{2})^{2}$ octic in the fermion operators, so on any quasi-free state both are finite Wick sums over the covariance matrix. A Gaussian state can in special cases be an exact $J^{2}$ eigenstate, so quasi-free failure is not automatic; but a generic Slater determinant with $\langle J^{2}\rangle=3/4$ need not be spin-$1/2$, and the variance is what separates a proton certificate from an accidental expectation value. Failure of the sharp certificate across the entire accepted covariance-only class is the structural branch point of Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>.

None of these conditions should be included as an emergence target. The proton is found only if the base geometric optimization produces a component satisfying them. Targeted runs remain valuable as existence and obstruction experiments, but must be labeled as synthesis rather than emergence.

<figure id="fig:firewall">

<figcaption>No-feedback emergence protocol. Only the base geometric objective drives optimization. In the certificates-blind backreaction mode the carried state’s energy density may enter that objective; Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a> records that this remains inside the quasi-free class. Cluster, fiber, color, exchange, and baryon observables are computed from accepted snapshots and cannot feed back into the objective in either emergence mode. Targeted synthesis is a separate, explicitly labeled mode.</figcaption>
</figure>

# The master recursive construction

Let $\mathcal{R}_{0}(\lambda)=L_{0}-\lambda I$ denote the microscopic one-particle response pencil. At every scale: $$\boxed{
\begin{aligned}
  P_{\ell} &= \mathrm{PersistentPartition}(\mathcal{R}_{\ell}),\\
  E_{v}^{\ell+1} &= \text{certified isolated subspace of } C_{v}^{\ell},\\
  \mathcal{R}_{\ell+1}(\lambda) &= \mathrm{Feshbach}_{P_{\ell}}(\mathcal{R}_{\ell}(\lambda)),\\
  V_{vw}^{\ell+1} &= \mathrm{Polar}_{r_{v}}
    \bigl((\Phi_{v}^{\ell})^{\dagger}WT_{vw}\Phi_{w}^{\ell}\bigr),\\
  \mathfrak{h}_{\ell+1} &= \mathbin{\boxplus}_{v}E_{v}^{\ell+1},\qquad
    J_{\ell+1}:\mathfrak{h}_{\ell+1}\to C(K),\qquad
    G_{\ell+1}=J_{\ell+1}^{\dagger}WJ_{\ell+1},\\
  \mathcal{H}_{\ell+1} &= \mathcal{F}_{-}\bigl(\mathfrak{h}_{\ell+1}\bigr),
\end{aligned}}$$ where $\mathbin{\boxplus}$ is the abstract labeled sum: one summand per retained fiber, with no claim that the geometric subspaces are independent inside $C(K)$.

The geometric subspaces $E_{v}\subset C(K)$ of adjacent components may overlap on shared interface cells, so their internal sum need not be direct. The recursion therefore never asserts $\bigoplus_{v}E_{v}\subset C(K)$. It forms the abstract labeled sum, carries the embedding $J_{\ell+1}$ and its Gram matrix $G_{\ell+1}$ exactly, and proceeds by exactly one of three declared options: carry $G$ in every subsequent formula; certify $\lVert G-I\rVert\le\varepsilon$ and propagate $\varepsilon$ through the composable amplitude budget of Section <a href="#sec:interactions" data-reference-type="ref" data-reference="sec:interactions">6</a>; or quotient $\ker G$ and restate the fiber ranks. A sheaf-stalk decomposition that assigns interface modes to link stalks is a valid realization of the same requirement (Hansen and Ghrist 2019), but it is not necessary.

At $\lambda=0$ the response step is the exact supported static Schur complement. For a nonzero band it is the exact energy-dependent pencil; a cached linear $\mathcal{R}_{\ell+1}$ is an AMLS/component-mode surrogate with a declared frequency window and residual. The transport rank is generic; only an anchored accepted rank-three fiber receives the color interpretation. This recursion supplies the response network, retained stalk, derived transport, and expanding state space without claiming that every coarse level is literally a new simplicial complex.

# Exactness and performance principles

The simulation should prefer an exact structural identity over a general dense numerical operation whenever both compute the same object:

- in the quasi-free sector, evolve the covariance matrix by $i\dot{\Gamma}=[h,\Gamma]$ and evaluate every polynomial certificate by Wick contraction; materialize a Fock vector only for oracle tests or explicitly non-Gaussian boundary data;

- use sparse static and shifted Schur/Feshbach solves, not explicit dense inverses, and use AMLS when a reusable linear band surrogate is needed;

- use Künneth sums only for actual product complexes, and occupation subset sums for $d\Gamma(L)$, not diagonalization of an eager Fock matrix;

- use exterior bit parity for exchange signs, not sampled phases;

- use exact $3\times 3$ determinants and the fixed $F_{3}$ frame only after a rank-three band passes its triangle-anchor certificate;

- use analytic Regge/Hodge derivatives and Wirtinger gradients, not finite differences;

- use Smith normal form for integer homology and a spectral threshold only as a cross-check;

- use matrix-determinant/Woodbury updates for local cobordism changes;

- cache component factorizations and invalidate only affected stars;

- keep tensor products lazy and block-sparse by occupation/parity;

- use $U(r)$ polar transport at generic rank, retaining determinant-line and projective/center data at $r=3$; and

- attach frequency window, residual, gap, leakage, signature, and condition-number certificates to every iterative eigensolve or reduction.

The exact route is not only faster. It prevents a numerical tolerance from becoming an undocumented physical postulate.

# Prior art and boundary of novelty

No single cited work establishes the full recursive spectral-fiber proposal. The construction is a synthesis of several mature ideas, and its novelty should be evaluated at the joins. Table <a href="#tab:priorart" data-reference-type="ref" data-reference="tab:priorart">3</a> states the boundary explicitly.

<div id="tab:priorart">

| Topic                                 | Established prior art                                                                                                                                                                                                                                                                                                                                                           | Additional claim made here                                                                                                                                                                                                                                                    |
|:--------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Topic                                 | Established prior art                                                                                                                                                                                                                                                                                                                                                           | Additional claim made here                                                                                                                                                                                                                                                    |
| continued on next page                |                                                                                                                                                                                                                                                                                                                                                                                 |                                                                                                                                                                                                                                                                               |
| Simplicial geometry and Hodge spectra | Regge curvature, discrete exterior calculus, and combinatorial Laplace spectra (Regge 1961; Desbrun et al. 2005; Horak and Jost 2013).                                                                                                                                                                                                                                          | Use one jointly optimized Regge–Hodge complex as the only microscopic carrier for geometry and quantum readouts.                                                                                                                                                              |
| Coarse response and recursive modules | Kron/Schur reduction, Feshbach maps, component-mode synthesis/AMLS, cellular sheaf Laplacians, modular communities, and self-similar network renormalization (Craig and Bampton 1968; Bennighof and Lehoucq 2004; Bach et al. 2003; Dörfler and Bullo 2013; Hansen and Ghrist 2019; Reichardt and Bornholdt 2006; Song, Havlin, and Makse 2005; Fortunato and Barthélemy 2007). | Treat a persistent component as a static response vertex; retain nonzero bands through shifted or certified component-mode reduction; recurse in operator-valued response networks, using a sheaf realization only when its factorization is certified.                       |
| Quasi-free many-body calculus         | Second quantization, generalized Hartree–Fock theory, and quasi-free/Gaussian state methods (Berezin 1966; Bach, Lieb, and Solovej 1994).                                                                                                                                                                                                                                       | Run the entire particle-certificate layer on the covariance matrix; prove mean-field geometry backreaction is Gaussian-closed; treat sharp-certificate failure as a structural dichotomy rather than a numerical shortfall.                                                   |
| Geometric gauge transport             | Berry and Wilczek–Zee holonomy, overlap-based lattice links, connection Laplacians/vector diffusion, and Wilson loops (Berry 1984; Wilczek and Zee 1984; Fukui, Hatsugai, and Suzuki 2005; Wilson 1974; Singer and Wu 2012).                                                                                                                                                    | Derive $U(r)$ transport from component frames; at anchored rank three retain the determinant line, projective $SU(3)/\mathbb{Z}_{3}$ class, and any chosen center lift, assigning no independent gauge link variable.                                                         |
| Color and fermion structure           | Quark/color triplets, exterior Fock/second quantization, topological exchange phases, and spin structures (Gell-Mann 1964; Han and Nambu 1965; Greenberg 1964; Berezin 1966; Laidlaw and DeWitt 1971; Leinaas and Myrheim 1977; Lawson and Michelsohn 1989; Cimasoni and Reshetikhin 2007).                                                                                     | Realize $\mathbf{1}\oplus\mathbf{3}\oplus\overline{\mathbf{3}}\oplus\mathbf{1}$ on three oriented edge modes, anchor abstract rank-three fibers to oriented faces, and test exchange by a Berry-cancelled determinant-line interferometer plus structural permutation parity. |
| Scale composition and boundaries      | TQFT cobordisms, general-boundary state assignments, categorical tensor composition, second quantization, and entanglement renormalization (Atiyah 1988; Oeckl 2003; Abramsky and Coecke 2004; Berezin 1966; Vidal 2007).                                                                                                                                                       | Keep simplicial gluing at the one-particle level, then build finite Fock stages functorially and require vacuum-embedding compatibility under refinement.                                                                                                                     |
| Kähler–Dirac boundary                 | Differential-form fermions and their taste structure (Becher and Joos 1982; Butt et al. 2021).                                                                                                                                                                                                                                                                                  | Do not infer Kähler–Dirac tastes from occupation exterior algebra; test for them only if the one-particle field is promoted to inhomogeneous cochains with a Kähler–Dirac operator.                                                                                           |
| Spectral spacetime                    | Diffusion spectral dimension on ensembles of simplicial geometries (Ambjørn, Jurkiewicz, and Loll 2005).                                                                                                                                                                                                                                                                        | Test whether many interacting Tessera cobordisms yield a stable four-dimensional spectral window while simultaneously supporting the particle certificates.                                                                                                                   |

Established ingredients and the additional Tessera claim.

</div>

<figure id="fig:priorart">

<figcaption>Relationship to prior art. The blue inputs are established research programs; the green center is the proposed Tessera synthesis; the amber outputs are new physical identifications and must be validated independently. An arrow denotes conceptual inheritance, not a proof of the downstream claim.</figcaption>
</figure>

# Falsification program

The formulation fails, or must be narrowed, if any of the following persists under refinement and tighter numerical certification:

1.  **No persistent rank-three clusters.** Certified components appear, but their localized fiber rank or spectral gap is unstable.

2.  **No oriented color anchor.** A rank-three band appears, but its calibrated anchor profile (Section <a href="#sec:quarks" data-reference-type="ref" data-reference="sec:quarks">10</a>) vanishes or drifts.

3.  **No faithful coarse response.** Schur-reduced components fail to reproduce static response, or shifted/AMLS reduction fails over its declared frequency band, within the stated residual.

4.  **No derived gauge covariance.** Wilson values depend on the local spectral frame after leakage is controlled, or the determinant and $\mathbb{Z}_{3}$ center sectors cannot be made path-consistent.

5.  **No fermion holonomy.** The Berry-cancelled exchange ratio or structural permutation sign does not give `-1`, or the verdict changes under relabeling.

6.  **No spinor rotation.** Exchange works but the reference-normalized $2\pi$ physical rotation does not give `-1`; in a manifold-like continuum claim, failure of a consistent spin lift is also decisive.

7.  **No inductive compatibility.** Adding vacuum modes changes already-computed amplitudes by a nonvanishing amount.

8.  **No quasi-free proton.** Every other certificate is met inside the covariance-only theory, but $\operatorname{Var}(J^{2})$ fails to converge to zero on every accepted candidate across refinement. This outcome is a branch point rather than a refutation of the geometry: it mandates adopting exactly one of the non-Gaussian mechanisms of Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>, as an explicit scope decision, before any proton claim is made.

9.  **No unforced baryon.** Targeted synthesis can build the certificates, but the stationary geometric ensemble never produces them without a proton-specific term.

10. **No continuum stability.** Dimensionless color, parity, charge, spin, and amplitude certificates drift rather than converge with refinement.

11. **Unexpected multiplicity.** A robust flavor/taste degeneracy is neither predicted by the stated one-particle operator nor stable enough to be promoted to an emergent flavor mechanism.

Holes may re-emerge and may correlate with some phases, but no claim in this paper depends on them doing so.

# Conclusion

The geometry is economical and precise: an edge carries a two-level mode, not an independently stored pure state; a quark candidate is a modular spectral component whose rank-three band is anchored to oriented faces by a calibrated profile; its transport is a certified $U(3)$ overlap with retained determinant-line and projective color data; and its fermionic sign is the exterior grading, checked dynamically only after cancelling ordinary Berry phase. Simplicial gluing constructs the one-particle operator and second quantization constructs the expanding Fock state. Three accepted components form a baryon through their normalized color wedge; the proton is the sharpest test because it also demands the correct determinant-line flux, charge, flavor, and a variance-certified spin response.

The exact claims are limited to their proper domains: static Schur response, energy-dependent Feshbach isospectrality, exterior/CAR algebra, second-quantized direct-sum composition, gauge covariance of accepted transport, and closure of the quasi-free class under every generator the model currently possesses. That last identity organizes the programme. Classical or mean-field geometry backreaction does not leave the Gaussian manifold, so the covariance matrix is not an approximation tier: it is the exact state representation of the present theory, and every particle certificate — including $\langle J^{2}\rangle$ and its variance — is a finite Wick sum over it.

What remains genuinely open is whether Tessera’s unforced Regge–Hodge dynamics produces the required anchored clusters, low-leakage holonomies, relative determinant windings, and sharp spin response. A tempting Hellmann–Feynman/envelope argument does not by itself make the first variation of transport Gram defect vanish at a Regge–Hodge stationary point, because the defect is not the optimized functional; Tessera will therefore measure that correlation as a conjectural scaling law rather than cite stationarity as a theorem. The decisive question is the dichotomy of Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>: either an exact covariance-only proton exists, or a genuinely non-Gaussian, geometry-mediated interaction is required. Either outcome is a result. The first makes the particle layer exactly and polynomially certifiable; the second would be the first internal evidence that the geometry must supply a true interaction term, through one of the five mechanisms this paper names.

# Repository evidence

- Tessera amplitude and obstruction results, `cobordism-results.md`.

- Tessera spectral-dimension status, `h_ds4_status.md`.

- Tessera interaction-history construction, `interaction-history-monte-carlo.md`.

- Existing total-space spin obstruction, `joint_proton_spin_findings.md`.

- Existing fixed-partition modularity implementation, `ModularityOptimizer.h`.

- Current visualization and joint-stationarity experiment, `proton_animation.py`.

- External-review dispositions and exact-claim ledger, review rounds one and two, `referee response`.

# Prior-art references

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-abramsky2004categorical" class="csl-entry">

Abramsky, Samson, and Bob Coecke. 2004. “A Categorical Semantics of Quantum Protocols.” In *Proceedings of the 19th Annual IEEE Symposium on Logic in Computer Science*, 415–25. <https://doi.org/10.1109/LICS.2004.1319636>.

</div>

<div id="ref-ambjorn2005spectral" class="csl-entry">

Ambjørn, Jan, Jerzy Jurkiewicz, and Renate Loll. 2005. “Spectral Dimension of the Universe.” *Physical Review Letters* 95: 171301. <https://doi.org/10.1103/PhysRevLett.95.171301>.

</div>

<div id="ref-atiyah1988tqft" class="csl-entry">

Atiyah, Michael F. 1988. “Topological Quantum Field Theories.” *Publications Mathématiques de l’IHÉS* 68: 175–86. <https://doi.org/10.1007/BF02698547>.

</div>

<div id="ref-bach2003feshbach" class="csl-entry">

Bach, Volker, Thomas Chen, Jürg Fröhlich, and Israel Michael Sigal. 2003. “Smooth Feshbach Map and Operator-Theoretic Renormalization Group Methods.” *Journal of Functional Analysis* 203 (1): 44–92. <https://doi.org/10.1016/S0022-1236(03)00057-0>.

</div>

<div id="ref-bach1994hartreefock" class="csl-entry">

Bach, Volker, Elliott H. Lieb, and Jan Philip Solovej. 1994. “Generalized Hartree–Fock Theory and the Hubbard Model.” *Journal of Statistical Physics* 76 (1–2): 3–89. <https://doi.org/10.1007/BF02188656>.

</div>

<div id="ref-becher1982dirackahler" class="csl-entry">

Becher, Peter, and Hans Joos. 1982. “The Dirac–kähler Equation and Fermions on the Lattice.” *Zeitschrift für Physik C* 15: 343–65. <https://doi.org/10.1007/BF01614426>.

</div>

<div id="ref-bennighof2004amls" class="csl-entry">

Bennighof, Jeffrey K., and Richard B. Lehoucq. 2004. “An Automated Multilevel Substructuring Method for Eigenspace Computation in Linear Elastodynamics.” *SIAM Journal on Scientific Computing* 25 (6): 2084–2106.

</div>

<div id="ref-berezin1966second" class="csl-entry">

Berezin, Felix A. 1966. *The Method of Second Quantization*. New York: Academic Press.

</div>

<div id="ref-berry1984phase" class="csl-entry">

Berry, Michael V. 1984. “Quantal Phase Factors Accompanying Adiabatic Changes.” *Proceedings of the Royal Society of London A* 392 (1802): 45–57. <https://doi.org/10.1098/rspa.1984.0023>.

</div>

<div id="ref-butt2022kahler" class="csl-entry">

Butt, Nouman, Simon Catterall, Arnab Pradhan, and Goksu Can Toga. 2021. “Anomalies and Symmetric Mass Generation for kähler–Dirac Fermions.” *Physical Review D* 104: 094504. <https://doi.org/10.1103/PhysRevD.104.094504>.

</div>

<div id="ref-cimasoni2007dimers" class="csl-entry">

Cimasoni, David, and Nicolai Reshetikhin. 2007. “Dimers on Surface Graphs and Spin Structures. i.” *Communications in Mathematical Physics* 275: 187–208. <https://doi.org/10.1007/s00220-007-0302-7>.

</div>

<div id="ref-craig1968coupling" class="csl-entry">

Craig, Roy R., Jr., and Mervyn C. C. Bampton. 1968. “Coupling of Substructures for Dynamic Analyses.” *AIAA Journal* 6 (7): 1313–19. <https://doi.org/10.2514/3.4741>.

</div>

<div id="ref-desbrun2005dec" class="csl-entry">

Desbrun, Mathieu, Anil N. Hirani, Melvin Leok, and Jerrold E. Marsden. 2005. “Discrete Exterior Calculus.” <https://arxiv.org/abs/math/0508341>.

</div>

<div id="ref-dorfler2013kron" class="csl-entry">

Dörfler, Florian, and Francesco Bullo. 2013. “Kron Reduction of Graphs with Applications to Electrical Networks.” *IEEE Transactions on Circuits and Systems I: Regular Papers* 60 (1): 150–63. <https://doi.org/10.1109/TCSI.2012.2215780>.

</div>

<div id="ref-fortunato2007resolution" class="csl-entry">

Fortunato, Santo, and Marc Barthélemy. 2007. “Resolution Limit in Community Detection.” *Proceedings of the National Academy of Sciences* 104 (1): 36–41. <https://doi.org/10.1073/pnas.0605965104>.

</div>

<div id="ref-fukui2005chern" class="csl-entry">

Fukui, Takahiro, Yasuhiro Hatsugai, and Hiroshi Suzuki. 2005. “Chern Numbers in Discretized Brillouin Zone: Efficient Method of Computing (Spin) Hall Conductances.” *Journal of the Physical Society of Japan* 74 (6): 1674–77. <https://doi.org/10.1143/JPSJ.74.1674>.

</div>

<div id="ref-gellmann1964quark" class="csl-entry">

Gell-Mann, Murray. 1964. “A Schematic Model of Baryons and Mesons.” *Physics Letters* 8 (3): 214–15. <https://doi.org/10.1016/S0031-9163(64)92001-3>.

</div>

<div id="ref-greenberg1964paraquark" class="csl-entry">

Greenberg, O. W. 1964. “Spin and Unitary-Spin Independence in a Paraquark Model of Baryons and Mesons.” *Physical Review Letters* 13: 598–602. <https://doi.org/10.1103/PhysRevLett.13.598>.

</div>

<div id="ref-han1965color" class="csl-entry">

Han, Moo-Young, and Yoichiro Nambu. 1965. “Three-Triplet Model with Double SU(3) Symmetry.” *Physical Review* 139: B1006–10. <https://doi.org/10.1103/PhysRev.139.B1006>.

</div>

<div id="ref-hansen2019sheaves" class="csl-entry">

Hansen, Jakob, and Robert Ghrist. 2019. “Toward a Spectral Theory of Cellular Sheaves.” *Journal of Applied and Computational Topology* 3: 315–58. <https://doi.org/10.1007/s41468-019-00038-7>.

</div>

<div id="ref-horak2013spectra" class="csl-entry">

Horak, Danijela, and Jürgen Jost. 2013. “Spectra of Combinatorial Laplace Operators on Simplicial Complexes.” *Advances in Mathematics* 244: 303–36. <https://doi.org/10.1016/j.aim.2013.05.005>.

</div>

<div id="ref-laidlaw1971indistinguishable" class="csl-entry">

Laidlaw, Michael G. G., and Cécile Morette DeWitt. 1971. “Feynman Functional Integrals for Systems of Indistinguishable Particles.” *Physical Review D* 3: 1375–78. <https://doi.org/10.1103/PhysRevD.3.1375>.

</div>

<div id="ref-lawson1989spin" class="csl-entry">

Lawson, H. Blaine, Jr., and Marie-Louise Michelsohn. 1989. *Spin Geometry*. Princeton: Princeton University Press.

</div>

<div id="ref-leinaas1977identical" class="csl-entry">

Leinaas, Jon M., and Jan Myrheim. 1977. “On the Theory of Identical Particles.” *Il Nuovo Cimento B* 37: 1–23. <https://doi.org/10.1007/BF02727953>.

</div>

<div id="ref-loukas2019graph" class="csl-entry">

Loukas, Andreas. 2019. “Graph Reduction with Spectral and Cut Guarantees.” *Journal of Machine Learning Research* 20 (116): 1–42. <https://jmlr.org/papers/v20/18-680.html>.

</div>

<div id="ref-oeckl2003boundary" class="csl-entry">

Oeckl, Robert. 2003. “A ‘General Boundary’ Formulation for Quantum Mechanics and Quantum Gravity.” *Physics Letters B* 575: 318–24. <https://doi.org/10.1016/j.physletb.2003.08.043>.

</div>

<div id="ref-regge1961" class="csl-entry">

Regge, Tullio. 1961. “General Relativity Without Coordinates.” *Il Nuovo Cimento* 19: 558–71. <https://doi.org/10.1007/BF02733251>.

</div>

<div id="ref-reichardt2006community" class="csl-entry">

Reichardt, Jörg, and Stefan Bornholdt. 2006. “Statistical Mechanics of Community Detection.” *Physical Review E* 74: 016110. <https://doi.org/10.1103/PhysRevE.74.016110>.

</div>

<div id="ref-singer2012vector" class="csl-entry">

Singer, Amit, and Hau-Tieng Wu. 2012. “Vector Diffusion Maps and the Connection Laplacian.” *Communications on Pure and Applied Mathematics* 65 (8): 1067–1144. <https://doi.org/10.1002/cpa.21395>.

</div>

<div id="ref-song2005selfsimilar" class="csl-entry">

Song, Chaoming, Shlomo Havlin, and Hernán A. Makse. 2005. “Self-Similarity of Complex Networks.” *Nature* 433: 392–95. <https://doi.org/10.1038/nature03248>.

</div>

<div id="ref-vidal2007entanglement" class="csl-entry">

Vidal, Guifré. 2007. “Entanglement Renormalization.” *Physical Review Letters* 99: 220405. <https://doi.org/10.1103/PhysRevLett.99.220405>.

</div>

<div id="ref-wilczekzee1984gauge" class="csl-entry">

Wilczek, Frank, and A. Zee. 1984. “Appearance of Gauge Structure in Simple Dynamical Systems.” *Physical Review Letters* 52: 2111–14. <https://doi.org/10.1103/PhysRevLett.52.2111>.

</div>

<div id="ref-wilson1974confinement" class="csl-entry">

Wilson, Kenneth G. 1974. “Confinement of Quarks.” *Physical Review D* 10: 2445–59. <https://doi.org/10.1103/PhysRevD.10.2445>.

</div>

</div>
