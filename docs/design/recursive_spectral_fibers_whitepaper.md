<div class="titlepage">

<div class="center">

Recursive Spectral Fibers on Simplicial Cobordisms

A geometric program for quarks, color, fermion statistics, Fock space, and baryons

Tessera – cobordism programme

**Abstract**

</div>

This paper proposes a single geometric formulation for the particle content already suggested by Tessera’s cobordism experiments. A persistent, spectrally certified connected simplicial component is treated as one effective vertex at the next resolution. The component’s selected localized Hodge eigenspace is its fiber, and the couplings between components induce transport between those fibers. Repeating this operation produces a nested, potentially fractal hierarchy of complexes. No particle label or auxiliary lattice is introduced, and the microscopic fields are exactly two per edge: a complex squared length, which alone determines the metric weights, and a complex $\mathbb{C}^{*}$ connection phase — an independent Abelian link field that twists the covariant hopping operator — together with one two-level occupation mode. The generally entangled state lives on the exterior Fock space of all active edge modes. Exchange signs, the derived color transport, and observables are obtained from that state and the Hodge/Regge operators already present in the construction.

The proposal has a substantial exact core. Static Schur reduction proves when a component may be replaced by a response vertex without changing supported boundary quadratic energies. Nonzero spectral bands instead use the energy-dependent Feshbach–Schur map, or a certified Craig–Bampton/AMLS linear surrogate. Simplicial gluing acts on the one-particle chain space; fermionic second quantization then turns direct sums into graded tensor products and coupling blocks into hopping terms. Every generator so obtained is quadratic, so the dynamics is exactly quasi-free; on sectors whose metric certificate is verified the state is carried without loss by a covariance matrix, Wick’s theorem evaluates every polynomial certificate, and mean-field geometry backreaction provably stays Gaussian. Three oriented edge-mode factors form the exact exterior algebra $\Lambda^{\bullet}\mathbb{C}^{3}=\mathbf{1}\oplus
\mathbf{3}\oplus\overline{\mathbf{3}}\oplus\mathbf{1}$; the one-occupation sector is a color qutrit, its bilinears close $\mathfrak{su}(3)$, the three-occupation sector is a color singlet, and the grading gives the fermionic exchange sign and canonical anticommutation relations. A rank-$r$ connection is derived from overlap of neighboring spectral frames. At rank three its determinant line and projective $SU(3)/\mathbb{Z}_{3}$ transport are retained rather than choosing an unrecorded cube-root branch. Closed holonomies are gauge-invariant observables rather than new degrees of freedom. Successive cobordism interactions generate the finite stages of an inductive-limit Fock space.

The physical identification remains a hypothesis to be tested. A quark is proposed to be a persistent, odd-parity, rank-three spectral fiber anchored to oriented faces of a certified component; a proton is three such components bound into one persistent supercomponent, occupying a normalized color wedge, carrying baryon number $+1$, electric charge $+1$, and a sharpness-certified total-space spin-$1/2$ readout. The paper separates exact identities, conditional theorems with certified hypotheses, numerical evidence, and new falsifiable conjectures, and it states the sharpest open question as a dichotomy: either an exact covariance-only proton exists, or a genuinely non-Gaussian, geometry-mediated interaction is required.

</div>

> The rendered vector diagrams are preserved in the LaTeX/PDF edition; this Markdown edition is the searchable text companion.

# Epistemic status and design constraint

Four kinds of statement are deliberately distinguished:

1.  **Exact identity** — follows algebraically from the stated finite complex, orientation, and pairing, with no further hypothesis.

2.  **Conditional theorem** — exact under hypotheses that must be explicitly certified wherever the theorem is used; the certificate is part of the claim.

3.  **Numerical evidence** — measured by a stated experiment and reported self-containedly with its method, scope, and residual.

4.  **Proposed physical identification** — a new hypothesis, or a calibrated observable, with an explicit falsification test.

The governing constraint is parsimony. The ontology is limited to:

- an oriented simplicial complex and its cobordisms;

- on each edge, a complex squared length and a complex $\mathbb{C}^{*}$ connection phase — two distinct fields, because the connection is gauge-variant and the geometry is not — together with one two-level occupation mode;

- incidence, Hodge, and Regge operators derived from that data;

- a generally entangled boundary/Fock state on those modes; and

- simplicial gluing followed by fermionic second quantization.

The declared microscopic fields end there. Spectral fibers, color frames, the derived rank-three color transport, Wilson loops, particle sectors, and coarse vertices are *derived views* of that same data. They are not separately sampled fields. This is important scientifically: adding further independent fields could fit a desired answer, while deriving every readout from the declared data leaves the construction falsifiable. One consequence is recorded in Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>: every generator this ontology currently supplies is quadratic after second quantization, so the reachable states are exactly the quasi-free class together with whatever non-Gaussian data is fed at the boundary.

A second constraint governs how the four kinds of statement are kept apart in practice. Wherever an exact structural identity and an approximation determine the same object, the identity is what the claim rests on; an approximation is admissible only as a conditional theorem, carrying the certificates that bound it — for a spectral reduction, its frequency window, residual, gap, leakage, signature, and conditioning. The reason is not economy. An uncertified tolerance silently becomes a physical postulate: the reader cannot tell which part of a reported number is the theory and which part is the numerics, and no falsification test can be run against a quantity whose error is undeclared.

<figure id="fig:concept">

<figcaption>Concept map for the recursive complex construction. Colors encode epistemic status, not physical sectors: blue is established or exact machinery, green is a derived observable, and amber is a proposed physical identification.</figcaption>
</figure>

# Present evidence in Tessera

Two existing results motivate the construction.

First, the state-operation-cobordism experiments show that the Hodge-carried register is a scaled isometry to machine precision — its Gram is the identity once one global scale is fixed by the anchor — and that its spectral value reproduces the quantum transition amplitude for every operation that the tested geometry actually carries. Generic fixed-complexity operations can remain obstructed, and the obstruction is visible both as a residual floor and as leakage from the carried subspace. The claim is therefore not that every finite complex realizes every gate; it is that a realized, isometrically embedded register computes the corresponding amplitude. The protocol reports the Gram residual and the carried-subspace leakage for every tested operation.

Second, interaction-history complexes exhibit a stable near-four-dimensional spectral regime. The strongest reported measurements approach, but do not yet prove, an exact spectral dimension of four; the return-probability estimator, its finite-size window, and its caveats accompany those measurements and bound what they establish. Diffusion-based spectral dimension on simplicial quantum geometries has important precedent in causal dynamical triangulations \[1\]; the Tessera evidence is an independent result for a different construction and should be compared at the level of the return-probability estimator and its finite-size window.

A third, more preliminary observation comes from joint Regge–Hodge stationarity experiments seeded with the phase pattern $\{1,\omega,\omega^{2}\}$, whose singlet diagnostics are evaluated while the stationarity objective changes the complex. The construction does not force register holes to appear, and holes have not re-emerged under the unforced dynamics. That negative result is useful: the proposed quark should therefore not be defined as a hole. It will be sought as a persistent modular spectral cluster, while Betti numbers remain independent topological observables.

# The microscopic geometric state

Let $K$ be a finite oriented simplicial complex. For every edge $e$, store the complex squared length $$z_{e}=\rho_{e}e^{i\theta_{e}}\in\mathbb{C}$$ and, separately, the connection phase $$\varphi_{e}\in\mathbb{C},$$ and attach the two-level occupation factor $\mathcal{H}_{e}=\operatorname{span}\{\lvert 0\rangle_{e},\lvert 1\rangle_{e}\}$.

The two edge fields are distinct, and the reason is gauge invariance. The connection is an independent Abelian link field: on the oriented edge $e$ from $x$ to $y$ it supplies the link variable $$U_{xy}=e^{i\varphi_{e}}\in\mathbb{C}^{*},\qquad U_{yx}=U_{xy}^{-1},$$ and under a gauge transformation $g:K_{0}\to\mathbb{C}^{*}$, $$U_{xy}\longmapsto g_{x}^{-1}U_{xy}\,g_{y},
  \qquad\text{equivalently}\qquad
  \varphi\longmapsto\varphi+d\chi,\quad g=e^{i\chi},$$ while $z_{e}$ and every metric weight $W(z)$ built from it remain invariant. The geometry must not transform: were $\varphi_{e}$ taken to be $\arg z_{e}$, a gauge transformation would change the squared length — that is, it would change the metric. So the phase is carried as its own field, and it twists the hopping rather than rescaling the weight.

The distinction is not cosmetic. The geometric Hodge operator of Section <a href="#sec:state" data-reference-type="ref" data-reference="sec:state">3</a> is built from $z$ alone and is blind to $\varphi$: its row sums vanish at degree zero, so the constant is harmonic and $\dim\ker L_{0}=b_{0}$ at any weights, positive, signed or complex. The connection enters a second and distinct operator, the Aharonov–Bohm operator, whose off-diagonal carries the link phase while its diagonal carries the magnitude; it is that operator, and not $L_{k}$, whose zero mode a nonzero flux lifts. Writing $\varphi$ into the metric weight instead would be an error twice over: it would make the metric gauge-variant, which is the very thing the separation exists to prevent, and it would destroy the derived form of $L_{k}$, since a rescaled weight is not a conjugation and does not preserve the spectrum. A gauge transformation acts on the twisted operator by the conjugation $\operatorname{diag}(g)^{-1}(\cdot)\operatorname{diag}(g)$, leaving its spectrum fixed and the geometry untouched. This microscopic Abelian connection is also distinct from the derived rank-three color transport of Section <a href="#sec:transport" data-reference-type="ref" data-reference="sec:transport">9</a>: the former is declared edge data with structure group $\mathbb{C}^{*}$, while the latter is reconstructed from neighboring spectral frames and carries its own certificates. Saying that an edge “carries a qubit” means that it carries the local mode algebra above, not that the global state is forced to be a product of normalized vectors $q_{e}$.

Because $\varphi_{e}$ is complex, $e^{i\varphi}=e^{i\operatorname{Re}\varphi}e^{-\operatorname{Im}\varphi}$, and the structure group is $\mathbb{C}^{*}=U(1)\times\mathbb{R}^{+}$: a compact part and a non-compact one. Only the compact part has winding, so only the compact part quantizes; the non-compact part acts as a local scale and carries no quantum number. A complex connection therefore twists by a similarity rather than by a unitary, and together with the complex edge geometry this fixes the operator class. Four consequences follow and are used throughout:

- $L$ is generically *non-normal* — beyond even the Hermitian-indefinite (Krein) special case — so matched left and right frames with $\Psi^{\dagger}W\Phi=I$ are mandatory rather than a fallback;

- eigenvalues are properly complex, so band isolation must be measured as separation in the complex plane and never to a sort-order neighbour;

- band projectors are oblique, so $\Gamma=P$ supplies the covariance of an actual quasi-free state only where the band’s restricted metric passes the positivity certificate of Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>; a band that fails it does not supply the quasi-free covariance used by this construction, and every readout needing one refuses on that band with the reason named; and

- under a similarity $L\mapsto SLS^{-1}$ eigenvalues are invariant while eigenvectors are only covariant, so observables must be spectral or holonomy-valued. This strengthens rather than weakens the discipline the rest of the paper already imposes on Wilson loops and the determinant line.

Three layers of terminology are used from here on, and each carries a different word. The *microscopic configuration* is the edge data $(K,z,\varphi)$: a coordinate on the space of operators, gauge-redundant, and not itself a state — an edge datum is not an amplitude. The *one-particle state* is the spectral data of $L(z,\varphi)$: its eigenvalues together with its selected eigenspaces and, in a defective non-normal sector, the associated spectral projections and Jordan structure, entering below as band projectors. This is the gauge-invariant content of the edge data, and it is what the recursion transports. The *many-body state* is a vector or density operator on the Fock space built over those modes: the one-particle spectral data fix which modes exist, and the occupation and coherence data on them fix the many-body state, represented without loss by its covariance matrix exactly on the certified quasi-free sector of Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>.

Writing $\mathfrak{h}_{K}=\operatorname{span}\{\lvert e\rangle : e\in K_{1}\}$ for the one-particle edge space, the microscopic quantum carrier is $$\mathcal{H}_{K}=\mathcal{F}_{-}(\mathfrak{h}_{K})=\Lambda^{\bullet}\mathfrak{h}_{K}
  \;\cong\;\widehat{\bigotimes}_{e\in K_{1}}\mathcal{H}_{e},$$ and a boundary many-body state is a vector or density operator on $\mathcal{H}_{K}$. It may be entangled. A one-particle color state $a^{\dagger}_{\phi}\lvert 0\rangle$ and the nonseparable proton-spin sectors are therefore native states, not exceptions to the ontology. For an isolated occupied band with projector $P$, the corresponding quasi-free reference state has covariance $$\Gamma_{ef}=\langle a^{\dagger}_{f}a_{e}\rangle=P_{ef},
  \qquad \langle n_{e}\rangle=P_{ee}.$$ Thus a per-edge Bloch vector or occupation is a derived marginal/readout. The quasi-free state is a useful analytic baseline, not a restriction of the state space: the lazy Fock construction of Section <a href="#sec:fock" data-reference-type="ref" data-reference="sec:fock">12</a> can represent explicitly non-Gaussian sectors. Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a> records, however, that no generator currently present in the model produces such sectors from Gaussian data; until one of the mechanisms listed there is adopted, non-Gaussianity can enter only as boundary data.

Let $$\partial_{k}:C_{k}(K)\longrightarrow C_{k-1}(K),
  \qquad \partial_{k-1}\partial_{k}=0$$ be the oriented boundary maps and let $W_{k}(z)$ be the metric weight on $k$-chains. With the weighted adjoint $$\partial^{*}_{k}=W_{k}^{-1}\partial^{\dagger}_{k}W_{k-1},$$ the degree-$k$ Hodge Laplacian is $$L_{k}=\partial_{k+1}\partial^{*}_{k+1}+\partial^{*}_{k}\partial_{k}.$$ Three metric regimes are distinguished, and the vocabulary is fixed here once. If $W_{k}=W_{k}^{\dagger}\succ 0$ the pairing is an ordinary Hilbert inner product and $L_{k}$ is self-adjoint in it. If $W_{k}=W_{k}^{\dagger}$ is indefinite the pairing is a Krein inner product; the word *Krein* is reserved for exactly this Hermitian-indefinite regime. If $W_{k}$ is genuinely complex the pairing is neither, and the operator class is then fixed by certificates rather than by vocabulary. Two are used: the sesquilinear identity $L^{\dagger}W=WL$, which is what licenses $W$-unitary closed evolution (Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>), and the bilinear identity $L^{\mathsf{T}}W=WL$, which expresses complex symmetry and which the untwisted diagonal-weight construction above supplies automatically. Matched left and right frames provide a biorthogonal normalization but do not by themselves prove either identity, so neither is assumed: each is checked where used. The $\mathbb{C}^{*}$ twist generically breaks both; the compact $U(1)$ part alone, on positive weights, preserves the sesquilinear identity and reproduces the Hermitian magnetic operator of graph theory \[34, 35\]. Whenever $L_{k}$ carries no self-adjointness certificate, left and right spectral frames and their biorthogonal condition numbers are reported rather than silently treating $L_{k}$ as Hermitian.

Both declared edge fields evolve. The squared lengths relax toward joint stationary points of the existing Regge and Hodge functionals. The connection phase relaxes against the degree-zero connection operator, and it must be that operator rather than $L_{k}$: the geometric Hodge operator is blind to $\varphi$ at every degree, and $\dim\ker L_{0}=b_{0}$ at any weights, so $L_{0}$ cannot register a connection at all, whereas the connection operator’s zero mode is lifted by a nonzero flux. The resulting stationarity condition on $\varphi$ is gauge-invariant by construction rather than by correction: a gauge transformation acts on that operator by the similarity $\operatorname{diag}(g)^{-1}(\cdot)\operatorname{diag}(g)$, so its spectrum is invariant for every $g:K_{0}\to\mathbb{C}^{*}$; a functional of that spectrum is constant along gauge orbits, and its gradient is therefore orthogonal to every gauge direction, with no projection off the orbit required.

In emergence mode, particle-specific observables below are read after optimization and are not inserted as target terms. Controlled synthesis mode may pin a carrier to test realizability, but that is a separate experiment. Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a> refines the emergence protocol into two labeled modes — strict no-backreaction, and certificates-blind mean-field backreaction — and records that both remain inside the quasi-free class.

This operator stack sits on established foundations: Regge calculus encodes piecewise-flat gravity in simplicial deficit angles \[2\], discrete exterior calculus supplies metric-dependent chain/cochain operators \[3\], and combinatorial Hodge spectra on simplicial complexes have a developed spectral theory \[4\]. Tessera’s proposal is not a replacement for those constructions; it is a constrained use of them as the sole source of the later particle readouts.

# A component is an exact static response vertex

Partition the $k$-cells of a connected component into interface cells $B$ and interior cells $I$, and block its Hodge operator as $$L=\begin{pmatrix} L_{BB} & L_{BI}\\ L_{IB} & L_{II}\end{pmatrix}.$$ In the positive self-adjoint regime, after projecting out incompatible interior zero modes, minimization over the interior has the exact solution $$x_{I}^{*}=-L_{II}^{+}L_{IB}x_{B},$$ and the exact effective boundary operator $$\boxed{\,L_{\mathrm{eff}}=L_{BB}-L_{BI}L_{II}^{+}L_{IB}\,}.$$ Here ${}^{+}$ denotes the Moore–Penrose inverse on the supported interior subspace. For every compatible boundary value, $$\min_{x_{I}}
  \begin{pmatrix}x_{B}\\ x_{I}\end{pmatrix}^{\dagger}
  L
  \begin{pmatrix}x_{B}\\ x_{I}\end{pmatrix}
  = x_{B}^{\dagger}L_{\mathrm{eff}}x_{B}.$$

This is the precise static, or zero-frequency, sense in which a connected component can be replaced by a coarse response vertex. In a Hermitian indefinite regime the same equation is a stationarity condition, not a minimum. For a non-normal block it is simply block elimination; solvability requires $$L_{IB}x_{B}\perp\ker L_{II}^{\dagger}.$$

The plain Schur complement does *not* preserve the nonzero spectrum. For a spectral parameter $\lambda$ such that $L_{II}-\lambda I$ is invertible, define the exact Feshbach–Schur response $$\boxed{\,F_{B}(\lambda)=L_{BB}-\lambda I
    - L_{BI}(L_{II}-\lambda I)^{-1}L_{IB}\,}.$$ Then, for $\lambda$ outside $\operatorname{spec}L_{II}$, the exact determinant factorization $$\det(L-\lambda I)=\det(L_{II}-\lambda I)\,\det F_{B}(\lambda)$$ holds, so $$\lambda\in\operatorname{spec}L \iff 0\in\operatorname{spec}F_{B}(\lambda).$$ The order of the zero of $\det F_{B}(\cdot)$ at $\lambda$ equals the algebraic multiplicity of $\lambda$ in $L$, while $\dim\ker F_{B}(\lambda)$ equals its geometric multiplicity; the two agree in the self-adjoint or otherwise semisimple setting but not in general. At an interior resonance the inverse is replaced only after checking the compatibility condition $L_{IB}x_{B}\perp\ker(L_{II}-\lambda I)^{\dagger}$ and retaining the resonant interior modes explicitly. Thus harmonic response uses $F_{B}(0)$, while a localized band centered at $\lambda_{C}$ uses $F_{B}(\lambda)$ over a stated frequency window. A linear reduced eigenproblem may instead retain interface constraint modes plus selected fixed-interface modes using Craig–Bampton component-mode synthesis or AMLS; that route is certified approximation whose error is controlled by residuals and separation from discarded modes, not an exact spectral identity \[5, 6, 7\].

The effective blocks between coarse components become operator-valued links. A harmonic or retained interior mode is not discarded; it becomes an explicit stalk/fiber coordinate attached to the response vertex.

For graph Laplacians this is the classical Kron reduction by Schur complement \[8\]. Spectral graph reduction provides related approximation guarantees when additional coarsening or truncation is performed \[9\]. The extension proposed here is to apply static response reduction degree by degree to weighted Hodge blocks, and shifted Feshbach or certified component-mode reduction to nonzero bands, while retaining localized zero, resonant, and selected interior modes as explicit fiber coordinates.

# Recursive spectral fibers

Let $P_{\ell}=\{C_{v}^{\ell}\}$ be an intrinsic partition at scale $\ell$ into persistent connected components. At $\ell=0$ the object is the microscopic simplicial complex $K_{0}$. After the first elimination the honest coarse object is generally not another simplicial complex: it is an operator-valued response network $\mathcal{R}_{\ell+1}$ whose vertices carry vector spaces and whose links carry linear response blocks. A cellular sheaf on the quotient graph is a natural realization when the blocks admit compatible restriction-map factorization \[10\]; otherwise Tessera retains the more general response network and does not invent incidence maps that the reduction did not determine.

Within component $C$, choose an isolated localized spectral band and, in the positive self-adjoint regime, a weighted orthonormal frame $$\Phi_{C}=(\phi_{1},\dots,\phi_{r}),
  \qquad \Phi_{C}^{\dagger}W_{C}\Phi_{C}=I_{r}.$$ The derived fiber is $$E_{C}=\operatorname{Ran}\Phi_{C}.$$

In a Hermitian indefinite regime record the inertia of $\Phi_{C}^{\dagger}W_{C}\Phi_{C}$ and normalize it to a signature matrix $J_{C}=\operatorname{diag}(I_{p},-I_{q})$. Negative Krein signature is a certificate, not an automatic identification with an antiparticle. Pair-creation experiments in this programme do, however, exhibit an opposite-signature selection rule with conserved real part under conjugate-pair formation; that measured behavior is numerical evidence for the particle/antiparticle reading, while the identification itself remains a proposed interpretation. In a non-normal regime use matched right and left frames $\Phi_{C},\Psi_{C}$ with $\Psi_{C}^{\dagger}W_{C}\Phi_{C}=I$ and report both residuals and the frame condition number.

It need not be a harmonic space and therefore need not be supported by a hole. What it does require is a spectral gap, localization, and persistence. A candidate component is accepted only if all of the following remain stable across a stated range of scales:

- a persistent connected cluster support, however proposed;

- a localized spectral projector with stable rank;

- a nonzero band gap separating it from discarded modes;

- overlap with its predecessor and successor components;

- lifetime across multiple cobordism frames; and

- small external transport leakage.

Community objectives supply deterministic cluster candidates \[11\], while network renormalization supplies tests for genuine self-similarity rather than visual resemblance \[12\]. The partition is therefore a measured part of the analysis: a recursively drawn pattern is not evidence of a fractal unless its scaling observables survive a refinement window. The community-detection stage uses Newman–Girvan modularity on the combinatorial one-skeleton; it is a heuristic proposal generator that does not see signed or complex Hodge weights and is subject to the modularity resolution limit \[13\]. Modularity may therefore propose candidate supports, but it may not veto an otherwise certified fiber: acceptance is conditioned only on the independent, weight-aware gap, localization, leakage, persistence, and refinement certificates above, together with the anchoring certificate of Section <a href="#sec:quarks" data-reference-type="ref" data-reference="sec:quarks">10</a> whenever a color interpretation is claimed.

This gives a type-stable hierarchy of response objects $$\cdots\longrightarrow\mathcal{R}_{2}\longrightarrow\mathcal{R}_{1}\longrightarrow K_{0}$$ in which a response vertex at one level resolves into a connected microscopic component plus retained stalk coordinates at the next finer level. “Self-similar” refers to closure of the response-network data type, not to a claim that every reduced operator is a simplicial Hodge Laplacian. A fractal-like pattern is permitted but not required: measured scaling of module count, volume, boundary size, and spectral gap decides whether the hierarchy is statistically self-similar.

<figure id="fig:recursion">

<figcaption>One recursive step. Persistent connected modules become stalk-bearing vertices of an operator-valued response network. Static response is preserved by the supported Schur complement; nonzero bands use shifted Feshbach or certified component-mode reduction. Selected internal modes remain attached as fibers, and a persistent supermodule can be reduced again at the next scale.</figcaption>
</figure>

# Interactions and the expanding Hilbert space

Two operations must not be conflated. For the Cartesian product of chain complexes $A$ and $B$, the graded tensor differential is the exact rule $$d_{A\mathbin{\widehat{\otimes}}B}(a\otimes b)=d_{A}a\otimes b+(-1)^{\deg a}a\otimes d_{B}b.$$ For a noninteracting product with product metric, $$L_{A\mathbin{\widehat{\otimes}}B}=L_{A}\otimes I+I\otimes L_{B},$$ so one-particle eigenvalues add and eigenvectors tensor. This identity is about a product complex, not about gluing two cobordisms.

Actual simplicial gluing is a pushout along a shared boundary. At the one-particle level it produces a chain space assembled from direct sums modulo boundary identifications (equivalently described by the relevant Mayer–Vietoris sequence) and a block operator $$L_{A\cup B}=\begin{pmatrix} L_{A} & C_{AB}\\ C_{BA} & L_{B}\end{pmatrix}$$ in a basis adapted to the two interiors. The coupling blocks are induced by the connecting simplices and shared-boundary constraints; they are not a Kronecker interaction term.

The expanding Hilbert space follows after applying the fermionic Fock functor to the one-particle space $\mathfrak{h}$. The exact identities are $$\mathcal{F}_{-}(\mathfrak{h}_{A}\oplus\mathfrak{h}_{B})\cong\mathcal{F}_{-}(\mathfrak{h}_{A})\mathbin{\widehat{\otimes}}\mathcal{F}_{-}(\mathfrak{h}_{B}),$$ and $$d\Gamma(L_{A}\oplus L_{B})=d\Gamma(L_{A})\mathbin{\widehat{\otimes}}I+I\mathbin{\widehat{\otimes}}d\Gamma(L_{B}).$$ For the coupling block, $$d\Gamma(C_{AB}+C_{BA})
  =\sum_{ij}(C_{AB})_{ij}\,a^{\dagger}_{A,i}a_{B,j}+\text{h.c.},$$ so geometric connections become hopping terms without adding a new field. If the one-particle eigenvalues are $\lambda_{1},\dots,\lambda_{M}$, then the free many-body spectrum is the set of occupation subset sums $\sum_{i}n_{i}\lambda_{i}$, $n_{i}\in\{0,1\}$, rather than the one-particle pairwise spectrum being relabeled as a Fock spectrum \[14\].

At the selected-fiber level, an interaction grows the carried space as $$\mathcal{H}_{AB}=E_{A}\mathbin{\widehat{\otimes}}E_{B},$$ and a later interaction appends another factor. This is a statement about state-space composition after second quantization, not the topology of the glued chain complex. When carried subspaces of adjacent components overlap on interface cells, the composite is built on the abstract labeled sum with an explicit embedding Gram matrix; Section <a href="#sec:master" data-reference-type="ref" data-reference="sec:master">15</a> states the exact rule. If $J_{C}$ embeds an abstract state into the geometric carrier, exact amplitude preservation requires $$J_{C}^{\dagger}W_{C}J_{C}=I.$$ Tensor products preserve isometry exactly. If $G=J_{C}^{\dagger}W_{C}J_{C}$ has Gram defect $\varepsilon=\lVert G-I\rVert$, then $$\lvert a^{\dagger}Gb-a^{\dagger}b\rvert
  \le\varepsilon\,\lVert a\rVert\,\lVert b\rVert,$$ and two tensor factors obey $$\varepsilon_{AB}\le\varepsilon_{A}+\varepsilon_{B}
    +\varepsilon_{A}\varepsilon_{B}.$$ Thus the amplitude claim has an explicit, composable error budget.

Cobordism composition as a map between boundary state spaces is the organizing idea of topological field theory \[15\]; the general-boundary program makes the region/boundary assignment explicit for quantum theory \[16\], and categorical quantum mechanics formalizes tensor composition and diagrammatic process semantics \[17\]. Tessera keeps only the parts that can be realized by its finite simplicial carrier and tests the resulting map numerically rather than assuming topological invariance.

# Quasi-free dynamics and the covariance layer

Every many-body generator exhibited in this paper is quadratic. Free propagation is $d\Gamma(L)$, gluing contributes $d\Gamma$ of a coupling block, and every derived transport is the second quantization of a one-particle map. The exact consequence is closure of the quasi-free class: if the many-body generator is always of the form $$H(t)=d\Gamma\bigl(h(t)\bigr)=\sum_{ij}h_{ij}(t)\,a^{\dagger}_{i}a_{j},$$ then Gaussian/quasi-free states remain Gaussian.

The closure survives self-consistency. Let the one-particle operator depend on the covariance and on the classical geometry, $$h=h\bigl(\Gamma(t),g(t)\bigr),$$ with the geometry in turn relaxed against the state’s energy density. That is nonlinear mean-field dynamics of generalized Hartree–Fock type \[18\]; it can localize and it can produce self-bound solutions, but it does not leave the Gaussian manifold. Classical or mean-field geometry backreaction alone therefore does not generate genuinely non-Gaussian correlations.

The emergence protocol accordingly splits into two labeled modes, both Gaussian-closed: *strict emergence*, in which the state does not act back on the geometry at all, and *certificates-blind mean-field backreaction*, in which the carried state’s energy density enters the joint stationarity objective while every particle certificate remains firewalled from it. The certificate firewall of Section <a href="#sec:proton" data-reference-type="ref" data-reference="sec:proton">14</a> applies to both modes.

Genuinely non-Gaussian correlations would require at least one of the following, none of which is currently part of the model:

1.  a genuine quartic effective interaction, $$H_{\mathrm{int}}
          =\sum_{ijkl}V_{ijkl}\,
            a^{\dagger}_{i}a^{\dagger}_{j}a_{k}a_{l};$$

2.  quantized geometry that becomes entangled with the fermions;

3.  integrating out dynamical geometry beyond the mean-field approximation, producing a retarded or quartic effective interaction;

4.  a cobordism map that is not the second quantization of a one-particle map; or

5.  measurement or postselection capable of taking Gaussian states outside the Gaussian class.

Adopting one of these is an explicit scope decision with its own certificates, not a background assumption. Until then, the statement that non-Gaussian sectors are representable (Section <a href="#sec:state" data-reference-type="ref" data-reference="sec:state">3</a>) must not be read as a statement that they are produced.

The quasi-free formulation is extremely attractive on its own terms. In the number-conserving case the entire state is the covariance matrix $$\Gamma_{ij}=\langle a^{\dagger}_{j}a_{i}\rangle,$$ with $\Gamma^{2}=\Gamma$ exactly for a pure Slater state; a pairing sector would extend $\Gamma$ to the full Nambu covariance without changing the closure statement.

The evolution law for $\Gamma$ is a conditional theorem, and its hypotheses are the certificates of Section <a href="#sec:state" data-reference-type="ref" data-reference="sec:state">3</a>. When the one-particle generator satisfies the sesquilinear certificate $h^{\dagger}W=Wh$ and the metric is static, the closed-system commutator law $$i\dot{\Gamma}=[h,\Gamma]$$ holds: the flow is $W$-unitary and preserves trace, positivity, and the CAR structure of the covariance on a certificate-positive band. When the metric itself evolves with the geometry, instantaneous $W$-self-adjointness is not sufficient; metric compatibility requires $$h^{\dagger}W-Wh=i\dot{W},$$ equivalently the commutator law written in a moving $W$-orthonormal frame with its connection term. A generator that fails these certificates defines only the two-sided flow $\Gamma\mapsto e^{-iht}\,\Gamma\,e^{ih^{\dagger}t}$ with explicit normalization: effective, postselected spectral dynamics, and stated as such wherever it is used. No completely positive open-system evolution is claimed anywhere in this paper — that would require an explicit master equation in Lindblad form \[36\], which this ontology does not supply; the non-Hermitian sectors here are of the effective, pseudo-Hermitian kind \[37\].

The band covariance is conditional in the same way. An accepted band supplies an oblique projector satisfying $$P^{2}=P,\qquad P^{\dagger}W=WP.$$ On a certificate-positive band, factor the restricted metric as $W=S^{\dagger}S$; then $SPS^{-1}$ is an ordinary orthogonal projector, hence a bona fide fermionic covariance with $\Gamma^{\dagger}=\Gamma$ and $0\preceq\Gamma\preceq I$, satisfying the one-body representability constraints \[38\]. The similarity $S$ is fixed by the metric, not chosen: an arbitrary non-unitary similarity need not preserve the CAR representation and is not permitted. A band whose restricted metric fails the positivity certificate supplies no such covariance, and every readout that needs one refuses on that band and names the reason.

On the certified sector, Wick’s theorem computes every polynomial observable exactly: occupations and parities, the Pauli/Gram determinants, the color wedge $\lvert S_{ABC}\rvert^{2}$, and both $\langle J^{2}\rangle$ and its variance. There is no reason to construct an exponential Fock vector except for oracle tests or for explicitly non-Gaussian boundary data.

The programme order follows. First test the strongest possible covariance-only theory. Treat failure of the sharp proton certificate of Section <a href="#sec:proton" data-reference-type="ref" data-reference="sec:proton">14</a> as a meaningful structural result rather than a numerical nuisance. Introduce a non-Gaussian interaction only if the geometry supplies one naturally, through one of the mechanisms above. The question that decides the next stage of the programme is stated exactly:

<div class="center">

</div>

Either answer is informative. A covariance-only proton would make the entire particle layer polynomially computable and exactly certifiable; a demonstrated obstruction would be the first internal evidence that the geometry must supply a true interaction term.

# A triangle carries the exact color algebra

Consider the three edge-mode factors around an oriented triangle and interpret $\lvert 1\rangle$ as an occupied edge mode. Choosing an oriented ordering $(e_{1},e_{2},e_{3})$ identifies their graded tensor product with the exterior algebra $$(\mathbb{C}^{2})^{\mathbin{\widehat{\otimes}}3}\cong\Lambda^{\bullet}\mathbb{C}^{3}
  =\mathbf{1}\oplus\mathbf{3}\oplus\overline{\mathbf{3}}\oplus\mathbf{1}.$$ The orientation of one triangle fixes the ordering up to a cyclic, hence even, permutation, so the local wedge sign is unambiguous. Globally the exterior algebra $\Lambda^{\bullet}\mathfrak{h}_{K}$ and the CAR are intrinsic; only a presentation in ordered tensor factors needs a deterministic mode order and the corresponding permutation parity. A Kasteleyn orientation is useful for two-dimensional surface-dimer Pfaffians but is not required to define this abstract Fock space \[19\]. A genuine continuum spinor interpretation is a separate question addressed by the rotation certificate below.

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

The triplet description of quarks and the three-quark construction of baryons originate with the quark model \[20\]; the additional color triplet was introduced to resolve the statistics and state-counting problem \[21, 22\]. The claim here is narrower and new: Tessera’s three oriented edge modes would provide a geometric carrier of the same representation content, not a derivation of QCD from the combinatorics alone.

<figure id="fig:triangle">

<figcaption>Exact representation content of three oriented edge-mode factors. The exterior sectors and their parity are algebraic identities. Interpreting the rank-three odd sector as quark color and the top wedge as a baryon color singlet is the physical hypothesis to be tested.</figcaption>
</figure>

## Geometric normalization

For the stored complex squared lengths $z_{i}=\rho_{i}e^{i\theta_{i}}$ on the three oriented edges, define $$c_{i}=\frac{z_{i}}
    {\sqrt{\lvert z_{1}\rvert^{2}+\lvert z_{2}\rvert^{2}
      +\lvert z_{3}\rvert^{2}}},
  \qquad
  \lvert c\rangle=\sum_{i=1}^{3}c_{i}\lvert i\rangle.$$ Then $\langle c|c\rangle=1$. Constraining the perimeter to one is a valid geometric scale gauge, but it is an $L^{1}$ condition and does not replace the $L^{2}$ Hilbert normalization. Normalized pure color states form $\mathbb{CP}^{2}$; $SU(3)$ is the transformation group, not the surface of the triangle itself.

## The existing omega phase pattern

Let $\omega=e^{2\pi i/3}$. The exact Fourier frame $$F_{3}=\frac{1}{\sqrt{3}}
  \begin{pmatrix}
    1 & 1 & 1\\
    1 & \omega & \omega^{2}\\
    1 & \omega^{2} & \omega
  \end{pmatrix}$$ is unitary. The existing pattern $(1,\omega,\omega^{2})/\sqrt{3}$ is therefore one color basis vector, not by itself the whole color fiber. Its cyclic orbit supplies an exact orthonormal triad.

# Color transport and Wilson loops from spectral frames

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

This construction is the discrete spectral-frame analogue of Berry transport \[23\] and its non-Abelian Wilczek–Zee generalization \[24\]. Overlap matrices are also a standard route to gauge-covariant lattice observables \[25\], as are connection Laplacians and vector diffusion maps \[33\]. Wilson loops themselves are foundational lattice gauge observables \[26\]. What is specific here is that the rank-three link matrix is not independently assigned: unlike the declared microscopic $\mathbb{C}^{*}$ connection of Section <a href="#sec:state" data-reference-type="ref" data-reference="sec:state">3</a>, it is reconstructed from neighboring Hodge frames and accompanied by a leakage certificate.

# Quarks as modular clusters

A *quark candidate* is proposed to be a component $Q$ satisfying all of the following derived conditions.

The abstract rank-three band must first be anchored to oriented two-simplices. For an oriented triangle $\tau\subset Q$, let $R_{\tau}:C_{1}(Q)\to\mathbb{C}^{3}$ restrict a one-chain to the three ordered boundary edges, including their incidence signs, and let $\lvert W_{\tau}\rvert$ be the modulus of the edge-weight block on those three edges. Define the weighted restriction $$A_{\tau}=\lvert W_{\tau}\rvert^{1/2}R_{\tau}\Phi_{Q}.$$ For a rank-three frame $\Phi_{Q}$, define the gauge-invariant anchor score $$a_{Q}^{2}=\sum_{\tau\subset Q}w_{\tau}\,
    \bigl\lvert\det A_{\tau}\bigr\rvert^{2},
  \qquad w_{\tau}\ge 0,\qquad \sum_{\tau}w_{\tau}=1,$$ with the convex weighting $\{w_{\tau}\}$ declared before the data are examined. In the positive regime $R_{\tau}^{\dagger}\lvert W_{\tau}\rvert
R_{\tau}\preceq W$, so each $\lvert\det A_{\tau}\rvert^{2}=\det(A_{\tau}^{\dagger}A_{\tau})\le 1$ and the score is calibrated: $a_{Q}^{2}\in[0,1]$, with value one exactly at full concentration on the weighted edge span of the anchoring faces. In signed sectors the restriction still uses $\lvert W_{\tau}\rvert^{1/2}$ and the Krein signature of the restricted block is reported separately.

The phases of the nonzero determinants give a local determinant-line trivialization; their coherence on overlapping triangles is recorded separately. The reported anchor datum is the profile, not only the score: the maximal term $\max_{\tau}\lvert\det A_{\tau}\rvert^{2}$, the participation ratio of the distribution $\{\lvert\det A_{\tau}\rvert^{2}\}$, and the determinant-phase dispersion accompany $a_{Q}^{2}$. Concentration on one triangle is a sufficient special case, not a requirement: an extended fiber may be anchored by an atlas of oriented faces. Because every 2-simplex has exactly three boundary edges, this anchor is independent of the ambient spectral dimension.

1.  $Q$ is a persistent cluster certified as in Section <a href="#sec:fibers" data-reference-type="ref" data-reference="sec:fibers">5</a>, however its support was proposed.

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

A physical $2\pi$ rotation uses the same reference normalization, and the rotation path is not left abstract: it is a closed loop constructed here on the total-space frame, normalized against its own co-moving, non-rotating reference. It acts on the total-space $J^{2}$ operator, whose construction and oracle values are established independently of the rotation cycle; the sharp total-space spin readout itself remains an open experimental item. If an emergent manifold-like regime supplies tangent frames, a continuum spinor claim additionally requires a lift of the frame holonomy from $SO(d)$ to $\mathrm{Spin}(d)$; obstruction by the second Stiefel–Whitney class is then a falsification certificate \[27\]. This requirement concerns the physical spin lift, not the existence of the abstract CAR/Fock algebra. The spin-statistics comparison is cleanest as $$\widehat{\chi}_{F}(\text{exchange})\,
  \widehat{\chi}_{F}(2\pi\ \text{rotation})^{-1}=+1,$$ with each factor separately near $-1$.

Configuration-space topology already explains how exchange classes can carry quantum phases \[28\] and, in two dimensions, more general statistics \[29\]. The Tessera proposal uses this precedent only as a diagnostic template. The minus sign from the graded exterior algebra is exact; the claim that an actual geometric exchange cobordism realizes the corresponding determinant holonomy remains an experiment.

# Fock space as an inductive limit of interactions

For $M$ oriented fermionic edge modes, $$\widehat{\bigotimes}_{m=1}^{M}\mathbb{C}^{2}
  \cong\Lambda^{\bullet}\mathbb{C}^{M}
  =\bigoplus_{n=0}^{M}\Lambda^{n}\mathbb{C}^{M},$$ and the dimension identity is exact: $$2^{M}=\sum_{n=0}^{M}\binom{M}{n}.$$

The exterior algebra is canonical as a functor of the one-particle space. Writing it as a literal ordered tensor product, or presenting the creation operators in Jordan–Wigner form, requires a chosen mode order. A deterministic order is fixed by oriented component lineage, with the parity of every reordering applied; all reported observables must be invariant under relabeling plus that induced unitary.

Adding a new noninteracting mode uses the vacuum embedding $$\iota_{M}:\mathcal{H}_{M}\hookrightarrow\mathcal{H}_{M+1},
  \qquad
  \iota_{M}(\psi)=\psi\mathbin{\widehat{\otimes}}\lvert 0\rangle.$$ The infinite Fock space is the direct limit $$\mathcal{F}=\varinjlim(\mathcal{H}_{M},\iota_{M}).$$ This makes the infinite expansion precise while every stage remains finite: at any finite stage only finitely many modes have interacted. Consistency requires $$\lVert\iota_{M}U_{M}-U_{M+1}\iota_{M}\rVert\longrightarrow 0$$ over a refinement sequence.

A bosonic gauge sector, if realized, need not add a new local oscillator. The exact statement is representation-theoretic: the traceless even bilinears $$a^{\dagger}_{i}a_{j}-\frac{1}{3}\delta_{ij}N$$ transform in the color octet of $\mathbf{3}\otimes\overline{\mathbf{3}}=\mathbf{1}\oplus\mathbf{8}$ and have even fermion parity. That identifies the octet quantum numbers among collective fermion-pair excitations; it does not by itself establish propagating bosonic gauge excitations, which would require separate dynamical and continuum evidence. Within this model, arbitrarily many such collective excitations are represented by adding more microscopic modes at finer resolution, and each finite edge-mode factor remains two-dimensional.

This scale-by-scale state growth is adjacent to entanglement renormalization, where local Hilbert data are reorganized across layers before truncation \[30\]. The distinction is material: Tessera uses static/shifted response reduction plus an inductive vacuum embedding, and it must certify compatibility between successive finite spaces rather than assume a fixed bond dimension.

## Occupation exterior algebra is not automatically Kähler–Dirac

The exterior algebra above is over the one-particle *mode space*; its degree is occupation number. A Kähler–Dirac field instead lives on the inhomogeneous differential-form/cochain space $\bigoplus_{k}C^{k}(K)$ and is acted on by $d-d^{*}$ (or a related Dirac square root). These constructions share exterior-algebra notation but are not the same operator or grading. Consequently the present model does not inherit lattice taste multiplicity merely from using $\Lambda^{\bullet}\mathfrak{h}_{K}$.

If a later Tessera model promotes its one-particle field to all cochain degrees and uses the Kähler–Dirac operator, the known flat four-dimensional decomposition into four Dirac spinors becomes an expected spectrum diagnostic, not an unexplained bug \[31, 32\]. Any observed near-fourfold cluster in the present model is reported as an empirical degeneracy until that stronger operator identification is made.

# Mass, charge, and form factor from world-tube crossings

The certificates below need a mass, a charge, a radius, and a form factor. This section specifies them. They are read from the same object — a band world tube crossing a surface — and differ only in how the crossings are summed.

## The surfaces are supplied by the cobordism

A cobordism has $\partial W=M_{0}\sqcup M_{1}$, and it supplies its own reference surface. Let $\tau(x)$ be the Lorentzian distance from the incoming boundary $M_{0}$, complex like the geometry that defines it. The construction assumes the cobordism is time-oriented and causally regular enough for $\operatorname{Re}\tau$ to be a certified temporal function — strictly increasing along every future-directed causal path, the discrete counterpart of the temporal functions guaranteed on globally hyperbolic spacetimes \[39\] — and takes as surfaces the level sets $$\Sigma_{t}=\{x : \operatorname{Re}\tau(x)=t\}.$$ The slicing is geometrically selected rather than chosen: $\tau$ is fixed by the cobordism and its incoming boundary, not by a coordinate. It is not assumption-free, and the assumptions are certificates: the levels must be totally ordered, every counted crossing must be transversal, and a cut locus, a nonregular level, or a null normal causes the readout to refuse with the failure named. $M_{0}$ supplies the reference surface and the reference values — every readout below is the difference between its value on $\Sigma_{t}$ and the same sum evaluated at $M_{0}$ — and it is a boundary hypersurface carrying boundary data, not a quantum state.

## The crossing decomposition

A persistent band (Section <a href="#sec:fibers" data-reference-type="ref" data-reference="sec:fibers">5</a>) tracked across cobordism frames sweeps a world tube. Where a tube $c$ crosses $\Sigma_{t}$, its crossing set $C(c)$ is the set of edges whose endpoints the level separates, and the tube is decomposed against the surface. The perpendicular projection is the band-weighted increment of the temporal function across the crossing, $$\pi_{\perp}(c)=\sum_{e\in C(c)}\mu_{c}(e)\,\Delta\tau(e)\in\mathbb{C},
  \qquad
  \Delta\tau(e)=\tau(e^{+})-\tau(e^{-}),$$ where $e^{-}$ and $e^{+}$ are the past and future endpoints of $e$. The increment $\Delta\tau(e)$ is complex because the geometric data are; it is built from the squared lengths $z$ alone and never contains the connection. The weight $\mu_{c}(e)$ is the band density on $e$, formed bilinearly from the matched left and right frames of the band and normalized to $\sum_{e\in C(c)}\mu_{c}(e)=1$. It is invariant under local $\mathbb{C}^{*}$ gauge transformations, because the gauge factor acts on right frames by $g^{-1}$ and on left frames by $g$ and cancels in the bilinear product; and $\pi_{\perp}$ is unchanged under relabeling the level values, because only increments of $\tau$ enter. The tangential component $\pi_{\parallel}$ is the complementary within-surface part of the same decomposition.

A crossing is admissible only if the band passes the positivity certificate of Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a> — a band that fails it supplies no covariance and no particle reading — and only if the crossing is timelike and transversal, which is the requirement that $\operatorname{Re}\pi_{\perp}$ be nonvanishing with a single sign across the crossing set. A spacelike, null, or grazing crossing has no particle reading at all, and the readout refuses and names the reason rather than labelling it. On an admissible crossing the sign $$\operatorname{sgn}\pi_{\perp}
  :=\operatorname{sgn}\operatorname{Re}\pi_{\perp}\in\{\pm 1\}$$ is well defined, gauge-invariant, and canonical rather than conventional: a bare complex number has no preferred sign, but the time orientation and the temporal function — exactly the hypotheses already assumed above — select the real part of the increment as the causal component, so no arbitrary choice enters. A future-directed crossing carries $+1$ and a past-directed crossing carries $-1$.

## Mass and charge are one sum taken two ways

Over the admissible crossings of $\Sigma_{t}$, relative to $M_{0}$: $$m_{\times}(\Sigma_{t})=\kappa_{m}\sum_{c}\lvert\pi_{\perp}(c)\rvert,
  \qquad
  B(\Sigma_{t})=\frac{1}{3}\sum_{c}\operatorname{sgn}\pi_{\perp}(c),$$ the second sum over certified quark tubes. Mass is the *incoherent* sum: moduli, nothing cancels. Baryon number is the *coherent* sum: signs, opposites cancel. That single difference reproduces the observed behaviour of the two quantities. Three forward quark tubes give $B=1$; a quark tube and an antiquark tube give $B=0$, and the same pair carries twice the crossing mass of one constituent, because the moduli add while the signs cancel. The normalization of one third per quark tube makes the coherent sum consistent with the independent determinant-line proposal $B=\nu/3$ of Section <a href="#sec:transport" data-reference-type="ref" data-reference="sec:transport">9</a>, and quantization of $B$ is a consequence of counting signed thirds, not an assumption.

The constant $\kappa_{m}$ is one declared calibration with dimensions of mass per length, fixed once against a single physical input; every subsequent mass is a prediction, and only ratios are meaningful before calibration. $m_{\times}$ is the *crossing-mass functional*: additivity over crossings and strict positivity per admissible crossing are properties of this functional by construction, not asserted universal laws of physical mass. Massless content lies outside its stated domain — a null crossing is refused, not counted at zero — and binding energy is not additive over constituents in physical composites, so agreement of the calibrated $m_{\times}$ with the physical proton mass, whose decomposition is measured \[40\], is a nontrivial benchmark of the identification rather than an algebraic consequence.

The sign is the physical content of $\pi_{\perp}$. A past-directed crossing is an antiparticle — the kinematic reading of the orientation reversal already used in Section <a href="#sec:quarks" data-reference-type="ref" data-reference="sec:quarks">10</a>, where reversing a tube sends $B=+1/3$ to $B=-1/3$. The same sign therefore appears twice, once as the crossing orientation and once as the determinant-line winding of Section <a href="#sec:transport" data-reference-type="ref" data-reference="sec:transport">9</a>. They must agree on every certified tube. Their agreement is a cross-check, not a redundancy: a tube on which they disagree is a defect signal.

Whether a negative signature of a band’s certified restricted metric marks an antiparticle or an unphysical sector is deliberately left open. It is to be settled by measuring whether that signature tracks the crossing sign, not by decree.

## Orientation gives baryon number; electric charge needs flavor

The coherent sum counts tube orientations, so it is baryon number. It is not by itself electric charge: an up and a down quark are both forward tubes with $B=+1/3$ and differ in $Q$. Electric charge follows from the relation already stated in Section <a href="#sec:quarks" data-reference-type="ref" data-reference="sec:quarks">10</a>, $$Q=I_{3}+\frac{B}{2},$$ once the isospin doublet is certified, and is cross-checked against the Gauss-flux read on nested enclosing surfaces. Two independent routes to $Q$, required to agree.

## Momentum, radius, and the form factor

The tangential component $\pi_{\parallel}$ carries the spatial content: momentum, and with it the extent of the crossing set on $\Sigma_{t}$, which is the radius.

Localizing charge gives two distinct observables, and they are kept distinct. Each admissible crossing contributes its signed unit at its position on $\Sigma_{t}$, so the crossings define a charge density $\rho$ on the surface.

The first observable is unconditional. Let $P_{\lambda}$ be the spectral projector of the slice Laplacian of $\Sigma_{t}$ onto the eigenvalue $\lambda$. The *spectral charge-power profile* $$S(\lambda)=\frac{\langle\rho,P_{\lambda}\,\rho\rangle}
                  {\langle\rho,P_{0}\,\rho\rangle}$$ is basis- and phase-invariant: it is built from eigenspace projectors, so degeneracies are handled and no eigenvector phase enters. The slice Laplacian is the discrete $-\nabla^{2}$, so $\lambda$ plays the squared momentum transfer honestly rather than by analogy. $S$ is an incoherent power, the analogue of a structure factor. It is not identified with the electromagnetic form factor, because a squared overlap loses the sign and the relative phase of the coherent matrix element.

The second observable is conditional. The compact $U(1)$ part of the $\mathbb{C}^{*}$ connection supplies a conserved current — the Noether current of the phase symmetry carried by the twisted hopping operator — and the physical electric form factor $G_{E}(Q^{2})$ is a normalized matrix element of that current’s charge density between states of certified momentum transfer. It exists only where the slice possesses translation and rotation structure stable enough for momentum-transfer states to be defined, and the charge radius $$\langle r^{2}\rangle=-6\,\frac{dG_{E}}{dQ^{2}}\bigg|_{Q^{2}=0}$$ is read only in a certified three-spatial-dimensional refinement regime: a finite spectrum has no literal derivative at zero, so the slope is defined by a documented small-$Q^{2}$ refinement extrapolation with stability and normalization certificates, as in continuum extractions of the proton radius \[41\]. The $\lambda\leftrightarrow Q^{2}$ dictionary uses the same single calibration that fixes $\kappa_{m}$, so no second scale is introduced. If the momentum-structure or extrapolation certificates fail, the electromagnetic radius is reported as unavailable rather than inferred from the spectral power. For a neutral system the normalizing monopole vanishes, the normalized profile refuses, and the unnormalized slope remains reportable.

Every quantity here is a difference against $M_{0}$, never an absolute. A curved complex carries curvature and flux in its ground state; an absolute reading would fold that background into the excitation, and a background-dependent mass is exactly the kind of definition-sensitivity that makes a dimensionful number unquotable.

# The proton as the maximally informative baryon

The proton is chosen because, beyond generic baryon structure, it demands a nontrivial flavor pattern, electric charge, spin, and experimentally meaningful form factors.

Let three persistent quark components $A,B,C$ have color frames and normalized color columns $c_{A},c_{B},c_{C}$. The invariant color volume is $$S_{ABC}=\epsilon_{ijk}c_{A}^{i}c_{B}^{j}c_{C}^{k}
  =\det[c_{A}\ c_{B}\ c_{C}].$$ Under a common $g\in SU(3)$, $S\mapsto\det(g)S=S$. Its squared magnitude is the Gram determinant $$\lvert S_{ABC}\rvert^{2}=\det(C^{\dagger}C)\in[0,1].$$ The value one means the three color directions form an orthonormal frame. Their normalized wedge is then the unique $\Lambda^{3}\mathbb{C}^{3}$ singlet. The proposed proton certificate is the conjunction:

- three persistent odd rank-three quark clusters with accepted triangle-anchor certificates;

- one persistent bound supercluster containing them;

- normalized color wedge with $\lvert S_{ABC}\rvert^{2}\approx 1$ and vanishing net color flux;

- flavor spectrum with the $uud$ occupation pattern, in the sense of the still-hypothetical isospin-doublet construction of Section <a href="#sec:quarks" data-reference-type="ref" data-reference="sec:quarks">10</a>;

- oriented baryon flux $B=1$ in the relative-winding sense of Section <a href="#sec:transport" data-reference-type="ref" data-reference="sec:transport">9</a>;

- Gauss flux $Q=+1$;

- a sharp total-space spin readout: $\langle J^{2}\rangle=3/4$ with $\operatorname{Var}(J^{2})
        =\langle (J^{2})^{2}\rangle-\langle J^{2}\rangle^{2}\approx 0$, a reference-normalized $2\pi\mapsto -1$, and, where applicable, an accepted spin lift;

- a finite radius and stable crossing-mass and form-factor readouts, in the world-tube-crossing sense of Section <a href="#sec:readout" data-reference-type="ref" data-reference="sec:readout">13</a>, the electromagnetic radius subject to its momentum-structure certificate — every one of them a difference against $M_{0}$, never an absolute; and

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

The geometric subspaces $E_{v}\subset C(K)$ of adjacent components may overlap on shared interface cells, so their internal sum need not be direct. The recursion therefore never asserts $\bigoplus_{v}E_{v}\subset C(K)$. It forms the abstract labeled sum, carries the embedding $J_{\ell+1}$ and its Gram matrix $G_{\ell+1}$ exactly, and proceeds by exactly one of three declared options: carry $G$ in every subsequent formula; certify $\lVert G-I\rVert\le\varepsilon$ and propagate $\varepsilon$ through the composable amplitude budget of Section <a href="#sec:interactions" data-reference-type="ref" data-reference="sec:interactions">6</a>; or quotient $\ker G$ and restate the fiber ranks. A sheaf-stalk decomposition that assigns interface modes to link stalks is a valid realization of the same requirement \[10\], but it is not necessary.

At $\lambda=0$ the response step is the exact supported static Schur complement. For a nonzero band it is the exact energy-dependent pencil; a linear $\mathcal{R}_{\ell+1}$ is an AMLS/component-mode surrogate with a declared frequency window and residual. The transport rank is generic; only an anchored accepted rank-three fiber receives the color interpretation. This recursion supplies the response network, retained stalk, derived transport, and expanding state space without claiming that every coarse level is literally a new simplicial complex.

# Prior art and boundary of novelty

No single cited work establishes the full recursive spectral-fiber proposal. The construction is a synthesis of several mature ideas, and its novelty should be evaluated at the joins. Table <a href="#tab:priorart" data-reference-type="ref" data-reference="tab:priorart">3</a> states the boundary explicitly.

<div id="tab:priorart">

| Topic                                 | Established prior art                                                                                                                                                                        | Additional claim made here                                                                                                                                                                                                                                                                                                                                                        |
|:--------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Topic                                 | Established prior art                                                                                                                                                                        | Additional claim made here                                                                                                                                                                                                                                                                                                                                                        |
| continued on next page                |                                                                                                                                                                                              |                                                                                                                                                                                                                                                                                                                                                                                   |
| Simplicial geometry and Hodge spectra | Regge curvature, discrete exterior calculus, and combinatorial Laplace spectra \[2, 3, 4\].                                                                                                  | Use one jointly optimized Regge–Hodge complex as the only microscopic carrier for geometry and quantum readouts.                                                                                                                                                                                                                                                                  |
| Coarse response and recursive modules | Kron/Schur reduction, Feshbach maps, component-mode synthesis/AMLS, cellular sheaf Laplacians, modular communities, and self-similar network renormalization \[5, 6, 7, 8, 10, 11, 12, 13\]. | Treat a persistent component as a static response vertex; retain nonzero bands through shifted or certified component-mode reduction; recurse in operator-valued response networks, using a sheaf realization only when its factorization is certified.                                                                                                                           |
| Quasi-free many-body calculus         | Second quantization, generalized Hartree–Fock theory, and quasi-free/Gaussian state methods \[14, 18\].                                                                                      | Evaluate every *polynomial* particle certificate as a finite Wick sum on the covariance matrix, on sectors whose metric certificates are verified, while the non-polynomial readouts are read from geometry and holonomy; prove mean-field geometry backreaction is Gaussian-closed; treat sharp-certificate failure as a structural dichotomy rather than a numerical shortfall. |
| Geometric gauge transport             | Berry and Wilczek–Zee holonomy, overlap-based lattice links, magnetic and connection Laplacians/vector diffusion, and Wilson loops \[23, 24, 25, 26, 33, 34, 35\].                           | Derive $U(r)$ transport from component frames; at anchored rank three retain the determinant line, projective $SU(3)/\mathbb{Z}_{3}$ class, and any chosen center lift, assigning no independent link variable beyond the declared microscopic $\mathbb{C}^{*}$ connection.                                                                                                       |
| Color and fermion structure           | Quark/color triplets, exterior Fock/second quantization, topological exchange phases, and spin structures \[20, 21, 22, 14, 28, 29, 27, 19\].                                                | Realize $\mathbf{1}\oplus\mathbf{3}\oplus\overline{\mathbf{3}}\oplus\mathbf{1}$ on three oriented edge modes, anchor abstract rank-three fibers to oriented faces, and test exchange by a Berry-cancelled determinant-line interferometer plus structural permutation parity.                                                                                                     |
| Scale composition and boundaries      | TQFT cobordisms, general-boundary state assignments, categorical tensor composition, second quantization, and entanglement renormalization \[15, 16, 17, 14, 30\].                           | Keep simplicial gluing at the one-particle level, then build finite Fock stages functorially and require vacuum-embedding compatibility under refinement.                                                                                                                                                                                                                         |
| Kähler–Dirac boundary                 | Differential-form fermions and their taste structure \[31, 32\].                                                                                                                             | Do not infer Kähler–Dirac tastes from occupation exterior algebra; test for them only if the one-particle field is promoted to inhomogeneous cochains with a Kähler–Dirac operator.                                                                                                                                                                                               |
| Spectral spacetime                    | Diffusion spectral dimension on ensembles of simplicial geometries \[1\].                                                                                                                    | Test whether many interacting Tessera cobordisms yield a stable four-dimensional spectral window while simultaneously supporting the particle certificates.                                                                                                                                                                                                                       |

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

5.  **No fermion holonomy.** The Berry-cancelled exchange ratio or structural permutation sign does not give $-1$, or the verdict changes under relabeling.

6.  **No spinor rotation.** Exchange works but the reference-normalized $2\pi$ physical rotation does not give $-1$; in a manifold-like continuum claim, failure of a consistent spin lift is also decisive.

7.  **No inductive compatibility.** Adding vacuum modes changes already-computed amplitudes by a nonvanishing amount.

8.  **No quasi-free proton.** Every other certificate is met inside the covariance-only theory, but $\operatorname{Var}(J^{2})$ fails to converge to zero on every accepted candidate across refinement. This outcome is a branch point rather than a refutation of the geometry: it mandates adopting exactly one of the non-Gaussian mechanisms of Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>, as an explicit scope decision, before any proton claim is made.

9.  **No unforced baryon.** Targeted synthesis can build the certificates, but the stationary geometric ensemble never produces them without a proton-specific term.

10. **No continuum stability.** Dimensionless color, parity, charge, spin, and amplitude certificates drift rather than converge with refinement.

11. **Unexpected multiplicity.** A robust flavor/taste degeneracy is neither predicted by the stated one-particle operator nor stable enough to be promoted to an emergent flavor mechanism.

Holes may re-emerge and may correlate with some phases, but no claim in this paper depends on them doing so.

# Conclusion

The geometry is economical and precise: an edge carries a two-level mode, not an independently stored pure state; a quark candidate is a modular spectral component whose rank-three band is anchored to oriented faces by a calibrated profile; its transport is a certified $U(3)$ overlap with retained determinant-line and projective color data; and its fermionic sign is the exterior grading, checked dynamically only after cancelling ordinary Berry phase. Simplicial gluing constructs the one-particle operator and second quantization constructs the expanding Fock state. Three accepted components form a baryon through their normalized color wedge; the proton is the sharpest test because it also demands the correct determinant-line flux, charge, flavor, and a variance-certified spin response.

The claims close in the four tiers of Section <a href="#sec:epistemic" data-reference-type="ref" data-reference="sec:epistemic">1</a>. The exact identities are limited to their proper domains: static Schur response, energy-dependent Feshbach isospectrality, exterior/CAR algebra, second-quantized direct-sum composition, and gauge covariance of accepted transport. The conditional theorems carry their certificates with them: closure of the quasi-free class holds under every generator the model currently possesses, and the commutator evolution law and the covariance representation hold exactly on sectors whose metric certificates are verified — there, every *polynomial* certificate, including $\langle J^{2}\rangle$ and its variance, is a finite Wick sum, and on those sectors the covariance matrix is not an approximation tier but the exact many-body state representation. The remaining readouts — the determinant winding, the Gauss flux, the rotation character, the spin lift, the anchor, and the crossing readouts of Section <a href="#sec:readout" data-reference-type="ref" data-reference="sec:readout">13</a> — are proposed physical identifications and calibrated observables: exact as computations at each finite discretization stage, with their continuum and physical meaning resting on declared calibrations, refinement extrapolations, and stability certificates, and refusing where those certificates fail. Exactness at a finite stage is never conflated with physical establishment.

What remains genuinely open is whether Tessera’s unforced Regge–Hodge dynamics produces the required anchored clusters, low-leakage holonomies, relative determinant windings, and sharp spin response. A tempting Hellmann–Feynman/envelope argument does not by itself make the first variation of transport Gram defect vanish at a Regge–Hodge stationary point, because the defect is not the optimized functional; the programme therefore measures that correlation as a conjectural scaling law rather than citing stationarity as a theorem. The decisive question is the dichotomy of Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>: either an exact covariance-only proton exists, or a genuinely non-Gaussian, geometry-mediated interaction is required. Either outcome is a result. The first makes the particle layer exactly and polynomially certifiable; the second would be the first internal evidence that the geometry must supply a true interaction term, through one of the five mechanisms this paper names.

<div class="thebibliography">

41

\[1\] Jan Ambjørn, Jerzy Jurkiewicz, and Renate Loll. Spectral dimension of the universe. *Physical Review Letters*, 95:171301, 2005. doi:10.1103/PhysRevLett.95.171301. URL <https://arxiv.org/abs/hep-th/0505113>.

\[2\] Tullio Regge. General relativity without coordinates. *Il Nuovo Cimento*, 19:558–571, 1961. doi:10.1007/BF02733251.

\[3\] Mathieu Desbrun, Anil N. Hirani, Melvin Leok, and Jerrold E. Marsden. Discrete exterior calculus, 2005. URL <https://arxiv.org/abs/math/0508341>.

\[4\] Danijela Horak and Jürgen Jost. Spectra of combinatorial laplace operators on simplicial complexes. *Advances in Mathematics*, 244:303–336, 2013. doi:10.1016/j.aim.2013.05.005. URL <https://arxiv.org/abs/1105.2712>.

\[5\] Roy R. Craig Jr. and Mervyn C. C. Bampton. Coupling of substructures for dynamic analyses. *AIAA Journal*, 6(7):1313–1319, 1968. doi:10.2514/3.4741.

\[6\] Jeffrey K. Bennighof and Richard B. Lehoucq. An automated multilevel substructuring method for eigenspace computation in linear elastodynamics. *SIAM Journal on Scientific Computing*, 25(6):2084–2106, 2004.

\[7\] Volker Bach, Thomas Chen, Jürg Fröhlich, and Israel Michael Sigal. Smooth feshbach map and operator-theoretic renormalization group methods. *Journal of Functional Analysis*, 203(1):44–92, 2003. doi:10.1016/S0022-1236(03)00057-0.

\[8\] Florian Dörfler and Francesco Bullo. Kron reduction of graphs with applications to electrical networks. *IEEE Transactions on Circuits and Systems I: Regular Papers*, 60(1):150–163, 2013. doi:10.1109/TCSI.2012.2215780. URL <https://arxiv.org/abs/1102.2950>.

\[9\] Andreas Loukas. Graph reduction with spectral and cut guarantees. *Journal of Machine Learning Research*, 20(116):1–42, 2019. URL <https://jmlr.org/papers/v20/18-680.html>.

\[10\] Jakob Hansen and Robert Ghrist. Toward a spectral theory of cellular sheaves. *Journal of Applied and Computational Topology*, 3:315–358, 2019. doi:10.1007/s41468-019-00038-7. URL <https://arxiv.org/abs/1808.01513>.

\[11\] Jörg Reichardt and Stefan Bornholdt. Statistical mechanics of community detection. *Physical Review E*, 74:016110, 2006. doi:10.1103/PhysRevE.74.016110.

\[12\] Chaoming Song, Shlomo Havlin, and Hernán A. Makse. Self-similarity of complex networks. *Nature*, 433:392–395, 2005. doi:10.1038/nature03248.

\[13\] Santo Fortunato and Marc Barthélemy. Resolution limit in community detection. *Proceedings of the National Academy of Sciences*, 104(1):36–41, 2007. doi:10.1073/pnas.0605965104. URL <https://arxiv.org/abs/physics/0607100>.

\[14\] Felix A. Berezin. *The Method of Second Quantization*. Academic Press, New York, 1966.

\[15\] Michael F. Atiyah. Topological quantum field theories. *Publications Mathématiques de l’IHÉS*, 68:175–186, 1988. doi:10.1007/BF02698547.

\[16\] Robert Oeckl. A “general boundary” formulation for quantum mechanics and quantum gravity. *Physics Letters B*, 575:318–324, 2003. doi:10.1016/j.physletb.2003.08.043. URL <https://arxiv.org/abs/hep-th/0306025>.

\[17\] Samson Abramsky and Bob Coecke. A categorical semantics of quantum protocols. In *Proceedings of the 19th Annual IEEE Symposium on Logic in Computer Science*, pages 415–425, 2004. doi:10.1109/LICS.2004.1319636. URL <https://arxiv.org/abs/quant-ph/0402130>.

\[18\] Volker Bach, Elliott H. Lieb, and Jan Philip Solovej. Generalized Hartree–Fock theory and the Hubbard model. *Journal of Statistical Physics*, 76(1–2):3–89, 1994. doi:10.1007/BF02188656.

\[19\] David Cimasoni and Nicolai Reshetikhin. Dimers on surface graphs and spin structures. i. *Communications in Mathematical Physics*, 275:187–208, 2007. doi:10.1007/s00220-007-0302-7.

\[20\] Murray Gell-Mann. A schematic model of baryons and mesons. *Physics Letters*, 8(3):214–215, 1964. doi:10.1016/S0031-9163(64)92001-3.

\[21\] Moo-Young Han and Yoichiro Nambu. Three-triplet model with double SU(3) symmetry. *Physical Review*, 139:B1006–B1010, 1965. doi:10.1103/PhysRev.139.B1006.

\[22\] O. W. Greenberg. Spin and unitary-spin independence in a paraquark model of baryons and mesons. *Physical Review Letters*, 13:598–602, 1964. doi:10.1103/PhysRevLett.13.598.

\[23\] Michael V. Berry. Quantal phase factors accompanying adiabatic changes. *Proceedings of the Royal Society of London A*, 392(1802):45–57, 1984. doi:10.1098/rspa.1984.0023.

\[24\] Frank Wilczek and A. Zee. Appearance of gauge structure in simple dynamical systems. *Physical Review Letters*, 52:2111–2114, 1984. doi:10.1103/PhysRevLett.52.2111.

\[25\] Takahiro Fukui, Yasuhiro Hatsugai, and Hiroshi Suzuki. Chern numbers in discretized brillouin zone: Efficient method of computing (spin) hall conductances. *Journal of the Physical Society of Japan*, 74(6):1674–1677, 2005. doi:10.1143/JPSJ.74.1674. URL <https://arxiv.org/abs/cond-mat/0503172>.

\[26\] Kenneth G. Wilson. Confinement of quarks. *Physical Review D*, 10:2445–2459, 1974. doi:10.1103/PhysRevD.10.2445.

\[27\] H. Blaine Lawson Jr. and Marie-Louise Michelsohn. *Spin Geometry*. Princeton University Press, Princeton, 1989.

\[28\] Michael G. G. Laidlaw and Cécile Morette DeWitt. Feynman functional integrals for systems of indistinguishable particles. *Physical Review D*, 3:1375–1378, 1971. doi:10.1103/PhysRevD.3.1375.

\[29\] Jon M. Leinaas and Jan Myrheim. On the theory of identical particles. *Il Nuovo Cimento B*, 37:1–23, 1977. doi:10.1007/BF02727953.

\[30\] Guifré Vidal. Entanglement renormalization. *Physical Review Letters*, 99:220405, 2007. doi:10.1103/PhysRevLett.99.220405. URL <https://arxiv.org/abs/cond-mat/0512165>.

\[31\] Peter Becher and Hans Joos. The dirac–kähler equation and fermions on the lattice. *Zeitschrift für Physik C*, 15:343–365, 1982. doi:10.1007/BF01614426.

\[32\] Nouman Butt, Simon Catterall, Arnab Pradhan, and Goksu Can Toga. Anomalies and symmetric mass generation for kähler–dirac fermions. *Physical Review D*, 104:094504, 2021. doi:10.1103/PhysRevD.104.094504. URL <https://arxiv.org/abs/2101.01026>.

\[33\] Amit Singer and Hau-Tieng Wu. Vector diffusion maps and the connection laplacian. *Communications on Pure and Applied Mathematics*, 65(8):1067–1144, 2012. doi:10.1002/cpa.21395. URL <https://arxiv.org/abs/1102.0075>.

\[34\] Elliott H. Lieb and Michael Loss. Fluxes, laplacians, and kasteleyn’s theorem. *Duke Mathematical Journal*, 71(2):337–363, 1993. doi:10.1215/S0012-7094-93-07114-1.

\[35\] Michaël Fanuel, Carlos M. Alaíz, and Johan A. K. Suykens. Magnetic eigenmaps for community detection in directed networks. *Physical Review E*, 95:022302, 2017. doi:10.1103/PhysRevE.95.022302.

\[36\] Göran Lindblad. On the generators of quantum dynamical semigroups. *Communications in Mathematical Physics*, 48(2):119–130, 1976. doi:10.1007/BF01608499.

\[37\] Ali Mostafazadeh. Pseudo-hermitian representation of quantum mechanics. *International Journal of Geometric Methods in Modern Physics*, 7(7):1191–1306, 2010. doi:10.1142/S0219887810004816. URL <https://arxiv.org/abs/0810.5643>.

\[38\] A. John Coleman. Structure of fermion density matrices. *Reviews of Modern Physics*, 35(3):668–686, 1963. doi:10.1103/RevModPhys.35.668.

\[39\] Antonio N. Bernal and Miguel Sánchez. Smoothness of time functions and the metric splitting of globally hyperbolic spacetimes. *Communications in Mathematical Physics*, 257:43–50, 2005. doi:10.1007/s00220-005-1346-1. URL <https://arxiv.org/abs/gr-qc/0401112>.

\[40\] Yi-Bo Yang, Jian Liang, Yu-Jiang Bi, Ying Chen, Terrence Draper, Keh-Fei Liu, and Zhaofeng Liu. Proton mass decomposition from the QCD energy momentum tensor. *Physical Review Letters*, 121:212001, 2018. doi:10.1103/PhysRevLett.121.212001. URL <https://arxiv.org/abs/1808.08677>.

\[41\] Alexander V. Gramolin and Rebecca L. Russell. Transverse charge density and the radius of the proton. *Physical Review D*, 105:054004, 2022. doi:10.1103/PhysRevD.105.054004. URL <https://arxiv.org/abs/2102.13022>.

</div>
