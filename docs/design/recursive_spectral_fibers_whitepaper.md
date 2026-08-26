<div class="titlepage">

<div class="center">

Recursive Spectral Fibers on Simplicial Cobordisms

A geometric program for quarks, color, fermion statistics, Fock space, and baryons

Tessera – cobordism programme

**Abstract**

</div>

This paper proposes a single geometric formulation for the particle content already suggested by Tessera’s cobordism experiments. A persistent, spectrally certified connected simplicial component is treated as one effective vertex at the next resolution. The component’s selected localized Riesz spectral subspace is its fiber, and the couplings between components induce transport between those fibers. Repeating this operation produces a nested, potentially fractal hierarchy of complexes. No particle label or auxiliary lattice is introduced, and the microscopic fields are exactly two per edge: a complex squared length, which alone determines the metric weights, and a multiplicative complex connection $U_e\in\mathbb{C}^{*}$ — an independent Abelian link field that twists the covariant hopping operator. Each active edge indexes one two-level occupation mode, but its occupation belongs to the Fock state and is not a third stored microscopic field. The generally entangled state lives on the exterior Fock space of all active edge modes. Exchange signs, the derived color transport, and observables are obtained from that state and the Hodge/Regge operators already present in the construction.

The proposal has a substantial exact core. Static Schur reduction proves when a component may be replaced by a response vertex without changing its supported complex boundary response. Nonzero spectral bands instead use the energy-dependent Feshbach–Schur map, or a certified Craig–Bampton/AMLS linear surrogate. Simplicial gluing acts on the one-particle chain space; fermionic second quantization then turns direct sums into graded tensor products and coupling blocks into hopping terms. Every generator so obtained is quadratic, so the dynamics is exactly quasi-free; a matched left/right Slater pair is carried without loss by the idempotent biorthogonal covariance $\Gamma=\Phi\widetilde\Phi^{\mathsf T}$, Wick reduction evaluates every polynomial matrix element, and mean-field geometry backreaction provably stays Gaussian. Three oriented edge-mode factors form the exact complex exterior algebra $\Lambda^{\bullet}E=\mathbf{1}\oplus E\oplus(\det E\otimes E^{\vee})
\oplus\det E$. Its traceless bilinears close $\mathfrak{sl}(3,\mathbb C)$; an $SU(3)$ real form is claimed only if a compatible quantum $*$-structure is separately certified. A rank-$r$ $GL(r,\mathbb C)$ connection is derived from overlap of neighboring Riesz spectral frames. At rank three its determinant line and projective $PGL(3,\mathbb C)$ transport are retained rather than choosing a cube-root, logarithm, polar factor, or real projection. Closed holonomies are gauge-invariant observables rather than new degrees of freedom. Successive cobordism interactions generate the finite stages of an inductive-limit Fock space.

The physical identification remains a hypothesis to be tested. A quark is proposed to be a persistent, odd-parity, rank-three spectral fiber anchored projectively to oriented faces of a certified cluster; an antiquark is the dual fiber on the oppositely oriented lineage. A proton is three such clusters bound into one persistent supercluster, occupying a covariantly trivial determinant wedge, carrying oriented cluster-lineage number three, baryon number $+1$, electric charge $+1$, and a sharpness-certified total-space spin-$1/2$ readout. None of these conditions requires a topological hole: the central experiment is precisely whether clusters and anti-clusters suffice. The paper separates exact identities, conditional theorems with certified hypotheses, numerical evidence, and new falsifiable conjectures, and it states the sharpest open question as a dichotomy: either an exact covariance-only proton exists, or a genuinely non-Gaussian, geometry-mediated interaction is required.

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

- on each edge, a complex squared length $z_e$ and a multiplicative $\mathbb{C}^{*}$ connection $U_e$ — two distinct fields, because the connection is gauge-variant and the geometry is not — together with one two-level occupation mode;

- incidence, Hodge, and Regge operators derived from that data;

- a generally entangled boundary/Fock state on those modes; and

- simplicial gluing followed by fermionic second quantization.

The declared microscopic fields end there. Spectral fibers, color frames, the derived rank-three color transport, spectral holonomies, particle sectors, and coarse vertices are *derived views* of that same data. They are not separately sampled fields. This is important scientifically: adding further independent fields could fit a desired answer, while deriving every readout from the declared data leaves the construction falsifiable. One consequence is recorded in Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>: every generator this ontology currently supplies is quadratic after second quantization, so the reachable states are exactly the quasi-free class together with whatever non-Gaussian data is fed at the boundary.

The formulation is also *complex-first*. Dynamical and physical formulas retain their full complex value. They do not apply $\operatorname{Re}$, $\operatorname{Im}$, $|\cdot|$, a sign function, a polar projection, or a principal logarithm/root in order to manufacture an observable. Complex amplitudes may therefore cancel before a final boundary probability is formed. Norms, singular values, and moduli may still be used as explicitly labeled *numerical certificates* for residual, conditioning, and convergence; they are not fed back as physical fields or substituted for the complex observable. A positive probability or compact unitary real form requires an antilinear quantum $*$-structure. This paper does not smuggle one into the Lorentzian bulk: it is supplied by certified boundary data or must emerge as a transport-compatible structure.

An arbitrary complex metric has no intrinsic ordering, timelike cone, or preferred square-root sheet. The causal datum used below is therefore only the orientation and incoming/outgoing coorientation already carried by the cobordism. Oriented intersection numbers are extracted from that discrete datum, not from the sign of a complex scalar. Complex Regge formulas are implemented through squared volumes and exponentiated holonomies wherever possible; when analytic continuation encounters unavoidable roots, the chosen Riemann sheet and its monodromy are carried as part of the state rather than reset to a principal branch. This is consistent with the care required for complexified spacetime metrics \[42\].

#### Convention and branch ledger.

The following choices are unavoidable, global, and reported; none is a new dynamical field:

- one incoming/outgoing cobordism coorientation fixes which integral lineage is called quark rather than anti-cluster; reversing it reverses all oriented charges together;

- simplicial orientation fixes exterior permutation parity, the only structural fermion sign;

- each spectral fiber carries its declared Riesz contour and its continuation under refinement;

- open determinant transports carry endpoint trivializations or a matched reference closure; no open-path winding is asserted without one;

- color transport retains $GL(3,\mathbb C)$, its determinant, and its projective class; no cube-root lift is silently selected;

- complex Regge roots carry a Riemann-sheet label and monodromy whenever a squared-volume formulation cannot avoid them;

- if a future amplitude integrates over complex geometric variables instead of evaluating a stationary carrier, its middle-dimensional integration cycle is supplied by the boundary problem and transported by analytic continuation; there is no canonical integral over all of $\mathbb C^N$;

- the Lorentzian momentum convention $p^2=m^2$ versus $p^2=-m^2$, the unit scale, and $B=N_q/3$ are fixed once as explicit calibrations; and

- a quantum $*$-structure is boundary data or an emergence certificate, never the result of taking a real or imaginary part in the bulk.

Topological holes do not appear in this ledger: their presence is measured by incidence homology, but their necessity for color or registers is the hypothesis being tested.

A second constraint governs how the four kinds of statement are kept apart in practice. Wherever an exact structural identity and an approximation determine the same object, the identity is what the claim rests on; an approximation is admissible only as a conditional theorem, carrying the certificates that bound it — for a spectral reduction, its complex spectral region, residual, contour separation, leakage, persistence, and conditioning. The reason is not economy. An uncertified tolerance silently becomes a physical postulate: the reader cannot tell which part of a reported number is the theory and which part is the numerics, and no falsification test can be run against a quantity whose error is undeclared.

<figure id="fig:concept">

<figcaption>Concept map for the recursive complex construction. Colors encode epistemic status, not physical sectors: blue is established or exact machinery, green is a derived observable, and amber is a proposed physical identification.</figcaption>
</figure>

# Present evidence in Tessera

Three existing results motivate the construction.

First, the state-operation-cobordism experiments show that the Hodge-carried register is a scaled isometry to machine precision — its Gram is the identity once one global scale is fixed by the anchor — and that its spectral value reproduces the quantum transition amplitude for every operation that the tested geometry actually carries. Generic fixed-complexity operations can remain obstructed, and the obstruction is visible both as a residual floor and as leakage from the carried subspace. The claim is therefore not that every finite complex realizes every gate; it is that a realized, isometrically embedded register computes the corresponding amplitude. The protocol reports the Gram residual and the carried-subspace leakage for every tested operation. In the reported flat-register suite, the scaled-isometry defect was $7.8\times10^{-16}$, the worst value deviation over $13\times9$ realized gate/pair tests was $5.0\times10^{-16}$, and the independent Choi reading agreed to $1.6\times10^{-16}$. These values belong to that finite, positive-register specialization; they motivate but do not prove the present complex-bilinear extension.

Second, interaction-history complexes exhibit a stable near-four-dimensional spectral regime. The strongest reported measurements approach, but do not yet prove, an exact spectral dimension of four; the return-probability estimator, its finite-size window, and its caveats accompany those measurements and bound what they establish. Diffusion-based spectral dimension on simplicial quantum geometries has important precedent in causal dynamical triangulations \[1\]; the Tessera evidence is an independent result for a different construction and should be compared at the level of the return-probability estimator and its finite-size window. The current reported peak is $D_S=4.245\pm0.024$ at $T=20{,}000$, with a naive geometric extrapolation $D_S(T\to\infty)\simeq4.07$. This is evidence for a near-four-dimensional window, not a proof of exactly four.

A third, more preliminary observation comes from joint Regge–Hodge stationarity experiments seeded with the phase pattern $\{1,\omega,\omega^{2}\}$, whose singlet diagnostics are evaluated while the stationarity equations change the complex. The construction does not force register holes to appear, and holes have not re-emerged under the unforced dynamics. That negative result is useful: the proposed quark should therefore not be defined as a hole. It will be sought as a persistent modular spectral cluster, while Betti numbers remain independent topological observables.

# The microscopic geometric state

Let $K$ be a finite oriented simplicial complex. For every edge $e$, store the complex squared length $z_e\in\mathbb C$ and, independently, a multiplicative connection variable. On an oriented edge from $x$ to $y$, $$U_{xy}\in\mathbb C^{*},\qquad U_{yx}=U_{xy}^{-1}.$$ The local occupation factor is $\mathcal{H}_e=\operatorname{span}\{\lvert0\rangle_e,\lvert1\rangle_e\}$. The global state is not a product of those factors; the factors specify only the local mode algebra.

The connection is stored multiplicatively, not as a phase $\varphi_e=\log U_e$. Under $g:K_0\to\mathbb C^{*}$, $$U_{xy}\longmapsto g_x^{-1}U_{xy}g_y ,
  \qquad z_e\longmapsto z_e .$$ Thus geometry is gauge-invariant while covariant hopping transforms by similarity. The elementary curvature datum is the branch-free face holonomy $$\mathcal F_\tau
  =\prod_{e\subset\partial\tau}U_e^{\,\epsilon_{\tau e}}\in\mathbb C^{*},$$ and infinitesimal variations use the Maurer–Cartan coordinate $U_e^{-1}\delta U_e$. Neither operation asks for an argument or logarithm. This microscopic Abelian connection is distinct from the derived rank-three transport of Section <a href="#sec:transport" data-reference-type="ref" data-reference="sec:transport">9</a>. Saying that an edge carries a qubit means that it carries the local mode algebra above, not that the global state is forced to be a product of normalized vectors.

## Topology and complex spectral geometry are separate

The integer incidence maps $$\partial_k:C_k(K)\longrightarrow C_{k-1}(K),
  \qquad \partial_{k-1}\partial_k=0$$ define topology before a metric is chosen: $$H_k(K;\mathbb C)=\ker\partial_k/\operatorname{im}\partial_{k+1},
  \qquad b_k=\dim H_k(K;\mathbb C).$$ Let $W_k(z)$ be an invertible complex metric weight and use the nondegenerate complex bilinear pairing $x^{\mathsf T}W_ky$. Its algebraic adjoint and Hodge operator are $$\partial_k^\sharp=W_k^{-1}\partial_k^{\mathsf T}W_{k-1},
  \qquad
  L_k^{H}=\partial_{k+1}\partial_{k+1}^{\sharp}
          +\partial_k^{\sharp}\partial_k .$$ For symmetric $W_k$ this gives the exact complex-symmetry identity $$(L_k^{H})^{\mathsf T}W_k=W_kL_k^{H}.$$ No positivity is used.

The ordinary positive-definite Hodge theorem does not extend by slogan to an arbitrary complex bilinear weight. Isotropic cancellations can enlarge or move the kernel, and even the coordinate representative of a degree-zero harmonic class need not be the constant vector. Accordingly this paper never identifies $\ker L_k^H$ with $H_k$ without a separate nondegeneracy theorem. Betti numbers are computed from incidence; spectral fibers are computed from the declared operator. A fiber may be non-harmonic and may live on a contractible cluster. This separation is what makes the no-hole hypothesis mathematically meaningful rather than terminological.

## The one-particle operator and its Riesz fibers

The untwisted geometric diagnostic is $L_k^H(z)$. The particle recursion uses a separately named covariant one-particle operator $h_k(z,U)$ obtained by replacing the relevant incidences or hoppings by their $U$-twisted versions. It satisfies $$h_k(z,U^g)=G_k(g)^{-1}h_k(z,U)G_k(g),
  \qquad h_k(z,1)=L_k^H(z)$$ for the chosen representation $G_k$ on $k$-cells. Color candidates use the edge-mode operator $h_1$; $L_0^H$ is never silently substituted for it.

Because $h_k$ is generically complex and non-normal, a band is selected by a closed contour $\Gamma_C$ in the complex spectral plane, not by sorting real parts or imaginary parts. Its exact Riesz projector is $$P_C=\frac{1}{2\pi i}\oint_{\Gamma_C}
       (\zeta I-h_C)^{-1}\,d\zeta ,
  \qquad P_C^2=P_C ,$$ with the contour, spectral separation, resolvent bound, rank, and refinement continuation reported as certificates \[40\]. A right frame and its algebraic dual are chosen so that $$P_C=\Phi_C\widetilde\Phi_C^{\mathsf T},
  \qquad
  \widetilde\Phi_C^{\mathsf T}\Phi_C=I .$$ This includes generalized eigenspaces when the enclosed part is defective. Changing the fiber frame by $g_C\in GL(r,\mathbb C)$ sends $\Phi_C\mapsto\Phi_Cg_C$ and $\widetilde\Phi_C^{\mathsf T}\mapsto
g_C^{-1}\widetilde\Phi_C^{\mathsf T}$, leaving $P_C$ unchanged.

Three layers of terminology are used from here on, and each carries a different word. The *microscopic configuration* is the edge data $(K,z,U)$: a coordinate on the space of operators, gauge-redundant, and not itself a state — an edge datum is not an amplitude. The *one-particle state* is the spectral data of $h(z,U)$: its eigenvalues together with its selected eigenspaces and, in a defective non-normal sector, the associated spectral projections and Jordan structure, entering below as band projectors. This is the gauge-invariant content of the edge data, and it is what the recursion transports. The *many-body state* is a vector or density operator on the Fock space built over those modes: the one-particle spectral data fix which modes exist, and the occupation and coherence data on them fix the many-body state. A matched left/right quasi-free state is represented without loss by its algebraic covariance as described in Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>; no positivity projection is part of that statement.

Writing $\mathfrak{h}_{K}=\operatorname{span}\{\lvert e\rangle : e\in K_{1}\}$ for the one-particle edge space, the microscopic quantum carrier is $$\mathcal{H}_{K}=\mathcal{F}_{-}(\mathfrak{h}_{K})=\Lambda^{\bullet}\mathfrak{h}_{K}
  \;\cong\;\widehat{\bigotimes}_{e\in K_{1}}\mathcal{H}_{e},$$ and a boundary many-body state is a vector or covector pair on $\mathcal{H}_K$. It may be entangled. A one-particle color state and the nonseparable proton-spin sectors are therefore native states, not exceptions to the ontology. For an isolated occupied band, the matched left/right quasi-free reference pair has the algebraic covariance $$\Gamma=\Phi\widetilde\Phi^{\mathsf T}=P,\qquad
  \Gamma^2=\Gamma,\qquad n_e=\Gamma_{ee}.$$ Thus a per-edge occupation is a derived readout. It remains a complex matrix element until a compatible $*$-structure identifies a physical bra with the adjoint of a ket. The full exterior space can represent non-Gaussian sectors, but Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a> records that no generator currently present produces them from Gaussian boundary data.

Both edge fields evolve. The squared lengths relax toward joint stationary points of the existing Regge and Hodge functionals; the multiplicative connection relaxes against the covariant operator it acts on. All variations are complex stationarity equations, never minimizations of a selected real projection. Since the spectrum and Riesz projectors transform covariantly, spectral stationarity is constant along gauge orbits. The variation $U^{-1}\delta U$ avoids choosing a logarithm branch during the update.

#### Holomorphic spectral constraints.

The notation for a targeted spectral calculation is fixed explicitly here. On an $M$-dimensional one-particle carrier, let $h=h_1(z,U):\mathfrak{h}_K\to\mathfrak{h}_K$ be the complex edge-mode operator above. For a positive integer $j$, define the power-sum spectral invariant $$p_j(h):=\operatorname{tr}(h^j).$$ This notation replaces the potentially ambiguous $I_j(h)$, reserving $I$ for an identity operator. Here $j$ is only the moment index; it is not a simplicial degree. If the eigenvalues of $h$, counted with algebraic multiplicity, are $\lambda_1,\ldots,\lambda_M$, then $$p_j(h)=\sum_{a=1}^{M}\lambda_a^j.$$ Thus $p_j$ is invariant under similarity, remains defined for a non-normal or defective matrix, and requires neither eigenvalue ordering nor a real projection. For a prescribed target multiset $\{\lambda_a^\star\}$, set $$p_j^\star=\sum_a(\lambda_a^\star)^j .$$ A holomorphic constrained action may then be written $$\mathcal S_{\mathrm{spec}}
  =\mathcal S_0(z,U)
   +\sum_{j=1}^{m}\xi_j\bigl(p_j(h(z,U))-p_j^\star\bigr),
  \qquad \xi_j\in\mathbb C .$$ Here $m\le M$ is the number of imposed moment constraints. The variable $\xi_j$ is an independent complex Lagrange multiplier, so stationarity in $\xi_j$ imposes the full complex equation $$\frac{\partial\mathcal S_{\mathrm{spec}}}{\partial\xi_j}
  =p_j(h)-p_j^\star=0.$$ For the full $M\times M$ operator, $p_1,\ldots,p_M$ determine the characteristic polynomial through Newton identities, but not its Jordan structure or spectral projectors. Those remain separately certified. For an isolated rank-$r$ fiber one instead uses $h_C=P_ChP_C|_{\operatorname{Ran}P_C}$ and $p_j(h_C)$ for $j=1,\ldots,r$. A numerical root finder may use a declared residual norm, but that norm is only a convergence certificate: the equations being solved are the complex constraints above. These target multipliers are absent in emergence mode and are permitted only in explicitly labeled controlled synthesis.

In emergence mode, particle-specific observables below are read after the stationary solve and are not inserted as target terms. Controlled synthesis mode may pin a carrier to test realizability, but that is a separate experiment. Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a> refines the emergence protocol into two labeled modes — strict no-backreaction, and certificates-blind mean-field backreaction — and records that both remain inside the quasi-free class.

This operator stack sits on established foundations: Regge calculus encodes piecewise-flat gravity in simplicial deficit angles \[2\], discrete exterior calculus supplies metric-dependent chain/cochain operators \[3\], and combinatorial Hodge spectra on simplicial complexes have a developed spectral theory \[4\]. Tessera’s proposal is not a replacement for those constructions; it is a constrained use of them as the sole source of the later particle readouts.

# A component is an exact static response vertex

Partition the cells carrying the declared one-particle operator $A$ into interface cells $B$ and interior cells $I$: $$A=\begin{pmatrix}A_{BB}&A_{BI}\\A_{IB}&A_{II}\end{pmatrix}.$$ If $A_{II}$ is invertible, the interior response equation has the exact solution $$x_I=-A_{II}^{-1}A_{IB}x_B$$ and the exact boundary operator $$\boxed{\,A_{\mathrm{eff}}
  =A_{BB}-A_{BI}A_{II}^{-1}A_{IB}\,}.$$ If $A_{II}$ is singular, Tessera uses a declared supported generalized inverse $A_{II}^{\#}$ and records its range and null projectors. Solvability is the purely algebraic left-kernel condition $$y^{\mathsf T}A_{IB}x_B=0
  \quad\text{for every }y\in\ker A_{II}^{\mathsf T}.$$ The inverse in $A_{\mathrm{eff}}$ is then replaced by $A_{II}^{\#}$ only if $A_{BI}\ker A_{II}=0$, so the boundary response is independent of the chosen interior solution; otherwise the null modes are retained as fiber coordinates. No Hermitian orthogonality is implied. This block elimination is the precise static sense in which a component becomes a response vertex. If a symmetric complex action matrix $K=K^{\mathsf T}$ supplies the equations, the same elimination applied to $K$ is stationarity of $x^{\mathsf T}Kx$ and gives the stationary boundary action. It is never called a minimum in the Lorentzian complex theory.

The plain Schur complement does *not* preserve the nonzero spectrum. For a spectral parameter $\lambda$ such that $A_{II}-\lambda I$ is invertible, define the exact Feshbach–Schur response $$\boxed{\,F_{B}(\lambda)=A_{BB}-\lambda I
    - A_{BI}(A_{II}-\lambda I)^{-1}A_{IB}\,}.$$ Then, for $\lambda$ outside $\operatorname{spec}A_{II}$, the exact determinant factorization $$\det(A-\lambda I)=\det(A_{II}-\lambda I)\,\det F_{B}(\lambda)$$ holds, so $$\lambda\in\operatorname{spec}A \iff 0\in\operatorname{spec}F_{B}(\lambda).$$ The order of the zero of $\det F_{B}(\cdot)$ at $\lambda$ equals the algebraic multiplicity of $\lambda$ in $A$, while $\dim\ker F_{B}(\lambda)$ equals its geometric multiplicity; the two agree in the self-adjoint or otherwise semisimple setting but not in general. At an interior resonance the inverse is replaced only after checking the compatibility condition $y^{\mathsf T}A_{IB}x_B=0$ for every $y\in\ker(A_{II}-\lambda I)^{\mathsf T}$ and retaining the resonant interior modes explicitly. Thus harmonic response uses $F_{B}(0)$, while a localized band enclosed by $\Gamma_C$ uses $F_B(\lambda)$ over a stated complex spectral region. A linear reduced eigenproblem may instead retain interface constraint modes plus selected fixed-interface modes using Craig–Bampton component-mode synthesis or AMLS; that route is certified approximation whose error is controlled by residuals and separation from discarded modes, not an exact spectral identity \[5, 6, 7\].

The effective blocks between coarse components become operator-valued links. A harmonic or retained interior mode is not discarded; it becomes an explicit stalk/fiber coordinate attached to the response vertex.

For graph Laplacians this is the classical Kron reduction by Schur complement \[8\]. Spectral graph reduction provides related approximation guarantees when additional coarsening or truncation is performed \[9\]. The extension proposed here is to apply static response reduction degree by degree to $L_k^H$ or to the declared covariant block $h_k$, and shifted Feshbach or certified component-mode reduction to nonzero bands, while retaining localized zero, resonant, and selected interior modes as explicit fiber coordinates.

# Recursive spectral fibers

Let $P_{\ell}=\{C_{v}^{\ell}\}$ be an intrinsic partition at scale $\ell$ into persistent connected components. At $\ell=0$ the object is the microscopic simplicial complex $K_{0}$. After the first elimination the honest coarse object is generally not another simplicial complex: it is an operator-valued response network $\mathcal{R}_{\ell+1}$ whose vertices carry vector spaces and whose links carry linear response blocks. A cellular sheaf on the quotient graph is a natural realization when the blocks admit compatible restriction-map factorization \[10\]; otherwise Tessera retains the more general response network and does not invent incidence maps that the reduction did not determine.

Within component $C$, choose an isolated localized spectral band by the Riesz contour of Section <a href="#sec:state" data-reference-type="ref" data-reference="sec:state">3</a>. The derived fiber is $$E_C=\operatorname{Ran}P_C,\qquad
  P_C=\Phi_C\widetilde\Phi_C^{\mathsf T},\qquad
  \widetilde\Phi_C^{\mathsf T}\Phi_C=I_r .$$ The band is allowed to be defective and non-normal. Left/right residuals, resolvent growth, projector norm, and frame condition number are therefore reported. The complex bilinear restriction $$B_C=\Phi_C^{\mathsf T}W_C\Phi_C$$ is retained as a complex matrix; its rank and determinant may diagnose a degenerate restriction, but no sign or inertia is extracted from it. Earlier conjugate-pair runs that classified sectors by a selected real projection remain historical numerical evidence, not an admissible definition in the complex-first formulation. Particle versus antiparticle is instead encoded by dual fiber and reversed oriented lineage in Section <a href="#sec:quarks" data-reference-type="ref" data-reference="sec:quarks">10</a>.

It need not be a harmonic space and therefore need not be supported by a hole. What it does require is complex-plane separation, localization, and persistence. A candidate component is accepted only if all of the following remain stable across a stated range of scales:

- a persistent connected cluster support, however proposed;

- a localized Riesz projector with stable rank;

- a closed complex-plane contour with nonzero separation and a controlled resolvent separating it from discarded modes;

- overlap with its predecessor and successor components;

- lifetime across multiple cobordism frames; and

- small external transport leakage.

Community objectives supply deterministic cluster candidates \[11\], while network renormalization supplies tests for genuine self-similarity rather than visual resemblance \[12\]. The partition is therefore a measured part of the analysis: a recursively drawn pattern is not evidence of a fractal unless its scaling observables survive a refinement window. The community-detection stage uses Newman–Girvan modularity on the combinatorial one-skeleton; it is a heuristic proposal generator that does not see complex Hodge weights and is subject to the modularity resolution limit \[13\]. Modularity may therefore propose candidate supports, but it may not veto an otherwise certified fiber: acceptance is conditioned only on the independent, weight-aware separation, localization, leakage, persistence, and refinement certificates above, together with the anchoring certificate of Section <a href="#sec:quarks" data-reference-type="ref" data-reference="sec:quarks">10</a> whenever a color interpretation is claimed.

This gives a type-stable hierarchy of response objects $$\cdots\longrightarrow\mathcal{R}_{2}\longrightarrow\mathcal{R}_{1}\longrightarrow K_{0}$$ in which a response vertex at one level resolves into a connected microscopic component plus retained stalk coordinates at the next finer level. “Self-similar” refers to closure of the response-network data type, not to a claim that every reduced operator is a simplicial Hodge Laplacian. A fractal-like pattern is permitted but not required: measured scaling of module count, volume, boundary size, and resolvent separation decide whether the hierarchy is statistically self-similar.

<figure id="fig:recursion">

<figcaption>One recursive step. Persistent connected modules become stalk-bearing vertices of an operator-valued response network. Static response is preserved by the supported Schur complement; nonzero bands use shifted Feshbach or certified component-mode reduction. Selected internal modes remain attached as fibers, and a persistent supermodule can be reduced again at the next scale.</figcaption>
</figure>

# Interactions and the expanding Hilbert space

Two operations must not be conflated. For the Cartesian product of chain complexes $A$ and $B$, the graded tensor differential is the exact rule $$d_{A\mathbin{\widehat{\otimes}}B}(a\otimes b)=d_{A}a\otimes b+(-1)^{\deg a}a\otimes d_{B}b.$$ For a noninteracting product with product metric, $$L_{A\mathbin{\widehat{\otimes}}B}=L_{A}\otimes I+I\otimes L_{B},$$ so one-particle eigenvalues add and eigenvectors tensor. This identity is about a product complex, not about gluing two cobordisms.

Actual simplicial gluing is a pushout along a shared boundary. At the one-particle level it produces a chain space assembled from direct sums modulo boundary identifications (equivalently described by the relevant Mayer–Vietoris sequence) and, for the declared covariant dynamics, a block operator $$h_{A\cup B}=\begin{pmatrix} h_A & C_{AB}\\ C_{BA} & h_B\end{pmatrix}$$ in a basis adapted to the two interiors. The coupling blocks are induced by the connecting simplices and shared-boundary constraints; they are not a Kronecker interaction term.

The expanding Hilbert space follows after applying the fermionic Fock functor to the one-particle space $\mathfrak{h}$. The exact identities are $$\mathcal{F}_{-}(\mathfrak{h}_{A}\oplus\mathfrak{h}_{B})\cong\mathcal{F}_{-}(\mathfrak{h}_{A})\mathbin{\widehat{\otimes}}\mathcal{F}_{-}(\mathfrak{h}_{B}),$$ and $$d\Gamma(h_A\oplus h_B)=d\Gamma(h_A)\mathbin{\widehat{\otimes}}I+I\mathbin{\widehat{\otimes}}d\Gamma(h_B).$$ For the two directed coupling blocks, the algebraic second quantization is $$d\Gamma(C)=
  \sum_{ij}(C_{AB})^{i}{}_{j}\,\varepsilon_{A,i}\iota_{B}^{j}
  +\sum_{ij}(C_{BA})^{i}{}_{j}\,\varepsilon_{B,i}\iota_{A}^{j},$$ where $\varepsilon$ is exterior creation and $\iota$ is contraction by the dual mode. No Hermitian-conjugate relation between $C_{AB}$ and $C_{BA}$ is assumed. Such a relation may be reported only after a compatible $*$-structure is certified. Thus geometric connections become hopping terms without adding a new field. If the one-particle eigenvalues are $\lambda_{1},\dots,\lambda_{M}$, then the free many-body spectrum is the set of occupation subset sums $\sum_{i}n_{i}\lambda_{i}$, $n_{i}\in\{0,1\}$, rather than the one-particle pairwise spectrum being relabeled as a Fock spectrum \[14\].

At the selected-fiber level, an interaction grows the carried many-body space as $$\mathcal{H}_{AB}=\mathcal{F}_{-}(E_A\oplus E_B)
  \cong\mathcal{F}_{-}(E_A)\mathbin{\widehat{\otimes}}\mathcal{F}_{-}(E_B),$$ and a later interaction appends another one-particle summand, hence another graded Fock factor. This is a statement about state-space composition after second quantization, not the topology of the glued chain complex. When carried subspaces of adjacent components overlap on interface cells, the composite is built on the abstract labeled sum with an explicit embedding/dual overlap matrix; Section <a href="#sec:master" data-reference-type="ref" data-reference="sec:master">15</a> states the exact rule. If $J_C$ embeds an abstract state into the geometric carrier and $\widetilde J_C^{\mathsf T}$ is assembled from the transported local left Riesz frames, exact amplitude preservation requires $$G_C=\widetilde J_C^{\mathsf T}J_C=I.$$ The left embedding is fixed before this test; replacing it post hoc by an arbitrary global dual would force $G=I$ by definition and destroy the certificate. Tensor products preserve the pairing exactly. If $G=\widetilde J_C^{\mathsf T}J_C$ has overlap defect $\Delta G=G-I$, the full complex amplitude error is exactly $$\widetilde a^{\mathsf T}Gb-\widetilde a^{\mathsf T}b
  =\widetilde a^{\mathsf T}\Delta G\,b.$$ For numerical certification only, any declared subordinate norm gives $|\widetilde a^{\mathsf T}\Delta G\,b|
\le\|\widetilde a\|\,\|\Delta G\|\,\|b\|$. If $\varepsilon=\|\Delta G\|$, two tensor factors obey $$\varepsilon_{AB}\le\varepsilon_{A}+\varepsilon_{B}
    +\varepsilon_{A}\varepsilon_{B}.$$ Thus the complex amplitude is retained, while its numerical error certificate has an explicit composable budget.

Cobordism composition as a map between boundary state spaces is the organizing idea of topological field theory \[15\]; the general-boundary program makes the region/boundary assignment explicit for quantum theory \[16\], and categorical quantum mechanics formalizes tensor composition and diagrammatic process semantics \[17\]. Tessera keeps only the parts that can be realized by its finite simplicial carrier and tests the resulting map numerically rather than assuming topological invariance.

# Quasi-free dynamics and the covariance layer

Every many-body generator exhibited in this paper is quadratic. Free propagation is $d\Gamma(h_1)$, gluing contributes $d\Gamma$ of a coupling block, and every derived transport is the second quantization of a one-particle map. The exact consequence is closure of the quasi-free class: if the many-body generator is always of the form $$H(t)=d\Gamma\bigl(h(t)\bigr)
  =\sum_{ij}h^{i}{}_{j}(t)\,\varepsilon_i\iota^j,$$ then Gaussian/quasi-free states remain Gaussian.

The closure survives self-consistency. Let the one-particle operator depend on the covariance and on the classical geometry, $$h=h\bigl(\Gamma(t),z(t),U(t)\bigr),$$ with the geometry in turn made stationary against the state’s bilinear action density. That is nonlinear mean-field dynamics of generalized Hartree–Fock type \[18\]; it can localize and it can produce self-bound solutions, but it does not leave the Gaussian manifold. Classical or mean-field geometry backreaction alone therefore does not generate genuinely non-Gaussian correlations.

The emergence protocol accordingly splits into two labeled modes, both Gaussian-closed: *strict emergence*, in which the state does not act back on the geometry at all, and *certificates-blind mean-field backreaction*, in which the carried state’s bilinear action density enters the joint stationarity equations while every particle certificate remains firewalled from them. The certificate firewall of Section <a href="#sec:proton" data-reference-type="ref" data-reference="sec:proton">14</a> applies to both modes.

Genuinely non-Gaussian correlations would require at least one of the following, none of which is currently part of the model:

1.  a genuine quartic effective interaction, $$H_{\mathrm{int}}
          =\sum_{ijkl}V_{ij}{}^{kl}\,
            \varepsilon_i\varepsilon_j\iota^l\iota^k;$$

2.  quantized geometry that becomes entangled with the fermions;

3.  integrating out dynamical geometry beyond the mean-field approximation, producing a retarded or quartic effective interaction;

4.  a cobordism map that is not the second quantization of a one-particle map; or

5.  measurement or postselection capable of taking Gaussian states outside the Gaussian class.

Adopting one of these is an explicit scope decision with its own certificates, not a background assumption. Until then, the statement that non-Gaussian sectors are representable (Section <a href="#sec:state" data-reference-type="ref" data-reference="sec:state">3</a>) must not be read as a statement that they are produced.

Among these possibilities, eliminating geometric fluctuations is the minimal mechanism that uses no new particle field. Let $x=(x_1,\ldots,x_R)^{\mathsf T}\in\mathbb C^R$ be retained geometric fluctuations about a stationary carrier, and suppose on a certified local domain that $$S_g(x)=\frac12x^{\mathsf T}Ax,
  \qquad
  h(x)=h_0+\sum_{a=1}^{R}x_aO_a,
  \qquad A=A^{\mathsf T},\quad \det A\ne0 .$$ For independent left/right Grassmann variables $\widetilde\psi,\psi$, define the bilinear geometric currents $$J_a=\widetilde\psi^{\mathsf T}O_a\psi .$$ The joint complex action is $$S(x,\widetilde\psi,\psi)
  =\frac12x^{\mathsf T}Ax
   +\widetilde\psi^{\mathsf T}h_0\psi+x^{\mathsf T}J .$$ Stationarity in $x$ gives $x=-A^{-1}J$, and exact substitution gives $$\boxed{
  S_{\mathrm{eff}}
  =\widetilde\psi^{\mathsf T}h_0\psi
   -\frac12J^{\mathsf T}A^{-1}J .}$$ The second term is quartic in the fermionic variables and is generically non-Gaussian. No conjugation, real projection, or new microscopic field has been introduced. The result is exact when the certified geometric block is quadratic and its matter coupling is linear; nonquadratic geometry produces higher effective interactions and must carry truncation certificates. Replacing $J$ by its expectation before elimination is only mean field and does *not* produce this many-body correlation. Computationally, the interaction remains factored through the $R\times R$ geometric response $A^{-1}$, so sparse solves can be used without constructing a dense four-index tensor. This is a proposed next-stage mechanism, not an interaction silently enabled in the present emergence runs.

The quasi-free formulation is especially attractive in the complex theory because it does not require a positivity projection. For $N$ occupied right modes and their matched algebraic duals, define $$\lvert\Phi_R\rangle=\phi_1\wedge\cdots\wedge\phi_N,\qquad
  \langle\Phi_L\rvert
    =\widetilde\phi^{\,1}\wedge\cdots\wedge\widetilde\phi^{\,N},
  \qquad
  \widetilde\Phi^{\mathsf T}\Phi=I_N .$$ Their overlap is one and the complete one-body datum is $$\Gamma=\Phi\widetilde\Phi^{\mathsf T},\qquad
  \Gamma^2=\Gamma,\qquad \operatorname{tr}\Gamma=N.$$ This is an algebraic Slater covariance, not a positive density matrix.

For an arbitrary complex one-particle generator, evolve the two frames by $$i\dot\Phi=h\Phi,\qquad
  -i\dot{\widetilde\Phi}^{\mathsf T}
     =\widetilde\Phi^{\mathsf T}h .$$ Then the dual pairing and idempotency are preserved and $$\boxed{\,i\dot\Gamma=[h,\Gamma]\,}$$ holds exactly. No $h^\dagger$, scalar renormalization, pseudo-Hermiticity, or positive metric is needed. A time-dependent frame contributes its ordinary algebraic connection to both equations and cancels in the same covariant commutator.

The biorthogonal Wick theorem reduces every polynomial left/right matrix element to contractions of $\Gamma$ \[41\]. Occupation amplitudes, parity, determinant wedges, and the complex spin-sharpness polynomial are therefore computable in polynomial space without constructing the exponential Fock vector. A Nambu doubling may be introduced if a pairing generator is actually derived, but it is not presumed.

A quantum probability is a different layer. If boundary preparation supplies, or bulk transport preserves, an antilinear involution $*$ and a compatible nondegenerate Hermitian form $H$, one may impose $\widetilde\Phi^{\mathsf T}=\Phi^\dagger H$ and recover the usual positive covariance wherever $H$ is positive on the measured boundary sector. Until that certificate exists the theory reports complex transition amplitudes and their exact cancellations, not probabilities. Likewise, no completely positive open-system evolution is claimed; that would require an explicit Lindblad generator \[36\].

The programme order follows. First test the strongest possible covariance-only theory. Treat failure of the sharp proton certificate of Section <a href="#sec:proton" data-reference-type="ref" data-reference="sec:proton">14</a> as a meaningful structural result rather than a numerical nuisance. Introduce a non-Gaussian interaction only if the geometry supplies one naturally, through one of the mechanisms above. The question that decides the next stage of the programme is stated exactly:

<div class="center">

</div>

Either answer is informative. A covariance-only proton would make the entire particle layer polynomially computable and exactly certifiable; a demonstrated obstruction would be the first internal evidence that the geometry must supply a true interaction term.

# A triangle carries the exact color algebra

Consider the three edge-mode factors around an oriented triangle and interpret $\lvert 1\rangle$ as an occupied edge mode. Choosing an oriented ordering $(e_1,e_2,e_3)$ defines a three-dimensional complex mode space $E$ and identifies their graded tensor product with $$(\mathbb C^2)^{\mathbin{\widehat{\otimes}}3}\cong\Lambda^\bullet E
  =\mathbb C\oplus E\oplus(\det E\otimes E^\vee)\oplus\det E .$$ The orientation of one triangle fixes the ordering up to a cyclic, hence even, permutation, so the local wedge sign is unambiguous. Globally the exterior algebra $\Lambda^{\bullet}\mathfrak{h}_{K}$ and the CAR are intrinsic; only a presentation in ordered tensor factors needs a deterministic mode order and the corresponding permutation parity. A Kasteleyn orientation is useful for two-dimensional surface-dimer Pfaffians but is not required to define this abstract Fock space \[19\]. A genuine continuum spinor interpretation is a separate question addressed by the rotation certificate below.

The sectors have occupation number $N=0,1,2,3$:

<div id="tab:sectors">

| Sector       | Dimension | Complex representation | Fermion parity |
|:-------------|:---------:|:-----------------------|:--------------:|
| $\Lambda^0E$ |     1     | scalar vacuum          |      even      |
| $\Lambda^1E$ |     3     | fundamental $E$        |      odd       |
| $\Lambda^2E$ |     3     | $\det E\otimes E^\vee$ |      even      |
| $\Lambda^3E$ |     1     | determinant line       |      odd       |

Exterior sectors of three oriented edge modes.

</div>

Let $\varepsilon_i$ be exterior creation by $e_i$ and $\iota^j$ contraction by the dual basis $e^j$. They satisfy the algebraic canonical anticommutation relations exactly: $$\{\iota^i,\iota^j\}=0,\qquad
  \{\varepsilon_i,\varepsilon_j\}=0,\qquad
  \{\iota^i,\varepsilon_j\}=\delta^i_j.$$ On the one-occupation sector, the bilinears $$E^i{}_j=\varepsilon_i\iota^j$$ satisfy $$=\delta^k_jE^i{}_\ell-\delta^i_\ell E^k{}_j.$$ These are $\mathfrak{gl}(3,\mathbb C)$; the traceless combinations are exactly $\mathfrak{sl}(3,\mathbb C)$. Thus the triangle carries a fundamental, its determinant-twisted dual, a determinant line, and the complex adjoint algebra without any conjugation. A coherent trivialization of $\det E$ reduces $GL(3,\mathbb C)$ to $SL(3,\mathbb C)$ and identifies $\Lambda^2E\simeq E^\vee$. Only a further transport-compatible *positive-definite* Hermitian form selects the compact real form $SU(3)$ and turns the dual into the usual $\overline{\mathbf3}$. A nondegenerate Hermitian form of signature $(p,q)$ would instead select the noncompact real form $SU(p,q)$. Compact color is therefore an emergence certificate, not a microscopic restriction.

The triplet description of quarks and the three-quark construction of baryons originate with the quark model \[20\]; the additional color triplet was introduced to resolve the statistics and state-counting problem \[21, 22\]. The claim here is narrower and new: Tessera’s three oriented edge modes would provide a geometric carrier of the same representation content, not a derivation of QCD from the combinatorics alone.

<figure id="fig:triangle">

<figcaption>Exact complex representation content of three oriented edge-mode factors. The exterior sectors and parity are algebraic identities. Interpreting <span class="math inline"><em>E</em></span> as quark color and a covariantly trivial determinant wedge as a baryon singlet is the physical hypothesis; compact <span class="math inline"><em>S</em><em>U</em>(3)</span> requires a certified <span class="math inline">*</span>-structure.</figcaption>
</figure>

## Projective geometric carrier

For complex squared lengths $(z_1,z_2,z_3)\ne0$ on the oriented edges, the branch-free color direction is the projective ray $$=[z_1:z_2:z_3]\in\mathbb{CP}^{2}.$$ No norm or square root is needed. A constraint such as $z_1+z_2+z_3=1$, where the sum is nonzero, is only an affine scale chart; a face-holonomy constraint $\mathcal F_\tau=1$ is a separate gauge-curvature condition. Neither condition makes the triangle itself an $SU(3)$ group manifold. If a boundary $*$-structure is later certified, the ray may be given its usual Hilbert normalization as a boundary readout.

## The existing omega phase pattern

Let $\omega$ be either nontrivial solution of $\omega^2+\omega+1=0$; choosing which one labels the orientation convention, and reversing the oriented triangle exchanges the two. The branch-free cyclic Fourier frame is $$\mathcal F_{3}=
  \begin{pmatrix}
    1 & 1 & 1\\
    1 & \omega & \omega^{2}\\
    1 & \omega^{2} & \omega
  \end{pmatrix}$$ and has nonzero determinant. The existing pattern $[1:\omega:\omega^2]$ is one projective color direction, not the whole fiber; its cyclic orbit supplies an exact algebraic basis. With the conventional positive boundary Hermitian form, multiplying by $1/\sqrt3$ gives the familiar unitary Fourier frame. That normalization belongs to the boundary $*$-structure and is not used in the bulk dynamics.

# Complex color transport and spectral holonomy

Let $T_{AB}$ be the chain-level transfer already induced by connecting simplices from component $B$ to component $A$. In matched Riesz frames of common rank $r$, the fiber map is $$M_{AB}=\widetilde\Phi_A^{\mathsf T}T_{AB}\Phi_B .$$ Under independent frame changes it transforms exactly as $$M_{AB}\longmapsto g_A^{-1}M_{AB}g_B .$$ When $\det M_{AB}\ne0$ it is retained as an element of $GL(r,\mathbb C)$. It is not projected to a polar factor. The numerical leakage certificate is instead computed before restriction, for example $$\ell_{AB}=\|(I-P_A)T_{AB}P_B\|,$$ and is reported with $\det M_{AB}$, the condition number, endpoint resolvent bounds, and left/right frame residuals. The norm here certifies the approximation; it does not replace $M_{AB}$ as the observable.

For a closed sequence $\gamma=(A_0,A_1,\ldots,A_n=A_0)$, with $M_{AB}:E_B\to E_A$, the full holonomy is $$H(\gamma)
  =M_{A_0A_{n-1}}\cdots M_{A_2A_1}M_{A_1A_0}.$$ transforms by $H\mapsto g_{A_0}^{-1}Hg_{A_0}$. Its characteristic polynomial, $\operatorname{tr}H^m$, determinant, and conjugacy class are therefore branch-free gauge observables. For a differentiable family of Riesz projectors, the Kato equation $$\dot{\mathcal U}=[\dot P,P]\,\mathcal U$$ gives the corresponding parallel transport inside the isolated bundle \[40\].

At rank three the faithful complex data are $$M_{AB}\in GL(3,\mathbb C),\qquad
  \delta_{AB}=\det M_{AB}\in\mathbb C^{*},\qquad
  [M_{AB}]\in PGL(3,\mathbb C).$$ A coherent determinant-line trivialization reduces this to $SL(3,\mathbb C)$, with projective quotient $PSL(3,\mathbb C)=SL(3,\mathbb C)/\mathbb Z_3$. Selecting $M/(\det M)^{1/3}$ independently on each link is forbidden: it chooses one of three sheets and discards the determinant transport. If a compatible positive-definite Hermitian form $H_C=H_C^\dagger>0$ later satisfies $M_{AB}^{\dagger}H_AM_{AB}=H_B$, the same data admit a compact $U(3)$, and with determinant trivialization an $SU(3)$, real form. An indefinite preserved Hermitian form gives $U(p,q)$ instead and is reported as such. Compact Wilson loops are a certified specialization, not the bulk definition.

The determinant line supplies a possible oriented flux readout. For a closed, full-rank world-tube family $M(t)$, $$\boxed{\,
  \nu=\frac{1}{2\pi i}\oint
  \operatorname{tr}\!\left(M^{-1}dM\right)
  =\frac{1}{2\pi i}\oint\frac{d(\det M)}{\det M}
  \in\mathbb Z\,}$$ is homotopy-invariant while the spectral contour and rank remain open. No argument or logarithm is chosen. Reversing the oriented loop sends $M\mapsto M^{-1}$ and $\nu\mapsto-\nu$. The integer character requires a closed loop. A quark or proton tube on a cobordism segment is an interval, so the open-segment definition is relative: either compose the physical transport with the inverse of a matched reference transport — the same non-exchanging reference construction used in Section <a href="#sec:exchange" data-reference-type="ref" data-reference="sec:exchange">11.1</a> — so that the composite closes, or fix endpoint trivializations supplied by the boundary registers. The reported $\nu$ is the integer winding of that closed composite, together with its reference specification. Without such closure no integer is emitted. Identifying $B=\nu/3$ is a proposed physical calibration, cross-checked against the independent cluster-lineage intersection of Section <a href="#sec:readout" data-reference-type="ref" data-reference="sec:readout">13</a>.

The dual transport is branch-free: $$M_{AB}^{\vee}=M_{AB}^{-\mathsf T},\qquad
  \det M_{AB}^{\vee}=(\det M_{AB})^{-1}.$$ This is the transport assigned below to an anti-cluster. Pair creation conserves total determinant winding under any continuous dual-pair homotopy that avoids $\det M=0$ and boundary flux.

This construction is the complex spectral-frame analogue of Berry transport \[23\] and its non-Abelian Wilczek–Zee generalization \[24\]. Overlap matrices are also a standard route to gauge-covariant lattice observables \[25\], as are connection Laplacians and vector diffusion maps \[33\]. Wilson loops themselves are foundational lattice gauge observables \[26\]. Those standard constructions usually use unitary frames; the algebraic $GL(r,\mathbb C)$ extension here keeps conjugacy covariance without imposing a compact real form. What is specific to Tessera is that the rank-three link is reconstructed from neighboring Riesz fibers, not independently assigned, and is accompanied by leakage and conditioning certificates.

# Quarks as modular clusters

A *quark candidate* is proposed to be a component $Q$ satisfying all of the following derived conditions.

The abstract rank-three band must first be anchored to oriented two-simplices. Choose one base vertex $p$ in $Q$. For an oriented triangle $\tau\subset Q$, let $\mathcal R_{\tau\to p}(U)$ restrict to its three ordered boundary edges and parallel-transport those three coefficients to $p$ using the already declared microscopic connection. The same base and declared path rule are used for every face. It is required to satisfy $$\mathcal R_{\tau\to p}(U^g)G_1(g)^{-1}
  =g_p^{-1}\mathcal R_{\tau\to p}(U).$$ The dressed Plücker coordinate $$\Delta_\tau
  =\det\!\left(\mathcal R_{\tau\to p}(U)\Phi_Q\right)\in\mathbb C$$ then transforms by the common factor $g_p^{-3}\det g_Q$ under microscopic gauge and $\Phi_Q\mapsto\Phi_Qg_Q$. Hence the profile $$=[\Delta_\tau]_{\tau\subset Q}
  \in\mathbb P\!\left(\mathbb C^{\,K_2(Q)}\right)$$ is frame-independent wherever it is nonzero. It records the complete complex interference pattern across candidate anchoring faces without selecting a largest modulus or fitting convex weights after seeing the band. Dependence on the declared paths is exactly microscopic face holonomy and is reported, not erased.

For a second fully invariant coordinate, let $\Pi_\tau(U)$ be the connection-dressed endomorphism that selects the same face atlas and satisfies $\Pi_\tau(U^g)=G_1(g)^{-1}\Pi_\tau(U)G_1(g)$. Define $$\alpha_\tau=
  \det\!\left(
    \widetilde\Phi_Q^{\mathsf T}
    \Pi_\tau(U)\Phi_Q
  \right)\in\mathbb C .$$ The inner matrix transforms by similarity under both gauge groups, so $\alpha_\tau$ is invariant. The anchor certificate is the persistent nonzero projective profile $[\Delta_Q]$, the complex profile $\{\alpha_\tau\}$, and determinant-line transition functions on overlaps. An extended fiber may be anchored by an atlas of faces; concentration on one face is not required. Numerical projective distances and condition numbers may be reported to certify stability, but no modulus, square root, free face weight, or real-valued score enters the physical definition. If the required connection-dressed covariance cannot be verified, the anchor refuses rather than reporting a gauge-dependent raw restriction. Because every 2-simplex has exactly three boundary edges, the construction is independent of ambient spectral dimension.

1.  $Q$ is a persistent cluster certified as in Section <a href="#sec:fibers" data-reference-type="ref" data-reference="sec:fibers">5</a>, however its support was proposed.

2.  Its selected color fiber has stable rank three.

3.  Its anchor atlas — projective profile, invariant coordinates, and determinant transitions — is stable.

4.  It occupies an odd exterior sector.

5.  Its $GL(3,\mathbb C)$ color transport stays full rank and has numerically bounded leakage over its lifetime.

6.  Its oriented cluster lineage has intersection number $N_Q=+1$ with a separating cut, and, where the determinant family closes interferometrically, relative winding $\nu=+1$. Reversing the lineage gives $N_Q=-1$, the dual fiber $E_Q^\vee$, transport $M^{-\mathsf T}$, and an anti-cluster. The two integer routes must agree.

7.  Its total spectral fingerprint is stable under refinement and vertex relabeling.

The distinction between an anti-cluster and the $\Lambda^2E$ sector of two quarks is made by dual lineage, determinant transport, and total occupation, not color alone. The primary integer is the oriented intersection $N_Q$; determinant winding is an independent agreement test when a closed relative family exists. Neither construction refers to a cycle in the support, a Betti number, or a topological hole. A cluster can be contained in a contractible four-ball and still have an oriented lineage crossing the cobordism.

Flavor and electric charge are not assumed as hidden labels. The conservative hypothesis is that two stable subclasses of the same cluster fiber provide an isospin doublet. On such a doublet, the measured orientation flux supplies baryon number and the standard relation $$Q=I_{3}+\frac{B}{2}$$ gives $Q_{u}=+2/3$ and $Q_{d}=-1/3$. This is a proposed identification, not yet a derivation: it succeeds only if an unlabeled two-dimensional spectral band emerges, is transported coherently, and its flavor-derived current agrees with the microscopic Ward-current flux at those values.

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

#### Odd-rank determinant cross-check (proposed).

There is a second geometric parity available at the fiber level. Interchanging two complete blocks $E_A$ and $E_B$ of ranks $r_A$ and $r_B$ acts on the orientation line of $E_A\oplus E_B$ with determinant $$\det P_{AB}=(-1)^{r_Ar_B}.$$ Two rank-three color fibers therefore give $(-1)^9=-1$ without extracting a sign from a complex amplitude. This identity is exact, but its physical use is conditional: it is the determinant of exchanging the *whole fiber frames*, not automatically the exchange of one occupied state. Promoting it to particle statistics amounts to the falsifiable hypothesis that cluster parity is fiber rank modulo two. The core construction continues to use occupation parity in $\Lambda^\bullet\mathfrak{h}_K$; simulations may report the rank-parity character as an independent cross-check and must not count both minus signs in the same exchange.

Pauli exclusion is the exact exterior-algebra statement $$v_1\wedge\cdots\wedge v_n=0
  \quad\Longleftrightarrow\quad
  v_1,\ldots,v_n\ \text{are linearly dependent}.$$ With any matched dual covectors $\alpha^i$, the coefficient is the complex determinant $\det[\alpha^i(v_j)]$; no norm is needed. If two complete one-particle modes coincide, the state vanishes. The complete-mode qualifier prevents double-counting the orientation parity: color, spin, flavor, space, and component support are wedged once as one mode.

## A label-independent exchange experiment

Let $H_{\mathrm{ex}}$ be the full $GL(r,\mathbb C)$ holonomy of the isolated odd subspace during an exchange and let $H_{\mathrm{ref}}$ be the holonomy of a non-exchanging reference motion with the same geometric footprint, timing, contours, and local frame convention. The branch-free interferometric exchange character is $$\chi_F=\det\!\left(H_{\mathrm{ex}}H_{\mathrm{ref}}^{-1}\right)
  \in\mathbb C^{*}.$$ The proposed dynamical test is the complex equation $$\chi_F(\text{single exchange})=-1,
  \qquad
  \chi_F(\text{double exchange})=+1.$$ No phase angle, sign of a component, or unit-modulus projection is taken.

As an independent structural cross-check, persistent component matching extracts the permutation $P_{\gamma}$ of localized odd blocks and reports $\operatorname{sgn}P_{\gamma}$ together with the norm of the residual in-block motion after comparison with the reference loop. The algebraic wedge sign is exact; the interferometric holonomy is the dynamical certificate. This permutation parity is the one deliberate discrete sign in the statistics construction. It is fixed by oriented exterior algebra rather than extracted from a complex dynamical variable.

A physical $2\pi$ rotation uses the same reference normalization, and the rotation path is not left abstract: it is a closed loop of the total-space frame generated by the existing total-space rotation construction, normalized against its own co-moving, non-rotating reference. The $J^{2}$ sharpness test is a separate calculation on the transported state; its construction and oracle values do not depend on the rotation cycle, and the sharp total-space spin readout itself remains an open experimental item. In the complex bulk, a continuum spinor claim additionally requires a lift of the complex orthogonal frame holonomy from $SO(d,\mathbb C)$ to $\mathrm{Spin}(d,\mathbb C)$. If a real Lorentzian section is certified at the boundary, this reduces to the corresponding $SO^{+}(1,d-1)$ to $\mathrm{Spin}^{+}(1,d-1)$ lift; obstruction by the second Stiefel–Whitney class is then a falsification certificate \[27\]. This requirement concerns the physical spin lift, not the existence of the abstract CAR/Fock algebra. Here $SO(d,\mathbb C)$ is the algebraic complexification of the Lorentzian frame group, not a Euclidean continuation or Wick rotation. The spin-statistics comparison is cleanest as $$\chi_F(\text{exchange})\,
  \chi_F(2\pi\ \text{rotation})^{-1}=+1,$$ with each factor separately converging to the complex number $-1$.

Configuration-space topology already explains how exchange classes can carry quantum phases \[28\] and, in two dimensions, more general statistics \[29\]. The Tessera proposal uses this precedent only as a diagnostic template. The minus sign from the graded exterior algebra is exact; the claim that an actual geometric exchange cobordism realizes the corresponding determinant holonomy remains an experiment.

# Fock space as an inductive limit of interactions

For $M$ oriented fermionic edge modes, $$\widehat{\bigotimes}_{m=1}^{M}\mathbb{C}^{2}
  \cong\Lambda^{\bullet}\mathbb{C}^{M}
  =\bigoplus_{n=0}^{M}\Lambda^{n}\mathbb{C}^{M},$$ and the dimension identity is exact: $$2^{M}=\sum_{n=0}^{M}\binom{M}{n}.$$

The exterior algebra is canonical as a functor of the one-particle space. Writing it as a literal ordered tensor product, or presenting the creation operators in Jordan–Wigner form, requires a chosen mode order. A deterministic order is fixed by oriented component lineage, with the parity of every reordering applied; all reported observables must be invariant under relabeling plus the induced exterior-algebra map.

Adding a new noninteracting mode uses the vacuum embedding $$\iota_{M}:\mathcal{H}_{M}\hookrightarrow\mathcal{H}_{M+1},
  \qquad
  \iota_{M}(\psi)=\psi\mathbin{\widehat{\otimes}}\lvert 0\rangle.$$ The infinite Fock space is the direct limit $$\mathcal{F}=\varinjlim(\mathcal{H}_{M},\iota_{M}).$$ This makes the infinite expansion precise while every stage remains finite: at any finite stage only finitely many modes have interacted. Consistency requires $$\lVert\iota_M\mathcal U_M-\mathcal U_{M+1}\iota_M\rVert
  \longrightarrow 0$$ over a refinement sequence. This norm is a numerical compatibility certificate; the maps $\mathcal U_M$ need not be unitary in the complex bulk.

A bosonic gauge sector, if realized, need not add a new local oscillator. The exact statement is representation-theoretic: the traceless even bilinears $$\varepsilon_i\iota^j-\frac{1}{3}\delta_i^jN$$ transform in the complex adjoint sector of $E\otimes E^\vee=\mathbb C\oplus\mathfrak{sl}(E)$ and have even fermion parity. After a compact color real form is certified, this is the familiar $\mathbf3\otimes\overline{\mathbf3}=\mathbf1\oplus\mathbf8$. It identifies the octet quantum numbers among collective fermion-pair excitations; it does not by itself establish propagating bosonic gauge excitations, which would require separate dynamical and continuum evidence. Within this model, arbitrarily many such collective excitations are represented by adding more microscopic modes at finer resolution, and each finite edge-mode factor remains two-dimensional.

This scale-by-scale state growth is adjacent to entanglement renormalization, where local Hilbert data are reorganized across layers before truncation \[30\]. The distinction is material: Tessera uses static/shifted response reduction plus an inductive vacuum embedding, and it must certify compatibility between successive finite spaces rather than assume a fixed bond dimension.

## Occupation exterior algebra is not automatically Kähler–Dirac

The exterior algebra above is over the one-particle *mode space*; its degree is occupation number. A Kähler–Dirac field instead lives on the inhomogeneous differential-form/cochain space $\bigoplus_{k}C^{k}(K)$ and is acted on by a first-order operator such as $d-\delta$, with $\delta$ the coderivative supplied by the declared metric structure. These constructions share exterior-algebra notation but are not the same operator or grading. Consequently the present model does not inherit lattice taste multiplicity merely from using $\Lambda^{\bullet}\mathfrak{h}_{K}$.

If a later Tessera model promotes its one-particle field to all cochain degrees and uses the Kähler–Dirac operator, the known flat four-dimensional decomposition into four Dirac spinors becomes an expected spectrum diagnostic, not an unexplained bug \[31, 32\]. Any observed near-fourfold cluster in the present model is reported as an empirical degeneracy until that stronger operator identification is made.

# Complex world-tube response: flux, poles, charge, and form factor

The proton certificate needs oriented number, mass, charge, and spatial response. These are not forced into one positive crossing sum. Each is the complex or integer object naturally supplied by the existing cobordism, operator pencil, and multiplicative connection.

## Cooriented cuts are supplied by the cobordism

Use the oriented boundary convention $\partial W=\overline M_0\sqcup M_1$. A separating slice is a cooriented codimension-one simplicial cut $\Sigma$ in a fixed relative homology class between the incoming and outgoing boundaries. Its coorientation is inherited from the cobordism; reversing the global cobordism orientation reverses every oriented charge together.

No Lorentzian distance, real projection, or level-set ordering is required. When a later real globally hyperbolic specialization exists, smooth temporal functions provide convenient representatives of the same separating class \[37\]; that theorem is not invoked for an arbitrary complex metric. Different homologous cuts must give the same conserved intersection and current flux whenever no source lies in the slab between them.

## Cluster lineage gives the oriented integer

Persistent cluster matching across frames produces an oriented tracking graph. A quark lineage is its integral one-chain $$c_Q\in C_1(W,\partial W;\mathbb Z).$$ For a transverse separating cut, define $$N_Q=c_Q\cdot\Sigma\in\mathbb Z .$$ This is the simplicial intersection pairing. Reversing the lineage sends $c_Q\mapsto-c_Q$ and $N_Q\mapsto-N_Q$; no sign is extracted from $z$, $U$, an eigenvalue, or a density. If the lineage has no source in the slab, homologous cuts give the same integer. Pair creation is represented by the boundary of an oriented pair surface and therefore creates $+1$ and $-1$ together.

For a collection of certified quark and anti-cluster lineages, $$N_q(\Sigma)=\sum_Q c_Q\cdot\Sigma,\qquad
  B(\Sigma)=\frac{N_q(\Sigma)}{3}.$$ The factor $1/3$ is one explicit physical calibration, not a topological theorem. On any tube for which the relative determinant family closes, the independent requirement is $\nu=N_Q$. A disagreement is a defect signal. Nothing here requires the cluster support to contain a hole: a path through a contractible region still intersects a separating cut.

## Mass is a complex bound-state pole

Mass is not defined by an incoherent sum of moduli. Let $F_C(s)$ be the exact meromorphic Feshbach response pencil of the entire persistent bound cluster $C$, continued in the complex spectral parameter $s$ on a domain that excludes unretained interior poles, and define $$D_C(s)=\det F_C(s).$$ A simple isolated zero is specified by $$D_C(s_C)=0,\qquad D_C'(s_C)\ne0.$$ Then $s_C$ is a pole of the supported resolvent $F_C(s)^{-1}$ and is the complex bound-state pole certificate. Multiple roots are retained with their algebraic multiplicity and local Smith/Jordan data rather than split by an ordering convention. The contour enclosing $s_C$, pole residue, separation, and refinement continuation are reported. Binding is measured by the composite pole and its response residue; it is not assumed additive over constituent poles.

Before spacetime translations emerge, $s_C$ is only a complex spectral pole. If a refinement regime supplies a nondegenerate complex Lorentzian momentum pairing, it may be identified with the invariant momentum square $p^2$ and hence with a mass-squared pole. The choice between conventions $p^2=m^2$ and $p^2=-m^2$ is a single declared global signature convention. The theory carries $s_C$ and does not take a square root; selecting a mass rather than mass squared would require a continuously tracked Riemann sheet. One physical scale calibration fixes units, after which pole ratios and binding shifts are predictions. Comparison with QCD mass decomposition is then a benchmark rather than an algebraic consequence \[38\].

## Orientation gives baryon number; electric charge needs flavor

The oriented lineage counts baryon number. It is not by itself electric charge: up and down clusters have the same lineage orientation and differ in flavor. The multiplicative connection supplies a separate complex Ward current. If the gauge-invariant stationary action is $\mathcal S(z,U,\Gamma)$, define on each oriented edge $$j_{xy}=U_{xy}\frac{\partial\mathcal S}{\partial U_{xy}} .$$ One orientation is chosen per geometric edge; the inverse-edge convention $U_{yx}=U_{xy}^{-1}$ gives $j_{yx}=-j_{xy}$. Under the infinitesimal complex gauge variation $\delta U_{xy}=(-\epsilon_x+\epsilon_y)U_{xy}$, invariance of the full action, including the transformed matter frames, gives the discrete Ward identity $$\partial j=0$$ in the bulk on the matter equations of motion. Thus $\Phi_j(\Sigma)=\langle j,\Sigma\rangle\in\mathbb C$ is unchanged between homologous cuts without a source. This construction uses $U\,\partial/\partial U$ and never separates a compact phase from a noncompact magnitude.

If an unlabeled isospin doublet emerges, its current $J_{I_3}$ and the lineage current $J_B=J_q/3$ define $$J_Q=J_{I_3}+\frac{1}{2}J_B,
  \qquad Q=I_3+\frac{B}{2}.$$ The integrated flavor-derived flux and the microscopic Ward-current flux must agree before either is called electromagnetic charge. Their normalization is fixed once on a boundary reference state; $Q_u=2/3$ and $Q_d=-1/3$ are then predictions of the emergent doublet identification, not hidden edge labels.

## Momentum, radius, and the form factor

Suppose first that a refinement regime supplies stable translation generators, a complex Lorentzian momentum pairing, and matched left/right bound states $\langle\Psi_L(p)\rvert,\lvert\Psi_R(p)\rangle$. For the charge-current insertion $J_Q$ of the preceding subsection, define the coherent form factor $$F_Q(p',p)=
  \frac{\langle\Psi_L(p')\rvert J_Q\lvert\Psi_R(p)\rangle}
       {\langle\Psi_L(p)\rvert J_Q\lvert\Psi_R(p)\rangle},
  \qquad q=p'-p,\qquad q^2=g(q,q)\in\mathbb C .$$ The denominator must be nonzero; otherwise the unnormalized matrix element is reported. Gauge and frame factors cancel between the matched bra, current, and ket. The complex squared-radius coefficient is $$R_Q^2=-6\,\frac{dF_Q}{d(q^2)}\bigg|_{q^2=0}\in\mathbb C .$$ No square root or real projection is taken. On a finite complex, the derivative is obtained from the local analytic response or a documented complex refinement continuation. A physical real charge radius is claimed only after a boundary $*$-structure and real Lorentzian momentum section make that interpretation available, as in continuum form-factor analyses \[39\].

If stable translations have not emerged, the paper does not relabel a slice Laplacian eigenvalue as momentum transfer. Instead let $\rho_R$ and $\widetilde\rho_L^{\mathsf T}$ be the right and left restrictions of the complex Ward current to $\Sigma$. The intrinsic spectral response is $$\mathcal R_Q(\lambda)=
  \widetilde\rho_L^{\mathsf T}
  (L_\Sigma^H-\lambda I)^{-1}\rho_R .$$ Its poles, residues, and analytic continuation are gauge- and basis-covariant complex data; it is not called an electromagnetic form factor. Degenerate bands are handled by Riesz contours rather than ordered eigenvectors.

Background removal is coherent. For every current or response, $$\Delta\mathcal O=\mathcal O_{\mathrm{state}}
                    -\mathcal O_{\mathrm{matched\ }M_0}$$ is formed before any boundary probability. Complex background and excitation terms may therefore cancel. The bound-state pole is likewise compared with the matched vacuum response, not converted into a sum of positive constituent crossings.

# The proton as the maximally informative baryon

The proton is chosen because, beyond generic baryon structure, it demands a nontrivial flavor pattern, electric charge, spin, and experimentally meaningful form factors.

Let three persistent quark components $A,B,C$ have color frames and projective color rays $[c_A]\in\mathbb P(E_A)$, $[c_B]\in\mathbb P(E_B)$, and $[c_C]\in\mathbb P(E_C)$. Choose a common base fiber $E_p$ in the bound supercluster, declared transport paths, and the representatives $c_A,c_B,c_C$ supplied by the many-body state. Set $$\widehat c_A=M_{pA}c_A,\qquad
  \widehat c_B=M_{pB}c_B,\qquad
  \widehat c_C=M_{pC}c_C .$$ Let $\Omega_p\in(\det E_p)^\vee$ be the dual determinant trivialization transported from the boundary reference. The common-frame color amplitude is $$S_{ABC}
  =\Omega_p\!\left(
     \widehat c_A\wedge\widehat c_B\wedge\widehat c_C
   \right)\in\mathbb C .$$ It is invariant under all local frame changes when $\Omega_p$ is transformed dually. Rescaling arbitrary representatives of the three rays rescales $S$; therefore its state-independent content is first the determinant ray and its vanishing/nonvanishing, while scalar amplitudes and ratios use the representatives fixed by the boundary/Fock state. Changing any path inserts the corresponding measured base-point holonomy on that transported vector; this dependence is therefore part of the certificate. The physical singlet condition is a nonzero, refinement-stable, covariantly trivial determinant wedge; no normalization to one or modulus square is imposed in the bulk. The proposed proton certificate is the conjunction:

- three persistent odd rank-three quark clusters with accepted projective triangle-anchor certificates, with no Betti-number or hole requirement;

- one persistent bound supercluster containing them;

- nonzero common-frame color wedge $S_{ABC}$, a coherent determinant trivialization, and projective holonomies compatible with a singlet;

- flavor spectrum with the $uud$ occupation pattern, in the sense of the still-hypothetical isospin-doublet construction of Section <a href="#sec:quarks" data-reference-type="ref" data-reference="sec:quarks">10</a>;

- oriented lineage $N_q=3$ and $B=1$, with relative determinant winding agreeing wherever its interferometric closure exists;

- complex Ward-current and flavor-current fluxes agreeing at $Q=+1$;

- a sharp total-space spin readout: $$(J^2-\tfrac34 I)\lvert\Psi_R\rangle=0,\qquad
        \langle\Psi_L\rvert(J^2-\tfrac34 I)=0,$$ together with reference-normalized complex holonomy $\chi_F(2\pi)=-1$ and, where applicable, an accepted spin lift;

- a stable bound-state pole $s_p$, coherent charge form factor and complex squared-radius coefficient when momentum exists, or the intrinsic spectral response when it does not — each compared coherently with $M_0$; and

- stability of every dimensionless certificate under refinement.

The two eigen-equations are essential. In a complex bilinear theory, $\langle J^2\rangle=3/4$ is only a matrix-element identity, and even a vanishing complex variance can result from isotropic cancellation without a sharp eigenstate. The right and left residual vectors must vanish algebraically; their coordinate norms are reported only as numerical certificates. Because $J^2$ is polynomial in the exterior generators, its action on a Slater pair is computed without constructing the full Fock matrix, using biorthogonal Wick reduction. A quasi-free state can in special cases satisfy the exact eigen-equations, so failure is not automatic. Failure across the entire accepted covariance-only class is the structural branch point of Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>.

None of these conditions should be included as an emergence target. The proton is found only if the base geometric stationary search produces a component satisfying them. Targeted runs remain valuable as existence and obstruction experiments, but must be labeled as synthesis rather than emergence.

<figure id="fig:firewall">

<figcaption>No-feedback emergence protocol. Only the base complex stationarity equations drive the geometric search. In certificates-blind backreaction mode the carried state’s bilinear action density may enter those equations; Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a> records that this remains inside the quasi-free class. Cluster, fiber, color, exchange, and baryon observables are computed from accepted snapshots and cannot feed back into the stationary equations in either emergence mode. Targeted synthesis is a separate, explicitly labeled mode.</figcaption>
</figure>

# The master recursive construction

Let $\mathcal{R}_0(\lambda)=h_1(z,U)-\lambda I$ denote the microscopic edge-mode response pencil. At every scale: $$\boxed{
\begin{aligned}
  P_{\ell} &= \mathrm{PersistentPartition}(\mathcal{R}_{\ell}),\\
  \Pi_v^{\ell+1}
    &=\frac{1}{2\pi i}\oint_{\Gamma_v}
      (\zeta I-h_v^\ell)^{-1}\,d\zeta,\qquad
  E_v^{\ell+1}=\operatorname{Ran}\Pi_v^{\ell+1},\\
  \mathcal{R}_{\ell+1}(\lambda) &= \mathrm{Feshbach}_{P_{\ell}}(\mathcal{R}_{\ell}(\lambda)),\\
  M_{vw}^{\ell+1}
    &=\widetilde\Phi_v^{\ell+1\,\mathsf T}
      T_{vw}^{\ell}\Phi_w^{\ell+1}
      \in\operatorname{Hom}(E_w^{\ell+1},E_v^{\ell+1}),\\
  \mathfrak{h}_{\ell+1} &= \mathbin{\boxplus}_{v}E_{v}^{\ell+1},\qquad
    J_{\ell+1}:\mathfrak{h}_{\ell+1}\to C_1(K),\qquad
    G_{\ell+1}
      =\widetilde J_{\ell+1}^{\mathsf T}J_{\ell+1},\\
  \mathcal{H}_{\ell+1} &= \mathcal{F}_{-}\bigl(\mathfrak{h}_{\ell+1}\bigr),
\end{aligned}}$$ where $\mathbin{\boxplus}$ is the abstract labeled sum: one summand per retained fiber, with no claim that the geometric subspaces are independent inside $C(K)$.

The geometric subspaces $E_{v}\subset C(K)$ of adjacent components may overlap on shared interface cells, so their internal sum need not be direct. The recursion therefore never asserts $\bigoplus_{v}E_{v}\subset C(K)$. It forms the abstract labeled sum, carries the embedding $J_{\ell+1}$, its algebraic dual, and overlap matrix $G_{\ell+1}$ exactly, and proceeds by exactly one of three declared options: carry $G$ in every subsequent formula; certify $\lVert G-I\rVert\le\varepsilon$ and propagate $\varepsilon$ through the composable amplitude budget of Section <a href="#sec:interactions" data-reference-type="ref" data-reference="sec:interactions">6</a>; or quotient $G$’s left and right radicals through a rank-revealing factorization and restate the fiber ranks. A sheaf-stalk decomposition that assigns interface modes to link stalks is a valid realization of the same requirement \[10\], but it is not necessary.

At $\lambda=0$ the response step is exact supported block elimination. For a nonzero complex band it is the exact energy-dependent pencil; a linear $\mathcal{R}_{\ell+1}$ is an AMLS/component-mode surrogate with a declared complex spectral region and residual. The transport rank is generic; only a projectively anchored accepted rank-three fiber receives the color interpretation, and only common-rank invertible links enter a $GL(r,\mathbb C)$ holonomy. No polar projection, eigenvalue ordering, or square-root branch occurs in the recursion. It supplies the response network, retained stalk, derived transport, and expanding state space without claiming that every coarse level is literally a new simplicial complex.

# Prior art and boundary of novelty

No single cited work establishes the full recursive spectral-fiber proposal. The construction is a synthesis of several mature ideas, and its novelty should be evaluated at the joins. Table <a href="#tab:priorart" data-reference-type="ref" data-reference="tab:priorart">3</a> states the boundary explicitly.

<div id="tab:priorart">

| Topic                                 | Established prior art                                                                                                                                                                                                            | Additional claim made here                                                                                                                                                                                                                                                                                                   |
|:--------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Topic                                 | Established prior art                                                                                                                                                                                                            | Additional claim made here                                                                                                                                                                                                                                                                                                   |
| continued on next page                |                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                              |
| Simplicial geometry and Hodge spectra | Regge curvature, discrete exterior calculus, and combinatorial Laplace spectra \[2, 3, 4\].                                                                                                                                      | Use one jointly stationary complex Regge–Hodge carrier for geometry and quantum readouts; keep incidence homology separate from kernels of complex-bilinear Hodge operators.                                                                                                                                                 |
| Coarse response and recursive modules | Kron/Schur reduction, Feshbach maps, component-mode synthesis/AMLS, cellular sheaf Laplacians, modular communities, and self-similar network renormalization, and Riesz spectral projections \[5, 6, 7, 8, 10, 11, 12, 13, 40\]. | Treat a persistent component as a static response vertex; retain nonzero bands through shifted or certified component-mode reduction; recurse in operator-valued response networks; and select complex bands by contours rather than by ordering a real projection.                                                          |
| Quasi-free many-body calculus         | Second quantization, generalized Hartree–Fock theory, Gaussian states, and nonorthogonal Wick reduction \[14, 18, 41\].                                                                                                          | Carry a complex matched Slater pair by the idempotent biorthogonal covariance $\Gamma=\Phi\widetilde\Phi^{\mathsf T}$; prove arbitrary complex quadratic and mean-field evolution is Gaussian-closed; require a separate $*$-certificate only when probabilities are claimed.                                                |
| Geometric gauge transport             | Berry and Wilczek–Zee holonomy, overlap-based lattice links, magnetic and connection Laplacians/vector diffusion, Kato transport, and Wilson loops \[23, 24, 25, 26, 33, 34, 35, 40\].                                           | Derive $GL(r,\mathbb C)$ transport from dual Riesz frames; at anchored rank three retain the determinant line and $PGL(3,\mathbb C)$ class without polar, root, or logarithm choices; treat $SU(3)$ as a certified compact real form.                                                                                        |
| Color and fermion structure           | Quark/color triplets, exterior Fock/second quantization, topological exchange phases, and spin structures \[20, 21, 22, 14, 28, 29, 27, 19\].                                                                                    | Realize $\mathbb C\oplus E\oplus(\det E\otimes E^\vee)\oplus\det E$ on three oriented edge modes, anchor abstract rank-three fibers projectively to oriented faces, model anti-clusters by dual/reversed lineages, and test exchange by a reference-cancelled determinant interferometer plus structural permutation parity. |
| Complex Lorentzian readouts           | Complexified spacetime metrics, relative homology/intersection, Ward currents, resonance poles, and coherent form factors \[42, 37, 38, 39\].                                                                                    | Use cobordism coorientation rather than the sign of a complex scalar; define baryon number by cluster-lineage intersection, mass by a complex composite pole, charge by a multiplicative-link Ward current, and radius by a coherent left/right form-factor derivative.                                                      |
| Scale composition and boundaries      | TQFT cobordisms, general-boundary state assignments, categorical tensor composition, second quantization, and entanglement renormalization \[15, 16, 17, 14, 30\].                                                               | Keep simplicial gluing at the one-particle level, then build finite Fock stages functorially and require vacuum-embedding compatibility under refinement.                                                                                                                                                                    |
| Kähler–Dirac boundary                 | Differential-form fermions and their taste structure \[31, 32\].                                                                                                                                                                 | Do not infer Kähler–Dirac tastes from occupation exterior algebra; test for them only if the one-particle field is promoted to inhomogeneous cochains with a Kähler–Dirac operator.                                                                                                                                          |
| Spectral spacetime                    | Diffusion spectral dimension on ensembles of simplicial geometries \[1\].                                                                                                                                                        | Test whether many interacting Tessera cobordisms yield a stable four-dimensional spectral window while simultaneously supporting the particle certificates.                                                                                                                                                                  |

Established ingredients and the additional Tessera claim.

</div>

<figure id="fig:priorart">

<figcaption>Relationship to prior art. The blue inputs are established research programs; the green center is the proposed Tessera synthesis; the amber outputs are new physical identifications and must be validated independently. An arrow denotes conceptual inheritance, not a proof of the downstream claim.</figcaption>
</figure>

# Falsification program

The formulation fails, or must be narrowed, if any of the following persists under refinement and tighter numerical certification:

1.  **No persistent rank-three clusters.** Certified components appear, but their Riesz-fiber rank, contour separation, or resolvent certificate is unstable.

2.  **No oriented color anchor.** A rank-three band appears, but its projective Plücker profile, invariant complex anchor coordinates, or overlap transition functions degenerate or drift.

3.  **No faithful coarse response.** Schur-reduced components fail to reproduce static response, or shifted/AMLS reduction fails over its declared complex frequency region, within the stated residual.

4.  **No derived gauge covariance.** The characteristic data of $GL(r,\mathbb C)$ holonomies depend on local Riesz frames after leakage is controlled, or determinant and $PGL(3,\mathbb C)$ transport cannot be made path-consistent.

5.  **No fermion holonomy.** The reference-cancelled complex determinant character or structural permutation parity does not give $-1$, or the verdict changes under relabeling.

6.  **No spinor rotation.** Exchange works but the reference-normalized $2\pi$ physical rotation does not give $-1$; in a manifold-like continuum claim, failure of a consistent spin lift is also decisive.

7.  **No inductive compatibility.** Adding vacuum modes changes already-computed amplitudes by a nonvanishing amount.

8.  **No quasi-free proton.** Every other certificate is met inside the covariance-only theory, but the left or right $(J^2-\tfrac34 I)$ residual fails to converge to zero on every accepted candidate across refinement. This outcome is a branch point rather than a refutation of the geometry: it mandates adopting exactly one of the non-Gaussian mechanisms of Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>, as an explicit scope decision, before any proton claim is made.

9.  **No unforced baryon.** Targeted synthesis can build the certificates, but the stationary geometric ensemble never produces them without a proton-specific term.

10. **No oriented-number agreement.** Cluster-lineage intersection is not cut-invariant, or it disagrees with closed relative determinant winding where both are defined.

11. **No bound-state/current response.** The three-cluster composite has no stable isolated complex pole, or its Ward-current and flavor-derived charge fluxes fail to agree.

12. **Clusters do not suffice.** After exhaustive refinement, accepted color/register fibers occur only when supported by nontrivial homology and never on contractible certified clusters, however those supports were proposed. A persistent conditional association must survive controls for cluster size, geometry, detector choice, and spectral gap before it counts. That result falsifies the paper’s no-hole sufficiency hypothesis, even though it may motivate a different topology-dependent model.

13. **No continuum stability.** Dimensionless color, parity, charge, spin, and amplitude certificates drift rather than converge with refinement.

14. **Unexpected multiplicity.** A robust flavor/taste degeneracy is neither predicted by the stated one-particle operator nor stable enough to be promoted to an emergent flavor mechanism.

Holes may re-emerge and may correlate with some phases, but they are neither an acceptance condition nor a hidden carrier. Their necessity is an experimental outcome to be tested, not assumed.

# Conclusion

The geometry is economical and complex-first. An edge carries $z_e$, $U_e$, and a two-level mode algebra, not an independently stored pure state. A quark candidate is a persistent cluster whose rank-three Riesz fiber is anchored projectively to oriented faces; no homology class or hole is required. Its transport is the full certified $GL(3,\mathbb C)$ overlap with determinant and $PGL(3,\mathbb C)$ data retained. An anti-cluster is the dual fiber on the reversed lineage. Fermion parity is the exterior grading and its geometric test is a reference-cancelled complex determinant, with no phase-angle extraction.

Simplicial gluing constructs the one-particle operator and second quantization constructs the expanding Fock state. A matched Slater pair evolves by the idempotent biorthogonal covariance without positivity or pseudo-Hermiticity. Three accepted clusters form a baryon only after transport to a common color fiber, where their wedge pairs nontrivially with a coherent dual determinant trivialization. The proton is the sharpest test because it also demands $N_q=3$, agreement of lineage and determinant winding, the emergent $uud$ doublet and charge current, a complex bound-state pole, and exact left/right spin-$1/2$ eigen-equations.

The claims close in the four tiers of Section <a href="#sec:epistemic" data-reference-type="ref" data-reference="sec:epistemic">1</a>. The exact identities are limited to their proper domains: static Schur response, energy-dependent Feshbach isospectrality, exterior/CAR algebra, second-quantized direct-sum composition, and gauge covariance of accepted transport. The conditional theorems carry their certificates with them: closure of the quasi-free class holds under every generator the model currently possesses. The biorthogonal commutator law $i\dot\Gamma=[h,\Gamma]$ is exact for arbitrary complex $h$, and every polynomial left/right matrix element is a finite Wick reduction. A $*$-structure is needed only for probabilities and compact real forms, and is never inferred by taking a real part. The remaining readouts — projective anchors, determinant winding, lineage intersection, Ward-current flux, rotation character, spin lift, complex bound-state pole, and coherent form factor — are proposed physical identifications. Their finite-stage computations may be exact while their physical interpretation remains conditional on calibration, boundary structure, and refinement stability.

What remains genuinely open is whether Tessera’s unforced Regge–Hodge dynamics produces the required anchored clusters on contractible as well as nontrivial supports, low-leakage holonomies, matching lineage and determinant integers, stable bound-state poles, conserved current response, and sharp spin. A tempting Hellmann–Feynman/envelope argument does not by itself make the first variation of transport overlap defect vanish at a Regge–Hodge stationary point, because the defect is not the stationary functional; the programme therefore measures that correlation rather than citing stationarity as a theorem. The decisive question is the dichotomy of Section <a href="#sec:quasifree" data-reference-type="ref" data-reference="sec:quasifree">7</a>: either an exact covariance-only proton exists, or a genuinely non-Gaussian, geometry-mediated interaction is required. Either outcome is a result. The first makes the particle layer exactly and polynomially certifiable; the second would be the first internal evidence that the geometry must supply a true interaction term. The most parsimonious named candidate is the exact elimination of a certified quadratic geometric block, which produces the factored quartic kernel $-\tfrac12J^{\mathsf T}A^{-1}J$ without adding a microscopic particle field.

<div class="thebibliography">

42

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

\[37\] Antonio N. Bernal and Miguel Sánchez. Smoothness of time functions and the metric splitting of globally hyperbolic spacetimes. *Communications in Mathematical Physics*, 257:43–50, 2005. doi:10.1007/s00220-005-1346-1. URL <https://arxiv.org/abs/gr-qc/0401112>.

\[38\] Yi-Bo Yang, Jian Liang, Yu-Jiang Bi, Ying Chen, Terrence Draper, Keh-Fei Liu, and Zhaofeng Liu. Proton mass decomposition from the QCD energy momentum tensor. *Physical Review Letters*, 121:212001, 2018. doi:10.1103/PhysRevLett.121.212001. URL <https://arxiv.org/abs/1808.08677>.

\[39\] Alexander V. Gramolin and Rebecca L. Russell. Transverse charge density and the radius of the proton. *Physical Review D*, 105:054004, 2022. doi:10.1103/PhysRevD.105.054004. URL <https://arxiv.org/abs/2102.13022>.

\[40\] Tosio Kato. *Perturbation Theory for Linear Operators*. Classics in Mathematics, second edition. Springer, Berlin and Heidelberg, 1995. doi:10.1007/978-3-642-66282-9.

\[41\] Hugh G. A. Burton. Generalised nonorthogonal matrix elements: Unifying Wick’s theorem and the Slater–Condon rules. *The Journal of Chemical Physics*, 154:144109, 2021. doi:10.1063/5.0045442. URL <https://arxiv.org/abs/2101.10944>.

\[42\] Matt Visser. Feynman’s $i\epsilon$ prescription, almost real spacetimes, and acceptable complex spacetimes. *Journal of High Energy Physics*, 2022:129, 2022. doi:10.1007/JHEP08(2022)129. URL <https://arxiv.org/abs/2111.14016>.

</div>
