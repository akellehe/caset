#### Vendored edition.

This is the tracked edition of the specification in `docs/design`. The chain-Hodge specification (CH) it amends is not tracked in the repository, so the amendments of §<a href="#sec:amend" data-reference-type="ref" data-reference="sec:amend">12</a> to CH are recorded here as well: default preset `L2`; presets `TOP` and `VOL` removed; `GRASSMANN_ALL` retained with the SVP deviation note; CH App. B superseded by §<a href="#sec:h1" data-reference-type="ref" data-reference="sec:h1">5</a>; CH §6.3 phase dressing retained as a geometric deformation only; CH App. A branch rule applies per top simplex; the auxiliary solve of §<a href="#sec:aux" data-reference-type="ref" data-reference="sec:aux">4.3</a> is the implementation path; the Lorentzian protocol requirement (Req. <a href="#req:lorentz" data-reference-type="ref" data-reference="req:lorentz">2</a>) added. The Python oracle the verification suite cites stays on the repository’s `issue-attachments` release; its values are pinned by the C++ tests.

> The rendered vector diagrams are preserved in the LaTeX/PDF edition; this Markdown edition is the searchable text companion.

# Status and changes from revision 1

Revision 1 instantiated the whitepaper *Recursive Spectral Fibers on Simplicial Cobordisms* (RSF) on the chain-Hodge specification (CH) with $W_k:=G_k=M_k\circ\Gamma_k$ (Grassmann projections of simplex blades). The scaling verification plan (SVP) then tested harmonic representatives on jittered flat and curved geometries against the known continuum answer and found the Grassmann representatives off by $5$–$10^{\circ}$ (Euclidean) and $63$–$88^{\circ}$ (Lorentzian) with no improvement under refinement, while the Whitney representatives were exact on flat meshes and second-order convergent on curved ones. The mechanism is dimensional: a blade pairing has degree $2k$ in length, the $L^{2}$ pairing degree $2k-d$; the ratio is a local $d$-volume that is constant on regular meshes and varies from cell to cell on irregular ones, so the Grassmann representative encodes the triangulation. Changes:

1.  Default metric: Whitney, $W_k:=G_k=M_k^{-1}$ on chains, with $M_k$ explicit and sparse (§<a href="#sec:metric" data-reference-type="ref" data-reference="sec:metric">4</a>).

2.  Chain formulation throughout. The geometric image $z=G_1h$ of CH Def. 7.3 appears as the readout and as an intermediate solve (§<a href="#sec:aux" data-reference-type="ref" data-reference="sec:aux">4.3</a>); on a boundary circle it is the vector of signed lengths. No second kind of object is introduced.

3.  Grassmann (`GRASSMANN_ALL`) is retained as a named option with its deviation documented; `TOP` and `VOL` are removed.

4.  Lorentzian protocol: compute on the Kontsevich–Segal allowable domain at reported $\varepsilon>0$ (§<a href="#sec:lorentz" data-reference-type="ref" data-reference="sec:lorentz">10</a>).

5.  The face-anchor and overlap-locality statements of revision 1 are reversed (§<a href="#sec:anchor" data-reference-type="ref" data-reference="sec:anchor">8</a>, §<a href="#sec:recursion" data-reference-type="ref" data-reference="sec:recursion">7</a>).

# Scope, tiers, and notation

Statements are labeled as in RSF: exact identity, conditional theorem with certificate, numerical evidence, proposed identification. Everything in §<a href="#sec:adjoint" data-reference-type="ref" data-reference="sec:adjoint">3</a>–§<a href="#sec:recursion" data-reference-type="ref" data-reference="sec:recursion">7</a> is exact or conditional; §<a href="#sec:lorentz" data-reference-type="ref" data-reference="sec:lorentz">10</a> contains labeled evidence.

<div class="center">

| Object               | RSF          | CH               | This document                                                                |
|:---------------------|:-------------|:-----------------|:-----------------------------------------------------------------------------|
| squared length       | $z_e$        | $s_e$            | $s_e$                                                                        |
| inverse chain metric | —            | $M_k$ (Option A) | $M_k$; dressed $M_k^{U}$; explicit and sparse                                |
| chain metric         | $W_k$        | $G_k$            | $G_k=M_k^{-1}$, applied by solves, never formed                              |
| Hodge operator       | $L_k^{H}$    | $L_k$            | $L_k$                                                                        |
| covariant operator   | $h_k(z,U)$   | —                | $h_k(s,U)$                                                                   |
| geometric image      | —            | $G_kh$           | $z=G_kh$; intermediate solve $M_kz=h$                                        |
| symmetric pencil     | —            | $A_k=G_kL_k$     | $A_k^{U}$; auxiliary sparse form $\widetilde{A}_k^{U}=M_k^{U}A_k^{U}M_k^{U}$ |
| gauge representation | $G_k(g)$     | —                | $\rho_k(g)=\operatorname{diag}(g_{b(\sigma)}^{-1})$                          |
| fiber transfer       | $M_{AB}$     | —                | $\mathsf{T}_{AB}$ ($M$ is reserved for mass matrices)                        |
| coarse overlap       | $G_{\ell+1}$ | —                | $\mathcal G_{\ell+1}$                                                        |

</div>

Reference orientation is ascending vertex id; $b(\sigma)=\min\sigma$. $A^{T}$ is the transpose; $A^{\dagger}$ appears only in certificates. $n_k=|K_k|$, $b_k$ the Betti numbers over $\mathbb Q$, $\epsilon_{\rm m}=2^{-52}$.

# The adjoint is the transpose

The incidence matrices are integer, so $\partial_k^{\dagger}=\partial_k^{T}$; “conjugate-transpose adjoint” means only that the metric is Hermitian. $M_k$ is $\sqrt{\det g_T}$ times a rational function of the $s_e$, hence complex symmetric; a Hermitian complex symmetric matrix is real, so the dagger would force $\operatorname{Re}M_k$ or $|M_k|$ and remove $\operatorname{Im}s_e$ from the operator. For the connection, $(\partial^{U})^{T}$ carries $U$ and $(\partial^{U})^{\dagger}$ carries $\bar U$; with $U_{yx}=U_{xy}^{-1}$ and $g\in\mathbb{C}^{*}$ only the transpose is covariant. The transpose is the adjoint of a symmetric bilinear form ($O(n,\mathbb{C})$); complexifying a real metric of any signature produces that, never a Hermitian form. Conjugation enters legitimately in the antilinear $*$-structure that defines probabilities (RSF §1, §7) and in numerical certificates. Costs: complex spectra, biorthogonal eigenvectors, possible Jordan blocks, $\ker L_k\ne H_k$ unless the rank conditions hold.

# The Whitney metric

## The inverse chain metric $M_k$

For a top simplex $T=[v_0<\dots<v_d]$ with Gram matrix $(g_T)_{ij}=\tfrac12(s_{v_0v_i}+s_{v_0v_j}-s_{v_iv_j})$, $i,j=1..d$, define $$|T|=\frac{\sqrt{\det g_T}}{d!},\qquad
\Gamma_{ij}=(g_T^{-1})_{ij}\ (i,j\ge1),\qquad \Gamma_{0j}=-\sum_{i\ge1}\Gamma_{ij},\qquad \Gamma_{00}=\sum_{i,j\ge1}\Gamma_{ij},$$ $$\int_T\lambda_a\lambda_b\,d\mathrm{vol}=|T|\,\frac{1+\delta_{ab}}{(d+1)(d+2)} .$$ $\Gamma_{ab}=\langle d\lambda_a,\, d\lambda_b\rangle_{g_T}$ is the pairing of the gradients of the barycentric coordinates; since $\nabla\lambda_a$ is the inward normal of the facet $\tau_a$ opposite $v_a$ with $|\nabla\lambda_a|=|\tau_a|/(d|T|)$, it equals $\pm\langle \vec{\tau}_a,\, \vec{\tau}_b\rangle/(d|T|)^{2}$: the projection pairing of facet volumes divided by the cell volume squared. With Whitney forms $w_v=\lambda_v$, $w_{[v_iv_j]}=\lambda_id\lambda_j-\lambda_jd\lambda_i$, $w_{[v_iv_jv_k]}=2(\lambda_id\lambda_j\wedge d\lambda_k+\lambda_jd\lambda_k\wedge d\lambda_i+\lambda_kd\lambda_i\wedge d\lambda_j)$, the mass matrices are $$\begin{aligned}
(M_0)_{vv'}&=\sum_{T\ni v,v'}|T|\,\frac{1+\delta_{vv'}}{(d+1)(d+2)},\\
(M_1)_{ee'}&=\sum_{T\supset e,e'}\frac{|T|}{(d+1)(d+2)}\Big[(1+\delta_{ik})\Gamma_{jl}-(1+\delta_{il})\Gamma_{jk}-(1+\delta_{jk})\Gamma_{il}+(1+\delta_{jl})\Gamma_{ik}\Big],\\
(M_2)_{tt}&=1/|t|\qquad(d=2),
\end{aligned}$$ with $e=[v_iv_j]$, $e'=[v_kv_l]$ in the local vertex numbering of $T$. For $d\ge3$, $M_2$ is obtained by expanding $w_t\cdot w_{t'}$ with $\langle d\lambda_a\wedge d\lambda_b,\, d\lambda_c\wedge d\lambda_e\rangle=\Gamma_{ac}\Gamma_{be}-\Gamma_{ae}\Gamma_{bc}$ and the same $\lambda$-integral. Each $M_k$ is complex symmetric and sparse: $(M_k)_{\sigma\tau}\neq0$ only if $\sigma\cup\tau$ lies in a top simplex. The chain metric is $G_k:=M_k^{-1}$, i.e. $\langle c,\, c'\rangle_k=c^{T}M_k^{-1}c'$, evaluated by solving $M_ku=c'$.

<div class="remark">

*Remark 1* (What the formula is). The polynomials $w_\sigma$ above are the piecewise-affine interpolants of the simplices in barycentric coordinates; $M_k$ is their Gram matrix under the piecewise-flat metric. They are a construction rule internal to the definition of $M_k$, fixed by the complex, and never stored, evolved, or supplied. What the rule buys is the identity $d\,W=W\,\partial^{T}$ (tangential continuity across facets), which makes constant forms on flat meshes exactly harmonic and produces the zeros in the geometric-fidelity tests. Both $|T|$ and the facet volumes inside $\Gamma$ are norms of wedge products of edge vectors; the rule supplies the coefficients $(1+\delta_{ab})/((d+1)(d+2))$ and the division by $|T|^{2}$. The inputs remain the squared lengths alone.

</div>

## Volume element and branch

The only root is $\sqrt{\det g_T}$, one per top simplex. For real Lorentzian data every top simplex has $\det g_T<0$, so $\sqrt{\det g_T}=i\sqrt{|\det g_T|}$ uniformly: a global factor $i$ on every $M_k$ that cancels in harmonic spaces, eigenvalues of the pencil, projectors, frames normalized by their own pairing, and transfers up to a common factor. It does not cancel in $\det(\Phi^{T}M_1\Phi)$, where it contributes $i^{r}$ (§<a href="#sec:spectra" data-reference-type="ref" data-reference="sec:spectra">9</a>). For complex data the branch is fixed by continuation from a Euclidean reference or by the Kontsevich–Segal rule (CH App. A) per top simplex. Relative branch ambiguities arise only for lower faces in $d\ge3$, which $M_k$ never uses.

<div class="requirement">

**Requirement 1** (Instance certificate). Every instance reports whether all its top simplices are Kontsevich–Segal allowable, the minimal margin $\pi-\sum_i|\arg\lambda_i(g_T)|$ over $T$, and, for Lorentzian instances, the rotation $\varepsilon$ of §<a href="#sec:lorentz" data-reference-type="ref" data-reference="sec:lorentz">10</a>.

</div>

## Chain operators and the auxiliary solve

With $G_k=M_k^{-1}$ the operators of CH §7 are $$\partial_1^{*}=M_1\partial_1^{T}M_0^{-1},\qquad \partial_2^{*}=M_2\partial_2^{T}M_1^{-1},\qquad
L_1=\partial_1^{*}\partial_1+\partial_2\partial_2^{*}=M_1\partial_1^{T}M_0^{-1}\partial_1+\partial_2M_2\partial_2^{T}M_1^{-1},$$ $$H_1=\{h:\ \partial_1h=0,\ \partial_2^{T}M_1^{-1}h=0\},\qquad A_1=G_1L_1=\partial_1^{T}M_0^{-1}\partial_1+M_1^{-1}\partial_2M_2\partial_2^{T}M_1^{-1}.$$

<div id="prop:aux" class="proposition">

**Proposition 1** (Auxiliary solve). *Let $z=G_1h$ be the geometric image of a chain $h$, so $h=M_1z$. Then $$h\in H_1\iff \partial_2^{T}z=0\ \text{ and }\ \partial_1M_1z=0,\qquad
A_1x=\lambda G_1x\iff \widetilde{A}_1z=\lambda M_1z\ \text{ with } z=G_1x,$$ $$\widetilde{A}_1=M_1A_1M_1=M_1\partial_1^{T}M_0^{-1}\partial_1M_1+\partial_2M_2\partial_2^{T}=\widetilde{A}_1^{T}.$$*

</div>

<div class="proof">

*Proof.* Substitute $h=M_1z$ into the definitions and multiply the pencil by $M_1$. ◻

</div>

$z$ is a function of the lengths and the chain, not a new variable: it is the readout that on a boundary circle equals the vector of signed lengths (§<a href="#sec:spectra" data-reference-type="ref" data-reference="sec:spectra">9</a>). Computationally it is the variable to solve for, because $M_1$ is sparse and $G_1$ is dense: $H_1=M_1\ker S$ with the sparse stacked matrix $S=\begin{pmatrix}\partial_2^{T}\\ \partial_1M_1\end{pmatrix}$, and only $M_0^{-1}$ is applied, by sparse factorization.

<div id="prop:rank" class="proposition">

**Proposition 2** (Rank conditions). *Let $Z$ solve $M_1Z=\partial_2$. The four conditions of CH Prop. 7.1 read $$\begin{aligned}
\text{(R1)}\ &\operatorname{rank}(\partial_2^{T}Z)=\operatorname{rank}\partial_2, &
\text{(R2)}\ &\operatorname{rank}(\partial_1M_1\partial_1^{T})=\operatorname{rank}\partial_1,\\
\text{(R3)}\ &\operatorname{rank}(\partial_1^{T}M_0^{-1}\partial_1)=\operatorname{rank}\partial_1, &
\text{(R4)}\ &\operatorname{rank}(\partial_2M_2\partial_2^{T})=\operatorname{rank}\partial_2 .
\end{aligned}$$ Under (R1)–(R2), $C_1=\operatorname{im}\partial_1^{*}\oplus\operatorname{im}\partial_2\oplus H_1$ and $\dim H_1=b_1$; under (R1)–(R4), $\ker L_1=H_1$ with no Jordan block at $0$, and the $\lambda=0$ Riesz projector is the projector onto $H_1$. For Euclidean data $M_k$ is positive definite and the conditions are automatic; for complex data they fail on a complex-codimension-one set; for real Lorentzian data they fail on a real-codimension-one set that refinement approaches (§<a href="#sec:lorentz" data-reference-type="ref" data-reference="sec:lorentz">10</a>).*

</div>

## The Grassmann option

`GRASSMANN_ALL` (CH §6) remains available: polynomial in $s$, branch-free, positive definite for Euclidean data, structurally nondegenerate. Its harmonic representatives depend on the triangulation at $O(1)$ (SVP Table 1) and its per-face block has rank two. It is not to be used where geometric fidelity or the face anchor of §<a href="#sec:anchor" data-reference-type="ref" data-reference="sec:anchor">8</a> is required.

# The covariant one-particle operator

<div id="def:twist" class="definition">

**Definition 1** (Twisted incidence, dressed mass). With $U_{xy}\in\mathbb{C}^{*}$, $U_{yx}=U_{xy}^{-1}$, $U_{xx}=1$, and $\rho_k(g)=\operatorname{diag}(g_{b(\sigma)}^{-1})_{\sigma\in K_k}$, $$(\partial_k^{U})_{\tau\sigma}=(\partial_k)_{\tau\sigma}\,U_{b(\tau)b(\sigma)},\qquad
(M_k^{U})_{\sigma\tau}=(M_k)_{\sigma\tau}\,U_{b(\sigma)b(\tau)} .$$

</div>

Both factors are single connection variables (the two base vertices lie in a common top simplex), so no path is chosen. The dressed chain metric is $G_k^{U}:=(M_k^{U})^{-1}$; dressing the sparse $M_k$ rather than the dense $G_k$ is what makes the construction implementable, and the two agree because $(M^{U})^{-1}$ transforms by the same similarity.

<div id="def:h" class="definition">

**Definition 2** (Covariant operator and pencil). $$\begin{aligned}
h_1(s,U)&=(G_1^{U})^{-1}(\partial_1^{U^{-1}})^{T}G_0^{U}\partial_1^{U}+\partial_2^{U}(G_2^{U})^{-1}(\partial_2^{U^{-1}})^{T}G_1^{U}\\
&=M_1^{U}(\partial_1^{U^{-1}})^{T}(M_0^{U})^{-1}\partial_1^{U}+\partial_2^{U}M_2^{U}(\partial_2^{U^{-1}})^{T}(M_1^{U})^{-1},
\end{aligned}$$ $$A_1^{U}=G_1^{U}h_1(s,U),\qquad
\widetilde{A}_1^{U}=M_1^{U}A_1^{U}M_1^{U}=M_1^{U}(\partial_1^{U^{-1}})^{T}(M_0^{U})^{-1}\partial_1^{U}M_1^{U}+\partial_2^{U}M_2^{U}(\partial_2^{U^{-1}})^{T}.$$ $A_1^{U}x=\lambda G_1^{U}x$ is solved as $\widetilde{A}_1^{U}z=\lambda M_1^{U}z$, $z=G_1^{U}x$.

</div>

<div id="prop:h" class="proposition">

**Proposition 3** (Exact properties).

1.  *$h_1(s,1)=L_1$ of §<a href="#sec:aux" data-reference-type="ref" data-reference="sec:aux">4.3</a>.*

2.  *$(M_k^{U})^{T}=M_k^{U^{-1}}$, $(\widetilde{A}_1^{U})^{T}=\widetilde{A}_1^{U^{-1}}$, and $h_1(s,U)^{T}=G_1^{U^{-1}}h_1(s,U^{-1})(G_1^{U^{-1}})^{-1}$.*

3.  *$h_1(s,U^{g})=\rho_1(g)h_1(s,U)\rho_1(g)^{-1}$, $\widetilde{A}_1^{U^g}=\rho_1(g)\widetilde{A}_1^{U}\rho_1(g)^{-1}$, $M_1^{U^g}=\rho_1(g)M_1^{U}\rho_1(g)^{-1}$.*

4.  *No flatness is required: for $t=[p<q<r]$, $\partial_1^{U}\partial_2^{U}t=U_{rp}(\mathcal F_t-1)[r]$, $\mathcal F_t=U_{rq}U_{qp}U_{pr}$.*

5.  *Pure gauge $U=1^{g}$ is isospectral to $L_1$.*

6.  *$\widetilde c^{T}G_k^{U}c$ is invariant under $\widetilde c\mapsto\rho_k^{-1}\widetilde c$, $c\mapsto\rho_kc$.*

</div>

<div class="proof">

*Proof.* (i) set $U=1$. (ii) transpose termwise using $(M^{U})_{\tau\sigma}=(M)_{\sigma\tau}U_{b(\sigma)b(\tau)}^{-1}$. (iii) entrywise from Def. <a href="#def:twist" data-reference-type="ref" data-reference="def:twist">1</a>, $\partial_k^{U^g}=\rho_{k-1}\partial_k^{U}\rho_k^{-1}$, $(U^{g})^{-1}=(U^{-1})^{g^{-1}}$, and $\rho^{T}=\rho$. (iv) direct expansion. (v) is (iii) at $U=1$. (vi) is the transformation law of $M_k^{U}$. ◻

</div>

Verified on random complex instances (torus $N=4,6,8$, $n_1=48,108,192$): covariance and transpose residuals $\le7\times10^{-14}$, $F_B$ symmetry $\le8\times10^{-12}$ (SVP Table 2).

# Riesz projectors, frames, covariance

For a contour $\Gamma_C$ enclosing an isolated part of the spectrum, $$P_C(U)=\frac{1}{2\pi i}\oint_{\Gamma_C}\big(\zeta I-h_1(s,U)\big)^{-1}d\zeta,\qquad
P_C(U)^{T}=G_1^{U^{-1}}P_C(U^{-1})\big(G_1^{U^{-1}}\big)^{-1},$$ computed on geometric images: $(\zeta I-h_1)^{-1}=M_1^{U}(\zeta M_1^{U}-\widetilde{A}_1^{U})^{-1}$, one sparse complex factorization per quadrature node. At $U=1$, $G_1P_C=P_C^{T}G_1$: the projector is symmetric for the chain metric.

<div id="prop:frames" class="proposition">

**Proposition 4** (Canonical left frame). *Let $\Phi_C(U)$ span $\operatorname{Ran}P_C(U)$ (chains) and $\Phi_C^{\vee}=\Phi_C(U^{-1})$ span the same band for the dual connection. Then $\operatorname{span}\widetilde\Phi_C=G_1^{U^{-1}}\operatorname{span}\Phi_C^{\vee}$, and with the pairing matrix $B_C(U)=(\Phi_C^{\vee})^{T}G_1^{U}\Phi_C(U)$, the RSF normalization $\widetilde\Phi^{T}\Phi=I$ is achieved by $\widetilde\Phi_C=G_1^{U^{-1}}\Phi_C^{\vee}B_C(U)^{-T}$ iff $\det B_C(U)\ne0$. At $U=1$: $\widetilde\Phi_C=G_1\Phi_C$ once $\Phi_C^{T}G_1\Phi_C=I$; the left frame is the geometric image of the right frame (its entries are edge integrals of the fiber, signed lengths on a boundary circle) and the residual frame gauge is $O(r,\mathbb{C})$. All of these are evaluated with solves: $G_1\Phi$ means the solution of $M_1X=\Phi$.*

</div>

Consequences: the left fiber of a cluster is the geometric image of the same band for $U^{-1}$, i.e. of its anti-cluster; RSF’s “complex bilinear restriction” is $B_C=\Phi^{T}G_1\Phi$; $\det B_C=0$ is the isotropic band, the only place where RSF’s general biorthogonal machinery is needed, and it coincides with the CH exceptional-point indicator; the quasi-free covariance is $\Gamma=\Phi\Phi^{T}G_1$, the $G_1$-orthogonal projector onto the fiber, with $n_e=\Gamma_{ee}$.

Certificates per band: contour and node count; $\|P^2-P\|$; $\operatorname{rank}P$ by SVD (CH tolerance policy); $\max_{\zeta\in\Gamma_C}\|(\zeta I-h_1)^{-1}\|$; $\det B_C$, $\operatorname{cond}B_C$; left/right residuals when semisimple.

# The recursion closes on symmetric pencils

<div id="prop:pencil" class="proposition">

**Proposition 5**. *Work with $\mathcal P(\lambda)=\widetilde{A}_1^{U}-\lambda M_1^{U}$ on geometric images (equivalently $A_1^{U}-\lambda G_1^{U}$ on chains). Partition edges into interface $B$ and interior $I$.*

1.  *$F_B(\lambda)=\mathcal P_{BB}-\mathcal P_{BI}\mathcal P_{II}^{-1}\mathcal P_{IB}$, $\det\mathcal P=\det\mathcal P_{II}\det F_B$; $F_B(\lambda;U)^{T}=F_B(\lambda;U^{-1})$, symmetric at $U=1$.*

2.  *Craig–Bampton/AMLS is the congruence $(T^{T}\widetilde{A}_1T,\ T^{T}M_1T)$.*

3.  *Retained fibers $J=[\Phi_{v_1}\ \Phi_{v_2}\cdots]$ (chains) with images $Z=G_1J$ and $\widetilde J=Z$ give $$\mathcal G_{\ell+1}=\widetilde J^{T}J=J^{T}G_1J=Z^{T}M_1Z,\qquad \hat A_{\ell+1}=J^{T}A_1J=Z^{T}\widetilde{A}_1Z ,$$ the chain metric and pencil restricted to the retained fibers. Off-diagonal blocks $Z_A^{T}M_1Z_B$ vanish unless the supports of the two geometric images share a top simplex (sparsity of $M_1$); with cluster supports defined on the images, overlaps are local. Revision 1’s nonlocality remark referred to supports defined on the chains, whose images spread.*

4.  *Transfer from the pencil block, $\mathsf{T}_{AB}(U)=(Z_A^{\vee})^{T}(\widetilde{A}_1^{U})_{AB}Z_B=(\Phi_A^{\vee})^{T}(A_1^{U})_{AB}\Phi_B$, obeys $\mathsf{T}_{BA}(U^{-1})=\mathsf{T}_{AB}(U)^{T}$; at $U=1$, $\mathsf{T}_{BA}=\mathsf{T}_{AB}^{T}$.*

</div>

The type stable under the RSF recursion is therefore “complex symmetric pencil with an inherited chain metric” at $U=1$, and the pair $(\mathcal P(\lambda;U),\mathcal P(\lambda;U^{-1}))$ related by transposition in general. RSF’s three options for the overlap collapse to “carry $\mathcal G$”, at no cost. Reversal of a hopping-defined transfer is transposition; RSF’s $M^{\vee}=M^{-T}$ follows only under the additional groupoid hypothesis $\mathsf{T}_{BA}=\mathsf{T}_{AB}^{-1}$, and the sign flip of the determinant winding must otherwise come from the closure convention.

# Face anchors

<div id="prop:face" class="proposition">

**Proposition 6**. *For a nondegenerate triangle $t$, the Whitney block $M_1^{(t)}\in\mathbb{C}^{3\times3}$ (the contribution of $t$ to $M_1$, restricted to its three edges) is nonsingular.*

</div>

<div class="proof">

*Proof.* The three Whitney 1-forms of $t$ are linearly independent functions on $t$ and $M_1^{(t)}$ is their Gram matrix in a nondegenerate metric. ◻

</div>

Verified: rank 3 on every triangle of the Lorentzian torus of §<a href="#sec:tests" data-reference-type="ref" data-reference="sec:tests">14</a>; the Grassmann block has rank 2. Hence RSF §10 has a concrete connection-dressed face endomorphism on chains, $\Pi_\tau(U)=G_1^{U}M_1^{(\tau)U}G_1^{U}$ with $(M_1^{(\tau)U})_{ee'}=(M_1^{(\tau)})_{ee'}U_{b(e)b(e')}$, covariant by the same argument as Prop. <a href="#prop:h" data-reference-type="ref" data-reference="prop:h">3</a>(iii), and the invariant anchor coordinate $\alpha_\tau=\det\big((\Phi_Q^{\vee})^{T}\Pi_\tau(U)\Phi_Q\big)=\det\big((Z_Q^{\vee})^{T}M_1^{(\tau)U}Z_Q\big)$, the face block paired through the geometric images, is a well-defined complex number that is generically nonzero for a rank-three fiber. Under the Grassmann metric the same quantity vanishes identically.

# Spectra and signature

<div class="proposition">

**Proposition 7** (One-dimensional complexes). *For a 1-complex with real $s_e$ of any signs, $M_0$ is real positive definite (up to the global factor of §<a href="#sec:branch" data-reference-type="ref" data-reference="sec:branch">4.2</a> when all edges are timelike) and $M_1=\operatorname{diag}(1/l_e)$. Then $\widetilde{A}_1=M_1\partial_1^{T}M_0^{-1}\partial_1M_1\succeq0$ and every eigenvalue of $L_1$ is real: writing the pencil on images, if $z^{\dagger}M_1z\ne0$ then $\lambda\in\mathbb{R}$; otherwise $\widetilde{A}_1z=0$ and $\lambda=0$. The harmonic chain is the fundamental cycle $h_e=\pm1$, and its geometric image is $z_e=(G_1h)_e=\pm l_e$, the signed edge lengths, with $l_e=\sqrt{s_e}$ on the declared branch.*

</div>

A 1-complex with edges of both causal types is not a slice of a Lorentzian manifold; its $M_1$ carries relative factors of $i$ and its spectrum is complex (Test T5b). Boundary circles of a Lorentzian cobordism are spacelike and real.

In the bulk $\partial_2M_2\partial_2^{T}$ is indefinite whenever $M_2$ has negative entries, and complex-conjugate pairs can occur. Evidence: on $3\times3$ tori with CDT-like random Lorentzian lengths the spectrum was real in $120/120$ draws ($N=3,5,7$, 40 each) under `L2` and under Grassmann; with incoherent random signs, $143/200$ draws had complex pairs (Grassmann; `L2` to be run). On the fixed Lorentzian torus of §<a href="#sec:tests" data-reference-type="ref" data-reference="sec:tests">14</a> the harmonic 2-plane has Gram $\Phi^{T}G_1\Phi=Z^{T}M_1Z=i\,Z^{T}M_1^{\rm real}Z$ with $\det(Z^{T}M_1^{\rm real}Z)<0$: one spacelike and one timelike harmonic cycle, signature $(1,1)$.

# Geometric fidelity and the Lorentzian protocol

<div class="finding">

*Finding 1* (SVP Tables 1, 4, 5). Angles between the span of the geometric images $G_1H_1$ and the edge integrals of the continuum harmonic forms, in degrees, for the Whitney metric. Flat jittered torus and flat cylinder (both signatures, $N\le10$): $<10^{-8}$. Curved conformally flat torus, Euclidean: $2.65,1.24,0.75,0.36$ at $N=8,12,16,24$ (second order). Same with complex conformal factor $0.3+0.2i$: $3.19,1.49,0.91$ at $N=8,12,16$. Curved torus with timelike direction rotated by $e^{-2i\varepsilon}$: $\varepsilon=0.1$: $3.40\to0.97$; $\varepsilon=0.3$: $3.14\to0.58$; $\varepsilon=0.6$: $3.02\to0.46$ (second order). Real Lorentzian, $\varepsilon=0$: $3.9,\,34.0,\,2.2,\,12.4$, non-monotone, while the gap $\varsigma_r/\varsigma_{r+1}$ of the stacked matrix collapses from $5\times10^{11}$ to $2\times10^{8}$. Complex conformal factor on a Lorentzian base (not allowable, argument sum $\ge\pi$): $65$–$90^{\circ}$.

</div>

The real Lorentzian failure is the rank-condition failure set of Prop. <a href="#prop:rank" data-reference-type="ref" data-reference="prop:rank">2</a> being approached under refinement: a nearly neutral exact or coexact chain forms, the harmonic representative becomes ill-conditioned, and the discrete problem inherits the non-ellipticity of the continuum one. On the Kontsevich–Segal allowable domain the local bilinear forms are sectorial and the Galerkin method is stable (conditional theorem; the sectoriality-to-stability step is standard, the application to the pencil is not written out here).

<div id="req:lorentz" class="requirement">

**Requirement 2** (Lorentzian protocol). A Lorentzian instance is computed as the family $s_e(\varepsilon)$ with the timelike part of every squared length rotated by $e^{-2i\varepsilon}$ (equivalently, complex lengths on the allowable side of the boundary), at one or more reported $\varepsilon>0$; results at $\varepsilon=0$ are reported only alongside their gap certificate and never alone. Extrapolation to $\varepsilon\to0$ is a separate, labeled step. This is the operational content of “complex lengths are the $i\varepsilon$” (CH §1) and of RSF’s requirement that spectral bands be selected on the complex plane.

</div>

# Numerical program

1.  Assemble $M_0,M_1,M_2$ (sparse), verify allowability margin, factor $M_0$.

2.  Harmonic chains: $H_1=M_1\ker S$ with $\ker S$ by sparse rank-revealing QR (dense SVD oracle); the images $\ker S$ are the readout. Check $\dim H_1=b_1$, (R1)–(R4), gap ratio.

3.  Dress $M_k$, build $\partial_k^{U}$, $\widetilde{A}_1^{U}$; assert Prop. <a href="#prop:h" data-reference-type="ref" data-reference="prop:h">3</a>(i)–(vi) on every instance.

4.  Riesz bands by trapezoidal quadrature on the pencil resolvent for $U$ and $U^{-1}$ on the same contour; $B_C$, frames, certificates.

5.  Pencil Feshbach/Craig–Bampton; coarse $(\hat{\widetilde{A}},\hat{\mathcal G})$; transfer with the reversal identity as a runtime assertion.

6.  For Lorentzian instances, the $\varepsilon$-family of Req. <a href="#req:lorentz" data-reference-type="ref" data-reference="req:lorentz">2</a>.

# Amendments and clarifications

#### CH.

Default preset `L2`; presets `TOP`, `VOL` removed; `GRASSMANN_ALL` retained with the SVP deviation note; App. B superseded by §<a href="#sec:h1" data-reference-type="ref" data-reference="sec:h1">5</a>; §6.3 phase dressing retained as a geometric deformation only; App. A branch rule applies per top simplex; the auxiliary solve of §<a href="#sec:aux" data-reference-type="ref" data-reference="sec:aux">4.3</a> is the implementation path; Req. <a href="#req:lorentz" data-reference-type="ref" data-reference="req:lorentz">2</a> added.

#### RSF.

§3: the nondegeneracy theorem is Prop. <a href="#prop:rank" data-reference-type="ref" data-reference="prop:rank">2</a>; $W_k=M_k^{-1}$. §3.2: left frames are the geometric images of right frames (Prop. <a href="#prop:frames" data-reference-type="ref" data-reference="prop:frames">4</a>); $G_k(g)=\rho_k(g)$. §5, §14: recursion closes on symmetric pencils with inherited chain metric $J^{T}G_1J$, local on image supports. §9: hopping transfer reverses by transposition. §10: face anchor realized by $\Pi_\tau=G_1^{U}M_1^{(\tau)U}G_1^{U}$, rank three. §5.6, §12b: Lorentzian bands are selected at reported $\varepsilon>0$.

# Interface sketch (additive to `tessera`)

    namespace tessera::chainhodge {
    enum class Preset { L2 /*default*/, GRASSMANN_ALL };
    struct WhitneyMass {            // sparse M_k; volumes on the declared branch
      static SpMatC assemble(const OrientedComplex&, const SquaredLengths&, int k, Branch);
      static double allowabilityMargin(const OrientedComplex&, const SquaredLengths&); // min over T of pi - sum|arg lambda_i|
    };
    struct ChainHodge {
      ChainHodge(const OrientedComplex&, const SquaredLengths&, Preset = Preset::L2, Branch = Branch::Continuation);
      SpMatC Minv(int k) const;                  // inverse chain metric M_k (explicit, sparse)
      MatC   applyG(int k, const MatC& c) const; // G_k c = M_k^{-1} c by sparse solve; geometric image
      SpMatC pencilAux(int k) const;             // A~_k = M_k A_k M_k (sparse; uses solves with M_{k-1})
      MatC   harmonicChains(int k, double kappa = 10) const;     // H_k = M_k * nullspace([d_{k+1}^T ; d_k M_k])
      MatC   geometricImage(int k, const MatC& H) const;         // G_k H (= the nullspace vectors)
      RankReport rankConditions(int k) const;    // R1..R4
      std::vector<int> betti() const;            // exact over Q
    };
    struct CovariantChainHodge {                 // dressed sparse M_k, twisted incidences
      CovariantChainHodge(const ChainHodge&, const Connection&);
      SpMatC Minv(int k) const;  SpMatC pencilAux(int k) const;
      MatC   applyH(int k, const MatC& c) const; // h_k(s,U) c, via solves; never formed densely
      CovariantChainHodge dual() const;          // U^{-1}
      Band band(int k, const Contour&) const;    // Phi, PhiDual (chains), B_C, certificates
      static MatC leftFrame(const Band&, const ChainHodge& dualMetric); // G^{U^-1} PhiDual B^{-T}
    };
    struct PencilSchur {
      static std::pair<MatC,MatC> feshbach(const SpMatC& A, const SpMatC& M, cplx lambda, const std::vector<int>& interface);
      static std::pair<SpMatC,SpMatC> craigBampton(const SpMatC& A, const SpMatC& M, const MatC& T);
      static std::pair<MatC,MatC> restrictToFibers(const SpMatC& Aaux, const SpMatC& M, const MatC& Z); // (Z^T A~ Z, Z^T M Z) on images
    };
    struct LorentzianFamily {                    // s(eps): timelike parts rotated by e^{-2 i eps}
      static SquaredLengths rotate(const SquaredLengths&, const CausalTypes&, double eps);
    };
    }

# Verification suite

Values from the Python oracle (, ). “Exact” marks hand-verified values.

1.  **T1–T4, covariant operator** (random complex $s,U$, torus $N=4,6,8$): covariance $\le7.1\times10^{-14}$, transpose identity $\le2.4\times10^{-14}$, $F_B$ symmetry $\le8.0\times10^{-12}$; curvature formula (iv) to $5\times10^{-16}$; pure-gauge isospectrality to $2\times10^{-13}$.

2.  **T5a, Euclidean 3-cycle** $l=(1,2,3)$ on edges $(01),(02),(12)$ (exact): $M_1=\operatorname{diag}(1,\tfrac12,\tfrac13)$, $M_0=\begin{pmatrix}1&\frac16&\frac13\\ \frac16&\frac43&\frac12\\ \frac13&\frac12&\frac53\end{pmatrix}$; harmonic chain $h=(1,-1,1)$ (the cycle in reference orientation), geometric image $z=G_1h=(1,-2,3)$, the signed lengths; eigenvalues of $L_1$ $\{0,\ 1.299254,\ 2.518928\}$.

3.  **T5b, mixed-signature 3-cycle** $s=(1,1,-1)$, principal branch: $M_1=\operatorname{diag}(1,1,-i)$, $h=(1,-1,1)$, $z=G_1h=(1,-1,i)$, the signed lengths including $l_{12}=i$; eigenvalues $\{0,\ -6i,\ 4.8-3.6i\}$. Not a Lorentzian slice; included to fix the branch behaviour.

4.  **T6, CDT-like torus** $s_{\rm slice}=1$, $s_{\rm trans}=-\tfrac12$, $s_{\rm diag}=\tfrac12$: $M_2=-2\sqrt2\,i\,I_{18}$ (exact: $\det g_t=-\tfrac12$); (R1)–(R4) ranks $17,8,8,17$; spectrum real to $3\times10^{-14}$, value (multiplicity):

    <div class="center">

    |            |            |       |            |      |     |          |     |      |           |       |            |
    |-----------:|-----------:|------:|-----------:|-----:|----:|---------:|----:|-----:|----------:|------:|-----------:|
    | $-13.9921$ | $-13.0909$ | $-12$ | $-10.6274$ | $-6$ | $0$ | $4.9655$ | $6$ | $48$ | $92.9132$ | $144$ | $493.9921$ |
    |          2 |          2 |     4 |          2 |    2 |   2 |        2 |   2 |    4 |         2 |     1 |          2 |

    </div>

    $\dim H_1=2$; harmonic Gram $\det(\Phi^{T}G_1\Phi)=\det(Z^{T}M_1Z)=0.211555$ with $M_1=i\,M_1^{\rm real}$, i.e. real signature $(1,1)$; per-triangle Whitney block rank 3.

5.  **T7, Euclidean torus** all $s_e=1$, value (multiplicity):

    <div class="center">

    |     |         |     |      |      |           |      |
    |----:|--------:|----:|-----:|-----:|----------:|-----:|
    | $0$ | $5.671$ | $8$ | $16$ | $24$ | $31.2521$ | $48$ |
    |   2 |       6 |   6 |    4 |    2 |         6 |    1 |

    </div>

6.  **G1–G3, G5, G6** as in SVP, with the comparison made on geometric images $G_1H_1$, with the acceptance criteria stated there (flat: $<10^{-8}$ degrees; curved Euclidean and allowable complex: estimated order $\ge1.5$; real Lorentzian: report angles and gap, no pass criterion).

7.  **T8, frames; T9, pencil reduction; T10, face rank** as in revision 1 with $G_1=M_1^{-1}$; T10 now expects rank 3.

<div class="thebibliography">

9 \[1\] Chain-Level Hodge Theory on Complex and Lorentzian Simplicial Complexes with the Grassmann Projection Metric (CH), September 2026. \[2\] Scaling Verification Plan for the Chain-Level Hodge Construction (SVP), September 2026. \[3\] Recursive Spectral Fibers on Simplicial Cobordisms (RSF), Tessera cobordism programme. \[4\] J. Dodziuk, Finite-difference approach to the Hodge theory of harmonic forms, Amer. J. Math. 98 (1976). \[5\] H. Whitney, *Geometric Integration Theory*, Princeton (1957). \[6\] M. Kontsevich, G. Segal, Wick rotation and the positivity of energy in quantum field theory, Quart. J. Math. 72 (2021). \[7\] T. Kato, *Perturbation Theory for Linear Operators*, Springer (1966).

</div>
