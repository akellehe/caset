

> The rendered vector diagrams are preserved in the LaTeX/PDF edition; this Markdown edition is the searchable text companion.

# Status

Before this document: CH tests 1–11 (single triangle, two triangles, one regular torus, a cycle, a Lorentzian triangle, small covariance checks) and IN tests T1–T7, N1 ($3\times3$ torus). None of these can detect a metric that is consistent but not geometric, because on a regular mesh every translation-invariant metric produces the translation-invariant harmonic representatives. The tests below were designed to detect exactly that, and did.

# The finding

<div id="find:main" class="finding">

*Finding 1* (Geometric fidelity). On jittered triangulations of a flat torus (§<a href="#sec:gen" data-reference-type="ref" data-reference="sec:gen">4</a>), the geometric images $G_1H_1$ of the harmonic chains span the continuum harmonic forms $\{dt,dx\}$ to machine precision for the Whitney chain metric, in both signatures and at every size tested. For the Grassmann `ALL` metric the principal angles between the two spans are $5$–$10^{\circ}$ (Euclidean) and $63$–$80^{\circ}$ (Lorentzian) at 25% jitter, with no decrease from $N=6$ to $N=10$. The same holds on a flat cylinder cobordism with two boundary circles (target $d\theta$): $0^{\circ}$ versus $7.5$–$8.4^{\circ}$ and $69$–$88^{\circ}$. On a curved (conformally flat) torus, where the continuum harmonic forms are still $\{dt,dx\}$ by conformal invariance, the Whitney angles decrease as $4.5,\,2.7,\,1.25,\,0.74^{\circ}$ for $N=6,8,12,16$ (ratio $\approx3.7$ per doubling: second order), while `ALL` stays at $12$–$15^{\circ}$.

</div>

The mechanism is the one anticipated in CH §6.5, Remark on Hodge stars: the Grassmann form is a pointwise inner product of blades and differs from the $L^{2}$ inner product by a local $d$-volume weighting. On a regular mesh that weighting is constant and invisible. On an irregular mesh it varies from cell to cell, and the harmonic representative it selects encodes the triangulation as well as the geometry. Two triangulations of the same flat torus give different “states”. That contradicts the design goal that the encoding be geometric.

<div class="finding">

*Finding 2* (`VOL` is structurally degenerate). The `VOL` preset weights top simplices only, so it inherits the structural kernel of `TOP` (CH Prop. 6.2). On every torus mesh above its nullspace had dimension $3=b_1+1$. Its $L^{2}$ scaling does not compensate: Euclidean angles $1$–$3^{\circ}$, Lorentzian $37$–$73^{\circ}$.

</div>

## Decision

1.  Default chain metric: `L2`, $G_k:=M_k^{-1}$ with $M_k$ the Whitney mass matrix of CH §6, Option A, volumes $|T|=\sqrt{\det g_T}/d!$.

2.  Branch: for real Lorentzian data every top simplex has $\det g_T<0$, so $\sqrt{\det g_T}=\pm i\sqrt{|\det g_T|}$ is a global factor and drops out of every harmonic and spectral quantity; $\sqrt{|\det g_T|}$ (the Lorentzian volume element) and the principal branch gave identical results in every test. For complex data use the continuation or Kontsevich–Segal rule of CH App. A per top simplex. Relative branch ambiguities arise only for lower-dimensional faces in $d\ge3$, which $M_k$ never uses.

3.  `GRASSMANN_ALL` is kept as a named option. It remains polynomial, branch-free, positive definite for Euclidean data, and covariantly dressable; its harmonic representatives are triangulation-dependent at $O(1)$ and this is stated wherever the option is offered.

4.  `VOL` and `TOP` are removed from the presets.

## What changes in CH and IN

Nothing algebraic in IN depends on which complex symmetric metric is used: the covariant construction (dress the sparse $M_k$ entrywise, then invert; verified, Table <a href="#tab:E" data-reference-type="ref" data-reference="tab:E">2</a>), the transpose identity, canonical left frames, pencil closure, the inherited coarse metric $\mathcal G_{\ell+1}=J^{T}G_1J$, and the reversal identity all hold for `L2`. The following statements were specific to the Grassmann form:

- CH §6 Def. 6.3 (presets), Prop. 6.1 (positivity of `ALL`), Prop. 6.2 (bipartite degeneracy of `TOP`): now describe the `GRASSMANN_ALL` option. Prop. 6.2 also explains why `VOL` failed.

- IN §3 (“`ALL` mandatory”): replaced by `L2`; the Riesz resolvent needs $G_1$ invertible, which $M_1$ provides for any nondegenerate mesh.

- IN Prop. 6.1(d), sparsity of overlap blocks: for `L2` the chain metric $M_1^{-1}$ is dense, so fiber overlaps $\Phi_A^{T}M_1^{-1}\Phi_B$ are not confined to shared simplices; they decay with distance but do not vanish. The inheritance statement itself is unchanged.

- IN Prop. 7.1 (a face carries two blade directions): Grassmann-specific. The Whitney mass matrix of one triangle restricted to its three edges is nonsingular (the three Whitney 1-forms are linearly independent functions), so the rank-three face anchor of RSF §10 is natural for `L2` and impossible for `GRASSMANN_ALL`.

- IN §7.2, Prop. 7.2 (real spectra of 1-complexes) holds for any real positive $M_0$ and real $M_1$; for a 1-complex $M_1=\operatorname{diag}(1/l_e)$ and the argument is unchanged.

- CH Prop. 7.4 (circle): the geometric image of the fundamental cycle is $(\pm l_e)$ under `L2`, as CH already stated for the $L^{2}$ scaling.

# Sparse formulation for `L2`

$M_k$ is sparse (pairs in a common top simplex); $G_k=M_k^{-1}$ is dense and must never be formed. Everything needed is available in cochain variables.

<div id="prop:cochain" class="proposition">

**Proposition 1** (Cochain form). *Let $y=G_1h=M_1^{-1}h$. Then $h\in H_1$ iff $$\partial_2^{T}y=0\qquad\text{and}\qquad \partial_1M_1y=0 ,$$ i.e. $y$ is a closed cochain that is coclosed for $M_1$; $y$ is the geometric image and $h=M_1y$ the chain. The pencil $A_1x=\lambda G_1x$ with $x=M_1y$ is equivalent to the symmetric sparse pencil $$\widetilde A_1\,y=\lambda\,M_1\,y,\qquad
\widetilde A_1=M_1\partial_1^{T}M_0^{-1}\partial_1M_1+\partial_2M_2\partial_2^{T},$$ which requires only solves with the sparse $M_0$. The Riesz projector on cochains is $\frac{1}{2\pi i}\oint(\zeta M_1-\widetilde A_1)^{-1}M_1\,d\zeta$.*

</div>

<div class="proof">

*Proof.* Substitute $G_k=M_k^{-1}$ and $x=M_1y$ into CH §7 and multiply by $M_1$. ◻

</div>

The covariant version replaces $M_k$ by the entrywise-dressed $M_k^{U}$ (same rule as CH/IN Def. 4.1, transport between base vertices) and $\partial_k$ by $\partial_k^{U}$, $\partial_k^{U^{-1}}$ in the two factors. The harmonic space is computed as the nullspace of the sparse matrix $\begin{pmatrix}\partial_2^{T}\\ \partial_1M_1\end{pmatrix}$ by rank-revealing sparse QR, or by an inverse-iteration eigensolver on the pencil at $\lambda=0$, with the dense SVD as oracle.

# Test families and generators

All generators are in ; parameters are seeds, sizes, jitter, and signature. Each produces the complex, squared lengths, and where known the continuum harmonic cochain(s) $W$ as edge integrals.

<div class="description">

$N\times N$ vertices, every square split the same way, vertices at lattice points displaced by $\mathrm{jitter}\cdot\mathrm{Unif}(-1,1)/N$ in both coordinates on the flat torus $[0,1)^2$; $s_e=\Delta x^2\pm\Delta t^2$ from coordinates (Euclidean $+$, Lorentzian $-$). Known: $H^1=\mathrm{span}\{dt,dx\}$, $W_e=(\Delta t_e,\Delta x_e)$. With $L_t=L_x$ and zero jitter the diagonals are null in the Lorentzian case and the construction is degenerate by design (CH §1); the plan uses jitter $\ge0.15$ or $L_t\ne L_x$.

$S^1\times[0,L/N]$, $N$ vertices per circle, $L+1$ layers, jitter on interior layers only; absolute complex; two boundary circles. Known: $b_1=1$, $H^1=\mathrm{span}\{d\theta\}$, $W_e=\Delta x_e$.

as F1 with $s_e=e^{2\varphi(\mathrm{mid}_e)}(\Delta x^2\pm\Delta t^2)$, $\varphi=a\sin2\pi t\cos2\pi x$, $a=0.3$. A genuinely curved Regge geometry (nonzero deficit angles). Known: harmonic 1-forms are conformally invariant in two dimensions, so $W$ is as in F1; the discrete answer must converge, not coincide.

regular combinatorics; slice edges $s\sim\mathrm{Unif}(0.5,1.5)$, transverse edges $s\sim-\mathrm{Unif}(0.1,0.9)$, diagonals $s\sim\mathrm{Unif}(0.2,1.5)$. No continuum target; used for spectral statistics.

regular combinatorics, $s_e\sim N(0,1)$ i.i.d.; incoherent causal structure; spectral statistics.

F1–F4 with $s_e\to s_e+i\varepsilon_e$, $\varepsilon_e\sim N(0,\sigma^2)$, and random $U_e\in\mathbb{C}^{*}$ for the covariant tests.

the `tessera` staircase prism of an F1/F2 base () gives 3-complexes with known $b_k$. Not yet exercised; listed so the C++ suite covers $d=3$ from the start.

</div>

# Test catalogue

Tolerance policy (CH §9.1): relative residuals are compared against $\tau=\kappa\,n\,\epsilon_{\rm m}\,\mathrm{cond}(G_1)$, $\kappa=10$; observed values are always reported, never only pass/fail. Angles are principal angles between subspaces in degrees.

## E: exact identities (must hold at every size)

1.  $h_1(s,U^{g})=\rho_1(g)h_1(s,U)\rho_1(g)^{-1}$ for random $g$.

2.  $h_1(s,U)^{T}=G_1^{U^{-1}}h_1(s,U^{-1})(G_1^{U^{-1}})^{-1}$ and $(M_k^{U})^{T}=M_k^{U^{-1}}$.

3.  Pure gauge: spectra of $h_1(s,1^{g})$ and $L_1$ agree (Hausdorff distance relative to $\|L_1\|$).

4.  Per triangle $[p<q<r]$: $\partial_1^{U}\partial_2^{U}t=U_{rp}(\mathcal F_t-1)[r]$.

5.  Feshbach: $F_B(\lambda)^{T}=F_B(\lambda)$ at $U=1$ and $\log\det\mathcal P=\log\det\mathcal P_{II}+\log\det F_B$ (log-determinants, real and imaginary parts separately), for random interface sets and random $\lambda$.

6.  Craig–Bampton congruence symmetry and the coarse pencil $(J^{T}A_1J,\ J^{T}G_1J)$ reproducing the fiber eigenvalues it retains.

7.  Transfer reversal $M_{BA}(U^{-1})=M_{AB}(U)^{T}$.

8.  Riesz projector: $\|P^2-P\|$, rank, $P(U)^{T}=G^{U^{-1}}P(U^{-1})(G^{U^{-1}})^{-1}$, convergence in the number of quadrature nodes.

9.  Frames: $\widetilde\Phi^{T}\Phi=I$ with $\widetilde\Phi=G_1^{U^{-1}}\Phi^{\vee}B_C^{-T}$; $\det B_C$ reported.

## T: topology

1.  $\dim\ker S_1=b_1$, with $b_1$ from exact integer rank (Smith normal form or rank modulo several primes; the C++ path must not use floating-point rank for $b_k$).

2.  (R1)–(R4) with gap ratios $\varsigma_r/\varsigma_{r+1}$; trend of the smallest nonzero singular value with $N$.

## G: geometric fidelity (the tests that found the defect)

1.  F1, both signatures, $N\in\{6,8,10,\dots\}$, jitter $0.25$: angles between $\mathrm{span}(G_1H_1)$ and $\mathrm{span}W$ must be $\le10^{-8}$ degrees for `L2`; reported for `GRASSMANN_ALL`.

2.  F2, both signatures: single angle to $d\theta$, same criterion; also the boundary circles’ images $\propto(l_e)$.

3.  F3, both signatures, $N\in\{6,8,12,16,24\}$: angles must decrease; estimated order from successive doublings reported; pass criterion (Euclidean) order $\ge1.5$; Lorentzian: monotone decrease reported, no threshold yet (§<a href="#sec:open" data-reference-type="ref" data-reference="sec:open">9</a>).

4.  Sanity of the generators: for F1/F2, deficit angles vanish to round-off; for F3 they do not.

## S: spectral statistics

1.  Fraction of geometries with complex-conjugate eigenvalue pairs, F4 versus F5, versus $N$.

2.  Signature of $G_1$ and of the harmonic Gram $\Phi^{T}G_1\Phi$ on Lorentzian tori (expected $(1,1)$ for $b_1=2$).

3.  Distribution of the exceptional-point indicator over the spectrum.

4.  Scaling of the first nonzero $|\lambda|$ with $N$.

## P: performance (C++)

Dense baseline is $O(n_1^3)$: the Python oracle handles $n_1\le10^3$ in seconds. The production path must be sparse: assembly $O(n_d)$; harmonic space by sparse rank-revealing QR of the stacked cochain matrix; contour integrals by one sparse complex LU per node; report wall time, memory, and fill-in versus $n_1$ on F1 with $N$ up to the largest feasible.

## X: cross-checks

Python oracle versus C++ on identical inputs (relative agreement $\le10^{2}\tau$); exact rational arithmetic for $N\le6$ on rational $s_e$ for `GRASSMANN_ALL` (polynomial entries) and for the integer topology.

# First results

All numbers from (dense, seeds recorded in the file). “Euc”/“Lor” = signature; $n_1$ = number of edges.

<div id="tab:G">

| G1: flat torus, jitter $0.25$                      | $n_1$ |    `ALL` angles |     `L2` angles | nullity (both) |
|:---------------------------------------------------|------:|----------------:|----------------:|---------------:|
| Euc $N=6$                                          |   108 |  $10.21,\ 5.78$ |         $0,\ 0$ |              2 |
| Euc $N=8$                                          |   192 |   $9.40,\ 4.30$ |         $0,\ 0$ |              2 |
| Euc $N=10$                                         |   300 |   $8.83,\ 5.16$ |         $0,\ 0$ |              2 |
| Lor $N=6$                                          |   108 | $79.75,\ 63.18$ |         $0,\ 0$ |              2 |
| Lor $N=8$                                          |   192 | $77.28,\ 70.10$ |         $0,\ 0$ |              2 |
| Lor $N=10$                                         |   300 | $78.84,\ 65.25$ |         $0,\ 0$ |              2 |
| G2: flat cylinder, jitter $0.25$                   | $n_1$ |     `ALL` angle |      `L2` angle |        nullity |
| Euc $N=6,L=4$                                      |    78 |          $8.44$ |             $0$ |              1 |
| Euc $N=8,L=6$                                      |   152 |          $7.55$ |             $0$ |              1 |
| Euc $N=10,L=8$                                     |   250 |          $8.24$ |             $0$ |              1 |
| Lor $N=6,L=4$                                      |    78 |         $68.82$ |             $0$ |              1 |
| Lor $N=8,L=6$                                      |   152 |         $88.50$ |             $0$ |              1 |
| Lor $N=10,L=8$                                     |   250 |         $86.76$ |             $0$ |              1 |
| G3: conformally flat torus, $a=0.3$, jitter $0.15$ | $n_1$ |    `ALL` angles |     `L2` angles |        nullity |
| Euc $N=6$                                          |   108 | $14.84,\ 12.19$ | $4.541,\ 2.336$ |              2 |
| Euc $N=8$                                          |   192 | $13.84,\ 12.70$ | $2.708,\ 1.271$ |              2 |
| Euc $N=12$                                         |   432 | $13.66,\ 12.56$ | $1.251,\ 0.702$ |              2 |
| Euc $N=16$                                         |   768 | $13.49,\ 12.56$ | $0.736,\ 0.406$ |              2 |
| Lor $N=6$                                          |   108 | $89.41,\ 85.54$ | $33.85,\ 11.17$ |              2 |
| Lor $N=8$                                          |   192 | $87.96,\ 68.45$ |  $28.02,\ 9.17$ |              2 |
| Lor $N=12$                                         |   432 | $88.18,\ 68.44$ | $16.77,\ 11.45$ |              2 |
| Lor $N=16$                                         |   768 | $89.41,\ 79.44$ | $12.82,\ 10.71$ |              2 |

Geometric-fidelity tests. Angles in degrees between the span of the geometric images of the harmonic chains and the span of the continuum harmonic cochains. “0” means below $10^{-8}$ degrees. Euclidean `L2` in G3 converges at second order (ratio $\approx3.7$ per doubling).

</div>

<div id="tab:E">

| E1–E5 at scale (`L2`, random complex $s,U$) | $n_1$ |       E1 covariance |        E2 transpose |   E5 $F_B$ symmetry | $\mathrm{cond}\,G_1$ |
|:--------------------------------------------|------:|--------------------:|--------------------:|--------------------:|---------------------:|
| torus $N=4$                                 |    48 | $7.1\times10^{-14}$ | $2.4\times10^{-14}$ | $1.5\times10^{-13}$ |    $6.3\times10^{1}$ |
| torus $N=6$                                 |   108 | $8.5\times10^{-15}$ | $8.7\times10^{-15}$ | $6.2\times10^{-12}$ |    $2.0\times10^{2}$ |
| torus $N=8$                                 |   192 | $2.0\times10^{-14}$ | $7.8\times10^{-15}$ | $8.0\times10^{-12}$ |    $2.3\times10^{2}$ |

Exact identities under the Whitney chain metric with covariant dressing of the sparse $M_k$. Relative residuals.

</div>

<div id="tab:S">

| S1: complex-conjugate pairs            |                `ALL` |                 `L2` |
|:---------------------------------------|---------------------:|---------------------:|
| F4 CDT-like, $N=3,5,7$, 40 draws each  | $0/40,\ 0/40,\ 0/40$ | $0/40,\ 0/40,\ 0/40$ |
| F4 CDT-like, $N=3$, 200 draws (IN)     |              $0/200$ |                    — |
| F5 random signs, $N=3$, 200 draws (IN) |            $143/200$ |             (to run) |

Spectral statistics. Real spectra persist for causally layered geometries under both metrics.

</div>

# The Lorentzian limit and Kontsevich–Segal allowability

Open item 1 of the first draft (slow, irregular Lorentzian convergence on F3) was traced to two causes and resolved into one statement.

First, the F3 generator with equal periods makes the lattice diagonals nearly null ($|s_e|\sim10^{-6}$); with period ratio $2$ no edge is within a factor $10^{2}$ of null. Second, and decisive: on the corrected mesh the real Lorentzian case still fails to converge, and the reason is visible in the certificates. Define the gap $\varsigma_r/\varsigma_{r+1}$ of the stacked matrix $S_1$ between its smallest nonzero and its numerically zero singular values. For Euclidean data it is $10^{13}$–$10^{14}$ at every $N$. For real Lorentzian data it collapses, $5\times10^{11}\to2\times10^{10}\to1\times10^{8}\to2\times10^{8}$ for $N=8,12,16,24$: under refinement the mesh approaches the set where the rank conditions (R1)–(R2) fail, i.e. where a nearly neutral exact or coexact chain exists. That is the discrete form of the statement that the Lorentzian Laplacian is hyperbolic rather than elliptic, and it was predicted in CH §7 (failure set of real codimension one).

A Kontsevich–Segal allowable deformation removes it. Rotate only the timelike direction, $$s_e=e^{2\varphi(\mathrm{mid}_e)}\big(\Delta x_e^{2}-e^{-2i\varepsilon}\Delta t_e^{2}\big),$$ whose local eigenvalue arguments are $\pi-2\varepsilon$ and $0$, sum $<\pi$. The continuum harmonic forms are still $\{dt,dx\}$ by two-dimensional conformal invariance, which holds algebraically for complex conformal factors and constant complex $g_0$.

<div id="tab:KS">

| G5: $\varepsilon$                             |                                  $N=8$ |                                  $N=12$ |                                 $N=16$ |                                 $N=24$ |
|:----------------------------------------------|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|
| $0$ (real Lorentzian)                         | $3.86,\ 2.51$ \[$5\!\times\!10^{11}$\] | $33.99,\ 5.07$ \[$2\!\times\!10^{10}$\] |  $2.24,\ 1.47$ \[$1\!\times\!10^{8}$\] | $12.39,\ 3.08$ \[$2\!\times\!10^{8}$\] |
| $0.03$                                        | $3.76,\ 2.45$ \[$3\!\times\!10^{12}$\] |  $5.92,\ 2.70$ \[$8\!\times\!10^{11}$\] | $2.02,\ 1.42$ \[$4\!\times\!10^{11}$\] | $2.07,\ 1.09$ \[$3\!\times\!10^{11}$\] |
| $0.1$                                         | $3.40,\ 2.16$ \[$2\!\times\!10^{13}$\] |  $2.60,\ 1.85$ \[$3\!\times\!10^{13}$\] | $1.55,\ 1.20$ \[$2\!\times\!10^{13}$\] | $0.97,\ 0.67$ \[$1\!\times\!10^{13}$\] |
| $0.3$                                         | $3.14,\ 1.68$ \[$4\!\times\!10^{13}$\] |  $1.59,\ 1.39$ \[$2\!\times\!10^{14}$\] | $1.04,\ 0.90$ \[$2\!\times\!10^{13}$\] | $0.58,\ 0.48$ \[$1\!\times\!10^{13}$\] |
| $0.6$                                         | $3.02,\ 1.54$ \[$7\!\times\!10^{14}$\] |  $1.44,\ 1.07$ \[$1\!\times\!10^{14}$\] | $0.89,\ 0.69$ \[$4\!\times\!10^{13}$\] | $0.46,\ 0.39$ \[$1\!\times\!10^{14}$\] |
| Euclidean reference ($\Delta t^2+\Delta x^2$) |                          $2.65,\ 1.82$ |                           $1.24,\ 1.04$ |                          $0.75,\ 0.60$ |                          $0.36,\ 0.34$ |

Curved torus with period ratio 2, jitter 0.15, `L2`. Entries: principal angles (degrees) to $\{dt,dx\}$, and in brackets the gap $\varsigma_r/\varsigma_{r+1}$. At $\varepsilon\ge0.1$ the convergence is second order and the gap is Euclidean-like; at $\varepsilon=0$ neither holds.

</div>

<div id="tab:conf">

| G6: complex conformal factor $a=0.3+0.2i$                        |         $N=8$ |        $N=12$ |        $N=16$ |
|:-----------------------------------------------------------------|--------------:|--------------:|--------------:|
| Euclidean base (allowable: arguments within $\pm0.4$)            | $3.19,\ 2.19$ | $1.49,\ 1.27$ | $0.91,\ 0.73$ |
| Lorentzian base (not allowable: arguments $\pi+2\beta,\ 2\beta$) | $89.1,\ 65.0$ | $89.9,\ 70.3$ | $89.3,\ 68.4$ |

A complex conformal factor deforms a Euclidean base into an allowable complex metric and convergence persists; on a Lorentzian base it leaves the argument sum at or above $\pi$ and the harmonic images are unrelated to the geometry.

</div>

<div class="finding">

*Finding 3*. Convergence of the `L2` harmonic representatives to the geometric harmonic forms on curved geometries is observed exactly on the Kontsevich–Segal allowable domain (Euclidean, complex-Euclidean, and $\varepsilon$-rotated Lorentzian), degrades as $\varepsilon\to0$, and fails on the real Lorentzian boundary and outside the domain. Mechanism (conditional theorem, not proved here): allowable metrics give sectorial bilinear forms, for which the Galerkin method is stable; indefinite forms have no such guarantee, and the collapsing gap is the observed instability. This is the numerical content of the statement in CH §1 that complex lengths are the $i\varepsilon$ of the construction. Real Lorentzian complexes should be treated as the $\varepsilon\to0$ limit of allowable ones, with $\varepsilon$ reported, not as a self-standing computation.

</div>

# Reporting format

One JSON record per test instance: `family`, `params` ($N$, $L$, jitter, signature, seed, $\sigma$), `preset`, `n0,n1,n2`, `betti`, `nullity`, `gap`, `rank_conditions`, `cond_G1`, `residuals` (E-series), `angles_deg` (G-series), `spectrum_summary` (counts of negative/zero/positive/complex), `ep_indicator_min`, `time_s`, `memory_MB`, `oracle_agreement`. Pass/fail is derived from the record, never stored without the underlying numbers.

# Open items

1.  Resolved in §<a href="#sec:ks" data-reference-type="ref" data-reference="sec:ks">7</a>: real Lorentzian curved geometries do not converge (collapsing gap); Kontsevich–Segal allowable deformations do. Remaining: a proof of the sectoriality/stability mechanism, and the rate at which the admissible $\varepsilon$ may shrink with $N$.

2.  F7 (three-dimensional families) has not been run; $M_2$ for $d=3$ must be implemented (CH §6, Option A, Gram-determinant rule).

3.  Whether the random-sign ensemble F5 produces complex pairs under `L2` at the same rate as under `ALL` is untested.

4.  The sparse production path (§<a href="#sec:sparse" data-reference-type="ref" data-reference="sec:sparse">3</a>) exists only as a formula; the oracle is dense.
