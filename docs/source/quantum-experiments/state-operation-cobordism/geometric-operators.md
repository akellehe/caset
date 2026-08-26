# Fixed-boundary spectral Choi synthesis and geometric operators

Issue [#899](https://github.com/akellehe/tessera/issues/899) reconstructs the
semantics that existed when report commit `5482829` was written. The audit
found that the initial implementation and the later period machinery had been
conflated:

- `78de485` introduced direct Choi-state synthesis through
  `RealizabilityOracle`;
- `165f8af` recorded the successful realizability experiment;
- `be3f7e7` later introduced carried-register period residuals;
- `5482829` presented a theoretical program and explicitly left the bridge
  from action-relaxed geometry to $Z(W)$ open.

The successful pre-paper calculation fixed the relative amplitudes of selected
components to $\operatorname{vec}(U)$, optimized all remaining state
amplitudes together with interior geometry, and minimized a Rayleigh
eigenresidual. It did not fit input/output pairs, impose charge conservation,
require a harmonic state, or include Regge stationarity. The newer period
experiment remains available as a separate diagnostic.

## Claims under test

The proposed correspondence uses

$$
W_{AB}=\operatorname{geo}(U),
\qquad
\partial W_{AB}
 =\overline{\operatorname{geo}(\psi_B)}
  \sqcup\operatorname{geo}(\psi_A),
\qquad
Z(W_{AB})=\langle\psi_A|U|\psi_B\rangle .
$$

Here $\sqcup$ is **disjoint union**. The experiment establishes only the
following topological part: its triangulated $W$ is a cobordism and its
oriented boundary decomposes into incoming and outgoing components. The code
does not define or prove either $\operatorname{geo}(\psi)$ or
$\operatorname{geo}(U)$, and it does not implement a TQFT functor $Z$.
Consequently the three displayed equalities remain conjectural.

The existing Choi--Jamiołkowski implementation does verify the algebraic
identity

$$
\langle\psi_A|U|\psi_B\rangle
 =
\langle\operatorname{vec}(
 |\psi_A\rangle\langle\psi_B|)
 \mid\operatorname{vec}(U)\rangle .
$$

That identity alone does not identify the geometric readout with $Z(W)$.

## Historical construction: direct spectral Choi synthesis

For a supplied $d\times d$ operator, define

$$
|J_U\rangle=\frac{\operatorname{vec}(U)}
                  {\|\operatorname{vec}(U)\|}.
$$

An explicit ordered set $F$ of $d^2$ cochain components is fixed to this
target before global normalization. Every component outside $F$ is an
auxiliary complex amplitude. Thus the normalized full cochain restricts to the
target ray on $F$; its support norm is not fixed. The relaxation varies only
interior edge weights and, at degree zero, interior connection phases, while
minimizing

$$
r_{\mathrm{eig}}(W,\psi)
=\left\|L_W\psi-
 \langle\psi,L_W\psi\rangle\psi\right\|^2.
$$

The eigenvalue is free. If the residual does not converge, the implementation
performs a boundary-preserving stellar subdivision and retries. Boundary edge
lengths and phases remain bit-identical. This is the numerical structure of
the old `RealizabilityOracle`, now exposed additively through
`MultiCobordism.relax_fixed_boundary_eigenstate` with explicit support cells
instead of the old fragile first-component convention.

Unvectorizing the normalized restriction $\psi|_F$ returns $U$, and then
applying that matrix to new states returns the corresponding outputs. This is
valid Choi algebra, but it is target-conditioned by construction: the same
$\operatorname{vec}(U)$ being recovered was pinned during the fit. It is not
operator learning from state pairs or target-free extraction from the bulk.

## Coupled construction: two fixed boundary components

The additive boundary-value mode implements the semantics missing from the
historical oracle. Let

$$
\partial W=\overline{\Sigma_B}\sqcup\Sigma_A
$$

have exactly two connected components. Each component is declared as one named
pinned region, and every degree-$k$ boundary cell is supplied in an explicit
ordered frame. Before the bulk fit, each input row is normalized and its
paired output is scaled by the same factor, preserving operator amplitudes.
Both restrictions are checked on their isolated components:

$$
r_{\partial,j}
=\left\|L_{\Sigma}\widehat\phi_j
 -\langle\widehat\phi_j,L_{\Sigma}\widehat\phi_j\rangle
  \widehat\phi_j\right\|^2
<\epsilon_{\partial}.
$$

For a complete collection of input/output pairs
$(b_j,a_j)$, the full cochain $p_j$ is constrained exactly by

$$
p_j|_{\Sigma_B}=b_j,\qquad p_j|_{\Sigma_A}=a_j .
$$

All other cochain amplitudes are pair-specific auxiliary variables. One edge
geometry is shared by every pair. Edges internal to either named boundary
region are never written; all other edge weights and, at $k=0$, connection
phases may vary. The fit minimizes

$$
R(W,\{p_j\})=
\sum_j\left\|
 L_W\widehat p_j-\bar\lambda\widehat p_j
\right\|^2,
\qquad
\bar\lambda=\frac1m\sum_j
\langle\widehat p_j,L_W\widehat p_j\rangle .
$$

The common eigenvalue is essential. If $R=0$, every linear combination
$p(c)=\sum_jc_jp_j$ is another $\bar\lambda$-eigenvector and has boundary
restriction

$$
p(c)|_{\partial W}
=\left(\sum_jc_jb_j,\ \sum_jc_ja_j\right).
$$

When $b_j$ is a basis and $a_j=Ub_j$, the fitted eigenspace therefore has
boundary trace equal to the graph of $U$ on the trained span. A new input is
attached by taking the same linear combination of the fitted witnesses; no
second optimization is performed.

This is stronger than direct Choi pinning because unseen states test the
fitted witness span. It is still boundary-conditioned. It does not prove that
the bulk-minus-boundary alone canonically determines $U$, exclude additional
eigenvectors with other boundary traces, or define the TQFT value $Z(W)$.

## Later construction: ordered period residuals

The boundary register is the charge-zero plane

$$
V=\ker(1,1,1)\subset\mathbb C^3,
\qquad
Q=|c\rangle\langle c|,
\quad c=(1,1,1)/\sqrt3 .
$$

A candidate preserves the represented charge when $[U,Q]=0$. For each
logical input $x_j$, the algebraic output $Ux_j$ is computed first. The
ordered input/output periods are then declared as one exact constraint:

$$
r_U(W;x_j,Ux_j)
 =
r_{\mathrm{period}}\bigl(
  (x_j,Ux_j);W\bigr).
$$

MultiCobordism accepts these explicit constraints without permuting their
hole-to-target assignment. This later experiment constructs a pair-of-pants
prism, declares every boundary facet as a pinned region, and runs Stage 2 with

- einstein_hilbert=False;
- real_squared_lengths_only=True;
- empty emergent input and output targets;
- objective $\Phi(W)=\sum_j r_U(W;x_j,Ux_j)$.

The 78 boundary edges remain bit-identical. Only the 24 non-boundary edges
move. The objective decomposition reports a Regge contribution of exactly
zero.

The scale-invariant hard period gap is recorded independently:

$$
g_U(W)=\sum_j
 \min_{h\in\ker L_1(W)}
 \left\|\operatorname{per}(h)-(x_j,Ux_j)\right\|^2 .
$$

At finite geometry, $g_U=0$ is the direct period-realizability test. The
later $r_U$ has the same exact zero set, but its magnitude also depends on
the non-harmonic representative and can become small while the target remains
a fixed distance from the harmonic period subspace.

## Two target-free readouts

### Boundary transport

Let $A$ and $B$ be the restrictions of the live harmonic period space to
orthonormal input and output charts on $V$. If both are full rank, the frozen
geometry determines

$$
T_W=B^T(A^T)^{-1}.
$$

No target enters this readout. A new state is attached after fitting by
$x\mapsto T_Wx$. A complete basis can therefore test whether the frozen
transport equals a candidate operator. One input/output pair cannot: it
specifies only one column of an otherwise undetermined map.

This readout uses the harmonic space of the full $W$. It is not a state of
$W-\partial W$.

### Framed bulk Choi promotion

Let

$$
K_W=\ker L_1(W-\partial W),
$$

where $W-\partial W$ is implemented as the subcomplex induced by interior
vertices. The new metric mode restricts the live signed Hodge weights before
taking the right kernel. The combinatorial mode remains the
backwards-compatible default.

An operator interpretation additionally requires an ordered tensor-product
frame $F$ of exactly $d^2$ interior 1-cells. Restrict $K_W$ to that frame.
The promotion rule is:

$$
\operatorname{rank}(K_W|_F)=
\begin{cases}
0 & \text{no Choi component},\\
1 & \text{one projective Choi ray},\\
>1 & \text{an operator family, hence non-identifiable}.
\end{cases}
$$

Only the rank-one case returns a normalized state $|J_W\rangle$ and

$$
U_W=\sqrt d\,\operatorname{unvec}(|J_W\rangle).
$$

Unitarity is measured, not assumed. Missing frames, repeated frame cells,
empty bulk, and multidimensional restrictions return explicit obstructions.
An arbitrary kernel basis vector is never reshaped into an operator.

## Deterministic results

The historical mode uses seed 0, 80 restarts, at most four subdivisions, and
requests $r_{\mathrm{eig}}<10^{-24}$.

| directly pinned Choi target | $\|[U,Q]\|_F$ | residual | growth | boundary drift | held-out error |
|---|---:|---:|---:|---:|---:|
| phase gate $\operatorname{diag}(1,e^{0.41i})$ | 0 | $7.43\times10^{-28}$ | 3 | 0 | 0 |
| charge-changing $X$ | 1.41421 | $2.69\times10^{-27}$ | 2 | 0 | $2.72\times10^{-16}$ |

Both targets converge. Charge conservation is therefore neither encoded nor
necessary in the historical inverse-eigenvector problem. The held-out column
is an algebraic check after unvectorizing the pinned support, not evidence that
the geometry learned an operator from examples. Because the reported residual
is squared, the corresponding defect norms are $2.73\times10^{-14}$ and
$5.19\times10^{-14}$; relative to $\|L\psi\|$, both are approximately
$4.1$--$4.3\times10^{-15}$, the relevant floating-point-scale statement.

The coupled control joins two three-edge circles by an annular triangulation.
The two Fourier modes of each circle form a degenerate isolated-boundary
eigenspace. A generic complex $2\times2$ unitary is taken from the neutral
sector of the charge-preserving qutrit operator. With seed 0, four restarts,
at most eight interior subdivisions, and $R<10^{-16}$, the deterministic
run gives:

| coupled diagnostic | value |
|---|---:|
| total full-$W$ residual | $7.36\times10^{-17}$ |
| subdivisions | 7 |
| boundary geometry drift | 0 |
| fixed restriction error | 0 |
| maximum isolated-boundary residual | $1.22\times10^{-30}$ |
| recovered operator error | $2.51\times10^{-16}$ |
| maximum held-out full-$W$ residual | $1.36\times10^{-16}$ |
| maximum held-out boundary error | $1.84\times10^{-16}$ |
| fitted Rayleigh-quotient spread | $2.88\times10^{-10}$ |

The finite quotient spread is consistent with the squared residual tolerance;
the optimizer constrains the common eigenvalue through the residual rather
than imposing equality as a separate algebraic equation. The held-out
residual and exact restrictions certify the linear-extension claim at the
reported tolerance.

The later period mode uses 12 one-step updates with $\alpha=0.05$.

| case | initial $r_U$ | final $r_U$ | hard gap | target-free error |
|---|---:|---:|---:|---:|
| one reflection pair | $2.67\times10^{-27}$ | unchanged | $1.55\times10^{-28}$ | unseen input: $2.0$ |
| complete identity basis | $3.51\times10^{-27}$ | unchanged | $1.68\times10^{-28}$ | $1.09\times10^{-14}$ |
| complete mapping-class basis | $1.53\times10^{-26}$ | unchanged | $9.60\times10^{-28}$ | $2.48\times10^{-14}$ |
| generic charge-preserving basis | $2.04983$ | $0.0193518$ | $0.135345$ | $0.520279$ |
| charge-leaking basis | $2.23710$ | not relaxed | $0.177925$ | charge commutator: $0.589544$ |

The one-pair reflection and identity constraints agree to
$1.57\times10^{-16}$. Both therefore have tiny period residuals on the same
geometry, whose frozen transport is the identity. Attaching the unseen second
basis state distinguishes them with error $2$. A single pair does not identify
an operator.

For the generic unitary,

$$
\|[U,Q]\|_F=1.36\times10^{-16},
$$

yet the hard gap remains $0.1353453088$, the boundary transport remains the
identity, and the operator error remains $0.5202793649$. Extending the same
run to 80 accepted steps lowers $r_U$ to $4.47\times10^{-9}$ without
changing either obstruction. Small $r_U$ is therefore not evidence that the
operator was stored.

The pair-of-pants prism has no interior 1-cells after deleting its full
boundary. Its bulk Choi readout correctly reports:

> bulk-minus-boundary has no interior 1-cells

A separate square-cylinder control has four interior edges and a
one-dimensional metric kernel. In canonical edge order it promotes to

$$
U_W=\frac1{\sqrt2}
\begin{pmatrix}
1&-1\\
1&1
\end{pmatrix}.
$$

Its unitarity error is $4.44\times10^{-16}$, its commutator with
$\sigma_y$ is $2.22\times10^{-16}$, and the Choi amplitude identity closes
to $1.39\times10^{-16}$. Metric perturbation changes it by only
$3.51\times10^{-16}$, while swapping two frame cells changes the promoted
operator by $2.0$. Thus the control validates the conditional promotion and
also proves that the tensor-product frame is essential external structure.

A two-cycle bulk has kernel dimension two and framed rank two. It is rejected
with “no unique Choi ray exists.”

## Reproduction

From the repository root:

    python examples/cobordism/geometric_operators.py
    python examples/cobordism/geometric_operators.py --live
    python -m pytest tests/cobordism/test_geometric_operators.py -q

The --live option retains the later period diagnostic's animation of the 24
free real squared lengths and, on logarithmic axes, both $r_U$, the hard
gap, and the coupled full-$W$ residual after each interior-growth pass. The
historical solve runs first and is reported in the same record. The default
machine-readable record is
/tmp/cobordism/geometric_operators.json; it is not committed.

## Scientific assessment

The experiment supports these statements:

1. a pinned Choi block can be extended to a full Laplacian eigenstate by
   relaxing interior geometry and auxiliary amplitudes;
2. the tested geometric boundary remains exactly fixed during that solve;
3. complete input/output boundary pairs can share one relaxed geometry and
   one full-cobordism eigenvalue, making their witness span extend linearly to
   unseen input combinations;
4. a complete period basis can identify target-free boundary transport;
5. a uniquely framed rank-one bulk kernel can be promoted to a Choi state.

The historical mode disproves charge conservation as the numerical
realizability criterion: a charge-changing gate converges equally well. The
later period mode separately falsifies charge conservation as a sufficient
condition on the tested prism. It also shows that a one-pair machine-precision
fit does not identify an operator and that the period-residual magnitude is
not a reliable convergence certificate without the hard gap.

This is not a global no-go theorem over other topologies, refinements, or
geometric fields. The current evidence does not establish a map
$U\mapsto W$ that can be read without the pinned target, nor a TQFT functor
$Z$. A paper claiming a universal charge-conserving geometric operator is
therefore not supported. A narrower paper on boundary-conditioned
common-eigenspace extension, identifiability, and framed bulk-kernel
obstructions is defensible if it includes a theorem or systematic topology
classification.
