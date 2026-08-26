# Geometric operators under residual-only relaxation

Issue [#899](https://github.com/akellehe/tessera/issues/899) tests the
operator interpretation using the machinery present at report commit
5482829. The correction is important: the full simplicial boundary is fixed,
the bulk is free, and the only optimized scalar is the historical quantum
residual \(r_U\). No Regge-stationarity condition is imposed.

The result separates period fitting, boundary transport, and bulk Choi
promotion. They are not equivalent.

## Claims under test

The proposed correspondence uses

\[
W_{AB}=\operatorname{geo}(U),
\qquad
\partial W_{AB}
 =\overline{\operatorname{geo}(\psi_B)}
  \sqcup\operatorname{geo}(\psi_A),
\qquad
Z(W_{AB})=\langle\psi_A|U|\psi_B\rangle .
\]

Here \(\sqcup\) is **disjoint union**. The experiment establishes only the
following topological part: its triangulated \(W\) is a cobordism and its
oriented boundary decomposes into incoming and outgoing components. The code
does not define or prove either \(\operatorname{geo}(\psi)\) or
\(\operatorname{geo}(U)\), and it does not implement a TQFT functor \(Z\).
Consequently the three displayed equalities remain conjectural.

The existing Choi--Jamiołkowski implementation does verify the algebraic
identity

\[
\langle\psi_A|U|\psi_B\rangle
 =
\langle\operatorname{vec}(
 |\psi_A\rangle\langle\psi_B|)
 \mid\operatorname{vec}(U)\rangle .
\]

That identity alone does not identify the geometric readout with \(Z(W)\).

## Residual-only construction

The boundary register is the charge-zero plane

\[
V=\ker(1,1,1)\subset\mathbb C^3,
\qquad
Q=|c\rangle\langle c|,
\quad c=(1,1,1)/\sqrt3 .
\]

A candidate preserves the represented charge when \([U,Q]=0\). For each
logical input \(x_j\), the algebraic output \(Ux_j\) is computed first. The
ordered input/output periods are then declared as one exact constraint:

\[
r_U(W;x_j,Ux_j)
 =
r_{\mathrm{period}}\bigl(
  (x_j,Ux_j);W\bigr).
\]

MultiCobordism now accepts these explicit constraints without permuting their
hole-to-target assignment. The experiment constructs the historical
pair-of-pants prism, declares every boundary facet as a pinned region, and
runs Stage 2 with

- einstein_hilbert=False;
- real_squared_lengths_only=True;
- empty emergent input and output targets;
- objective \(\Phi(W)=\sum_j r_U(W;x_j,Ux_j)\).

The 78 boundary edges remain bit-identical. Only the 24 non-boundary edges
move. The objective decomposition reports a Regge contribution of exactly
zero.

The scale-invariant hard period gap is recorded independently:

\[
g_U(W)=\sum_j
 \min_{h\in\ker L_1(W)}
 \left\|\operatorname{per}(h)-(x_j,Ux_j)\right\|^2 .
\]

At finite geometry, \(g_U=0\) is the direct realizability test. The historical
\(r_U\) has the same exact zero set, but its magnitude also depends on the
non-harmonic representative and can become small while the target remains a
fixed distance from the harmonic period subspace.

## Two target-free readouts

### Boundary transport

Let \(A\) and \(B\) be the restrictions of the live harmonic period space to
orthonormal input and output charts on \(V\). If both are full rank, the frozen
geometry determines

\[
T_W=B^T(A^T)^{-1}.
\]

No target enters this readout. A new state is attached after fitting by
\(x\mapsto T_Wx\). A complete basis can therefore test whether the frozen
transport equals a candidate operator. One input/output pair cannot: it
specifies only one column of an otherwise undetermined map.

This readout uses the harmonic space of the full \(W\). It is not a state of
\(W-\partial W\).

### Framed bulk Choi promotion

Let

\[
K_W=\ker L_1(W-\partial W),
\]

where \(W-\partial W\) is implemented as the subcomplex induced by interior
vertices. The new metric mode restricts the live signed Hodge weights before
taking the right kernel. The combinatorial mode remains the
backwards-compatible default.

An operator interpretation additionally requires an ordered tensor-product
frame \(F\) of exactly \(d^2\) interior 1-cells. Restrict \(K_W\) to that frame.
The promotion rule is:

\[
\operatorname{rank}(K_W|_F)=
\begin{cases}
0 & \text{no Choi component},\\
1 & \text{one projective Choi ray},\\
>1 & \text{an operator family, hence non-identifiable}.
\end{cases}
\]

Only the rank-one case returns a normalized state \(|J_W\rangle\) and

\[
U_W=\sqrt d\,\operatorname{unvec}(|J_W\rangle).
\]

Unitarity is measured, not assumed. Missing frames, repeated frame cells,
empty bulk, and multidimensional restrictions return explicit obstructions.
An arbitrary kernel basis vector is never reshaped into an operator.

## Deterministic results

The default command uses 12 one-step residual updates with
\(\alpha=0.05\).

| case | initial \(r_U\) | final \(r_U\) | hard gap | target-free error |
|---|---:|---:|---:|---:|
| one reflection pair | \(1.23\times10^{-27}\) | unchanged | \(3.51\times10^{-29}\) | unseen input: \(2.0\) |
| complete identity basis | \(3.29\times10^{-27}\) | unchanged | \(1.56\times10^{-28}\) | \(8.47\times10^{-15}\) |
| complete mapping-class basis | \(9.63\times10^{-27}\) | unchanged | \(5.79\times10^{-28}\) | \(1.76\times10^{-14}\) |
| generic charge-preserving basis | \(2.04983\) | \(0.0193518\) | \(0.135345\) | \(0.520279\) |
| charge-leaking basis | \(2.23710\) | not relaxed | \(0.177925\) | charge commutator: \(0.589544\) |

The one-pair reflection and identity constraints agree to
\(1.57\times10^{-16}\). Both are therefore fitted to machine precision by the
same geometry, whose frozen transport is the identity. Attaching the unseen
second basis state distinguishes them with error \(2\). A single pair does not
identify an operator.

For the generic unitary,

\[
\|[U,Q]\|_F=1.36\times10^{-16},
\]

yet the hard gap remains \(0.1353453088\), the boundary transport remains the
identity, and the operator error remains \(0.5202793649\). Extending the same
run to 80 accepted steps lowers \(r_U\) to \(4.47\times10^{-9}\) without
changing either obstruction. Small \(r_U\) is therefore not evidence that the
operator was stored.

The pair-of-pants prism has no interior 1-cells after deleting its full
boundary. Its bulk Choi readout correctly reports:

> bulk-minus-boundary has no interior 1-cells

A separate square-cylinder control has four interior edges and a
one-dimensional metric kernel. In canonical edge order it promotes to

\[
U_W=\frac1{\sqrt2}
\begin{pmatrix}
1&-1\\
1&1
\end{pmatrix}.
\]

Its unitarity error is \(4.44\times10^{-16}\), its commutator with
\(\sigma_y\) is \(2.22\times10^{-16}\), and the Choi amplitude identity closes
to \(1.39\times10^{-16}\). Metric perturbation changes it by only
\(3.51\times10^{-16}\), while swapping two frame cells changes the promoted
operator by \(2.0\). Thus the control validates the conditional promotion and
also proves that the tensor-product frame is essential external structure.

A two-cycle bulk has kernel dimension two and framed rank two. It is rejected
with “no unique Choi ray exists.”

## Reproduction

From the repository root:

    python examples/cobordism/geometric_operators.py
    python examples/cobordism/geometric_operators.py --live
    python -m pytest tests/cobordism/test_geometric_operators.py -q

The --live option animates the 24 free real squared lengths and, on a
logarithmic axis, both \(r_U\) and the hard gap after every accepted update.
The non-live and live paths use the same one-step update loop. The default
machine-readable record is
/tmp/cobordism/geometric_operators.json; it is not committed.

## Scientific assessment

The experiment supports these statements:

1. a fixed cobordism can carry selected boundary periods;
2. a complete basis can identify its target-free boundary transport;
3. a uniquely framed rank-one bulk kernel can be promoted to a Choi state;
4. charge leakage is an obstruction.

It falsifies the proposed sufficiency statement on the tested topology:
charge conservation alone does not imply geometric realizability. It also
shows that a one-pair machine-precision fit does not identify an operator and
that the historical \(r_U\) magnitude is not a reliable convergence
certificate without the hard gap.

This is not a global no-go theorem over other topologies, refinements, or
geometric fields. A positive universal paper is therefore not scientifically
supported by the current machinery. A narrower paper on identifiability,
topology-specific obstructions, and the rank-one framed-kernel criterion is
defensible and non-trivial if it adds a theorem or a systematic topology
classification; the present experiment alone is better treated as a rigorous
negative result and foundation for that paper.
