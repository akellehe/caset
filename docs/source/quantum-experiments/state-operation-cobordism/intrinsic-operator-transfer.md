# Intrinsic operator transfer after relaxation

Issue [#899](https://github.com/akellehe/tessera/issues/899) tests the operator
claim without supplying the target during readout. The result is negative for
the proposed universal statement on the fixed prism topology, while validating
target-independent transport for the identity and mapping-class controls.

The method under test is the stationary-action objective and anchored value
readout proposed at report commit `5482829`. It is not the earlier successful
`RealizabilityOracle` calculation, which directly pinned
$\operatorname{vec}(U)$ and minimized a Rayleigh eigenresidual. At the report
commit the harmonic operator used symmetric coordinates and the value code used
their Euclidean pairing. Current `main` returns unwhitened cochains of the signed
Hodge operator, so the equivalent live pairing used here is explicitly
$h^\dagger W_1k$. This coordinate correction does not involve the target.

## Question and construction

The boundary register is the charge-zero plane

$$
V=\ker(1,1,1)\subset\mathbb C^3,
\qquad
Q=|c\rangle\langle c|,
\quad c=(1,1,1)/\sqrt3.
$$

An operator conserves the represented charge when $[U,Q]=0$. Let $E$ be
the fixed orthonormal two-row basis of $V$. For the live harmonic space
$\ker L_1(W)$, the input and output period restrictions in this basis are
matrices $A$ and $B$. A graph-like frozen fill determines

$$
T_W=B^T(A^T)^{-1}.
$$

This is an intrinsic readout: its constructor receives the period matrix,
harmonic matrix and boundary orientations, but no target operator. A new
logical input $x\in V$ is attached after relaxation and produces
$T_Wx$.

The target is used only in the fitting objective. Instead of one selected
state, this later period experiment pins the complete basis

$$
r_U(W)=\sum_{j=1}^{2}
  r_{\mathrm{period}}\bigl(e_j,Ue_j;W\bigr).
$$

The two basis equations determine a linear map on $V$. A single equation
does not.

The causal prism has spacelike boundary edges and timelike free interior
edges. The boundary is held bit-identical while an admissibility-gated line
search descends

$$
\Phi(W)=
\left\|\frac{\partial S_{\mathrm{Regge}}}
                 {\partial \ell^2_{\mathrm{free}}}\right\|^2
+\Gamma r_U(W).
$$

The gradient uses the exact complex Regge Hessian and the existing analytic
period-residual gradient. The experiment separately records the hard period
gap

$$
r_{\mathrm{gap}}=
\sum_j\min_h\|\operatorname{periods}(h)-(e_j,Ue_j)\|^2.
$$

It has the same finite-geometry zero set as the period residual but cannot
appear to improve merely because the non-harmonic leaked representative is
rescaled.

## Status of the geometric claims

The constructed $W$ is an admissible simplicial cobordism between two copies
of the register surface. With input $B$ and output $A$, its oriented
boundary has the standard form

$$
\partial W=\overline{\Sigma_B}\sqcup\Sigma_A,
$$

where $\sqcup$ is disjoint union. The code then places state data on these
fixed components as harmonic-period constraints. It does not construct or prove
a state-to-geometry map $\Sigma_\psi=\operatorname{geo}(\psi)$, nor an
operator-to-cobordism map $W=\operatorname{geo}(U)$. Those identifications
remain conjectural. What is established here is narrower: once a graph-like
$W$ is frozen, its two boundary restrictions intrinsically determine a linear
transport $T_W$.

## Frozen amplitude readout

Transport alone is not H3. In the input period chart the frozen geometry also
determines the anchor-normalized Gram matrix $G_W$; the first training-basis
state is the fixed anchor. The actual amplitude operator is

$$
(G_W)_{ij}=s_1h_i^\dagger W_1h_j,
\qquad
\mathcal A_W=(T_W^\dagger)^{-1}G_W,
\qquad
Z_W(q,x)=\langle q|\mathcal A_W|x\rangle .
$$

Thus $Z_W(q,x)=\langle q|T_W|x\rangle$ requires both unitary transport and
an isometric chart, $G_W=I$. The experiment reports these as independent
conditions.

## Reproduction

From the repository root:

    python examples/cobordism/intrinsic_operator_transfer.py
    python -m pytest tests/cobordism/test_intrinsic_operator_transfer.py -q

The script writes the complete JSON record to
/tmp/cobordism/intrinsic_operator_transfer.json by default. Raw output is not
committed.

## Result

The table is the deterministic default run: $\Gamma=1$, 12 bounded
relaxation steps, and 16 unseen complex unit inputs per case.

| target | $\|[U,Q]\|_F$ | final $r_U$ | hard gap | $\|T_W-U\|_F$ | max unseen-state error | $\|\mathcal A_W-U\|_F$ |
|---|---:|---:|---:|---:|---:|---:|
| identity | 0 | $2.85\times10^{-27}$ | $1.76\times10^{-28}$ | $6.09\times10^{-15}$ | $6.08\times10^{-15}$ | 0.203659 |
| mapping-class three-cycle | 0 | $4.86\times10^{-27}$ | $2.02\times10^{-28}$ | $9.48\times10^{-15}$ | $8.62\times10^{-15}$ | 0.203659 |
| generic charge-preserving SU(2) block | $1.36\times10^{-16}$ | $8.38\times10^{-3}$ | 0.135345 | 0.520279 | 0.367893 | 1.44187 |
| charge-leaking unitary | 0.589544 | 2.23710 | 0.177925 | 0.091034 projected | 0.423477 | 0.262576 |

For the generic charge-preserving target, relaxation reduced the period
residual from 2.04983 to 0.00838458. Nevertheless:

- the hard period gap remained 0.1353453088;
- the frozen transport error remained 0.5202793649;
- unseen inputs failed with maximum error 0.367893;
- the boundary drift was exactly zero and all accepted geometries remained
  Lorentzian-admissible.

The reduced residual therefore does not represent operator learning. The
prism's boundary restriction remains the graph of the identity throughout
metric relaxation.

The one-state control makes the identification error explicit. The logical
reflection $\operatorname{diag}(1,-1)$ fixes the selected first basis state,
giving a one-state hard gap of $3.51\times10^{-29}$, while the complete
basis gives gap 2.0 and rejects it.

Finally, the causal staircase charts are not isometric after relaxation:
$\|G_W-I\|_F=0.203659$ for both positive transport controls. Removing
the arbitrary overall scale by trace normalization still leaves shape error
0.200212. Consequently the intrinsic transport generalizes correctly, but the
H3 amplitude equality is not certified on this substrate.

## Scientific conclusion

This experiment supports three restricted statements:

1. a frozen cobordism can intrinsically encode identity and mapping-class
   transport;
2. complete-basis pinning makes that claim testable on unseen states;
3. charge leakage is an obstruction.

It does **not** support either of the proposed sufficient conditions:

- charge conservation alone does not make a generic unitary realizable on the
  fixed prism topology;
- stationary-action relaxation does not by itself preserve the isometric
  chart required by H3.

The run exhausted its 12-step budget rather than solving the free-edge Regge
stationarity equations. That does not affect the fixed-topology transfer
finding: $T_W$ is determined by the boundary restriction of the same
cohomology class and remained invariant while the metric-dependent residual
changed. It does mean this run is not a global no-go theorem over other
topologies, singular strata, or enlarged geometric fields.

A universal paper therefore still requires a topology-changing construction
whose frozen intrinsic readout covers a continuous charge-preserving family,
plus an isometry or Gram-correction theorem. The present result is suitable as
a rigorous negative/control section of that paper, not as evidence for the
universal headline.
