# Orientation and content branches as local systems

## Why the orientation covector is not a metric weight

For adjacent top cells (a,b) sharing a facet, let (s_a,s_b\in\{\pm1\})
be their canonical boundary coefficients at that facet. Coherent induced
orientations obey

\[
  \varepsilon_b=g_{ab}\varepsilon_a,
  \qquad g_{ab}=-s_as_b.
\]

On an orientable connected complex, `orientationCovector()` chooses one
trivialization (arepsilon), unique up to a global sign. Multiplying the
diagonal Hodge weight by that representative directly would make the operator
depend on the arbitrary global sign. The invariant datum is instead the flat
(\mathbb Z_2) connection (g). A local basis change (q_a\in\{\pm1\}) acts as

\[
  f_a\mapsto q_af_a,
  \qquad g_{ab}\mapsto q_ag_{ab}q_b.
\]

The dual covariant graph Laplacian

\[
  (L_g)_{aa}=\deg(a),
  \qquad (L_g)_{ab}=-g_{ab}
\]

then transforms by similarity, (L_g\mapsto QL_gQ), so its spectrum is gauge
independent. `OrientationLocalSystem` stores this connection, a deterministic
spanning-forest trivialization, and the residual non-tree holonomies. The old
`orientationCovector()` is recovered exactly when every holonomy is (+1).
When a loop has product (-1), the complex is retained and reported as
non-orientable rather than rejected.

`HodgeLaplacian::orientationConnectionLaplacian()` exposes this isolated dual
sector. It does not change the existing primal metric Hodge operator or the
joint objective. A complete production integration would replace the relevant
ordinary cochain differential by a differential with local coefficients; raw
signing of (W_k) is not that construction.

## Square-root continuation

`Simplex::volume()` returns the pointwise principal value

\[
  V=\frac{\sqrt{\det G}}{d!}.
\]

The principal value is discontinuous across its cut even when the geometry
moves continuously. `ContentBranchTracker` instead lifts a sequence of
**accepted** geometries to the double cover: for a surviving cell it chooses
the member of ({V,-V}) nearest the preceding lift. A newly created cell is
seeded from the canonical orientation-local-system trivialization.

A local root flip is accompanied by the same (\mathbb Z_2) gauge change on
all incident connection links. Covariant spectra and Wilson-loop products are
therefore invariant across a principal-cut crossing. If (det G) winds once
around zero, the continued (V) returns with the opposite sign; only after two
windings does it return to the original sheet. Squared content
(V^2=\det G/(d!)^2) is blind to this monodromy.

The tracker must not be mutated by line-search probes or rejected optimizer
trials. Doing that would make the objective depend on evaluation order and
invalidate its analytic gradient. It is updated only after a step is accepted;
an objective that eventually uses the lift should consume an immutable
snapshot captured at that acceptance boundary.

## Aharonov--Bohm analogy

The loop observable

\[
  \mathcal W(\gamma)=\prod_{(ab)\in\gamma}g_{ab}\in\{+1,-1\}
\]

is the discrete (\mathbb Z_2) Wilson loop. Every link can be gauged to (+1)
locally (and on a spanning tree), yet a non-contractible loop may retain
(\mathcal W=-1). This is the same global mechanism as the Aharonov--Bohm
effect: a locally flat connection can still alter interference and the
covariant spectrum through nontrivial holonomy. Here the only nontrivial phase
is (-1=e^{i\pi}), so the analogy is an Aharonov--Bohm phase of (pi), not a
claim that the orientation connection is the electromagnetic field.

The two-sheeted content lift has the same structure in determinant space. A
loop around the branch point (det G=0) has (mathbb Z_2) monodromy and flips
(V); (V^2) is the even representation and cannot see that sign. This is the
precise sense in which the observed (W=V) and (W=V^2) geometries can respond
differently to parity/orientation transport.
